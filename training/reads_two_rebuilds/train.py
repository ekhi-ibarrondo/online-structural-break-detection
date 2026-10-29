"""Train the reads_two_rebuilds member (GRU head over 354 streaming features).

    python -m features.extract --data-dir "$DATA_DIR" --out-dir "$OUT_DIR/features"
    F="$OUT_DIR/features"; O="$OUT_DIR/reads_two_rebuilds"
    python -m reads_two_rebuilds.train fit-full  --features-dir "$F" --out-dir "$O"   # GPU
    python -m reads_two_rebuilds.train fit-oof   --features-dir "$F" --out-dir "$O"   # GPU
    python -m reads_two_rebuilds.train score-oof --features-dir "$F" --out-dir "$O" --workers 8
    python -m reads_two_rebuilds.train map       --features-dir "$F" --out-dir "$O"
    python -m reads_two_rebuilds.train head      --out-dir "$O"

* ``fit-full``: 3 learners (b = 0, 1, 2) on all 10,000 series, recipe seed k = 5;
  transform statistics over all rows.
  -> ``model_full_reads_two_rebuilds_s0/learner_s0_k5_b{b}.npz``.
* ``fit-oof``: the same recipe per fold of ``fold_masks(10000, 0, 5)`` with
  k = 7 (statistics from the fold's training rows).  -> ``oof_weights/``.
* ``score-oof``: out-of-fold logits of every point with the 3 learners of its
  fold, computed on the CPU in float64 with fp16-quantised inputs (the
  inference arithmetic).  -> ``oof_full_reads_two_rebuilds_s0.npy``.
* ``map``: per-step CDF map of those logits.  -> ``maps_reads_two_rebuilds_train10k_s0.npz``.
* ``head``: the member descriptor ``model_full_reads_two_rebuilds_s0/head.json``.

GPU training is not bit-reproducible (cuDNN GRU kernels, float16 table); the
recipe, seeds and data are fixed.
"""
from __future__ import annotations

import os

os.environ.setdefault('CUBLAS_WORKSPACE_CONFIG', ':4096:8')

import argparse  # noqa: E402
import hashlib  # noqa: E402
import json  # noqa: E402
import multiprocessing as mp  # noqa: E402
import time  # noqa: E402
from pathlib import Path  # noqa: E402

import numpy as np  # noqa: E402

from common.data import load_train  # noqa: E402
from common.folds import fold_masks  # noqa: E402
from common.step_maps import N_MIN, fit_step_maps, load_step_maps, save_step_maps  # noqa: E402

from . import model as G  # noqa: E402

S_RECIPE = 0
K_FULL, K_OOF = 5, 7
BAGS = 3
FOLDS = 5
BLOCK = 48
CHUNK = 65536
LAYOUT = Path(__file__).resolve().parents[1] / 'reads_three_others' / 'layout.json'


def log(msg: str) -> None:
    print(f"[{time.strftime('%H:%M:%S')}] {msg}", flush=True)


def weights_name(k: int, b: int, fold: int | None = None) -> str:
    if fold is None:
        return f'learner_s{S_RECIPE}_k{k}_b{b}.npz'
    return f'learner_s{S_RECIPE}_k{k}_f{fold}_b{b}.npz'


def sha_file(p: Path) -> str:
    return hashlib.sha256(Path(p).read_bytes()).hexdigest()


class Pop:
    """Rows of the 10,000 series (ascending ids) as laid out by ``features.extract``."""

    def __init__(self, features_dir: Path, data_dir: Path | None):
        with np.load(features_dir / 'index.npz') as z:
            self.ids, self.n_on, self.offs = z['ids'], z['lengths'], z['offsets']
        self.n_series = len(self.ids)
        self.n_points = int(self.offs[-1])
        self.TT = np.concatenate([np.arange(n) for n in self.n_on]).astype(np.int64)
        self.rows_of = [np.arange(self.offs[i], self.offs[i + 1]) for i in range(self.n_series)]
        self.src = {nm: np.load(features_dir / f'{nm}.npy', mmap_mode='r') for nm in G.SRC_ORDER}
        self.Y = None
        if data_dir is not None:
            series = load_train(data_dir, cache_dir=features_dir / 'cache')
            by_id = {int(s.id): s for s in series}
            Y = []
            for i, n in zip(self.ids.tolist(), self.n_on.tolist(), strict=True):
                lab = np.asarray(by_id[i].labels, np.int64)
                assert len(lab) == n, f'series {i}: online length differs from the feature arrays'
                Y.append(lab)
            self.Y = np.concatenate(Y)

    def rows_of_series(self, idx) -> np.ndarray:
        return np.concatenate([self.rows_of[i] for i in idx])

    def X32(self, i: int) -> np.ndarray:
        a, b = int(self.offs[i]), int(self.offs[i + 1])
        return np.concatenate([np.asarray(self.src[nm][a:b]) for nm in G.SRC_ORDER], axis=1)


def compute_stats(pop: Pop, row_sets: dict) -> dict:
    """{name: training rows} -> {name: (median (354,), scale (354,))}, one pass per block of 48 columns."""
    out = {k: ([], []) for k in row_sets}
    t0 = time.time()
    for nm in G.SRC_ORDER:
        c = G.NCOL[nm]
        med = {k: np.full(c, np.nan) for k in row_sets}
        sc = {k: np.full(c, np.nan) for k in row_sets}
        for jb0 in range(0, c, BLOCK):
            jb = list(range(jb0, min(c, jb0 + BLOCK)))
            blk = np.asarray(pop.src[nm][:, jb0:jb[-1] + 1], dtype=np.float32)
            for jj, j in enumerate(jb):
                col = np.ascontiguousarray(blk[:, jj])
                for k, rows in row_sets.items():
                    med[k][j], sc[k][j] = G.col_stats(col, rows)
            del blk
            log(f'statistics {nm}: columns {jb[0]}..{jb[-1]} ({time.time() - t0:.0f}s)')
        for k in row_sets:
            assert np.isfinite(med[k]).all() and np.isfinite(sc[k]).all() and (sc[k] > 0).all(), (nm, k)
            out[k][0].append(med[k])
            out[k][1].append(sc[k])
    return {k: (np.concatenate(m).astype(np.float64), np.concatenate(s).astype(np.float64))
            for k, (m, s) in out.items()}


def build_table(pop: Pop, med: np.ndarray, sc: np.ndarray, dev):
    """Transformed float16 table on the GPU (one padding row of zeros at the end)."""
    import torch

    feat = torch.zeros((pop.n_points + 1, G.N_FEAT), dtype=torch.float16, device=dev)
    for a in range(0, pop.n_points, CHUNK):
        b = min(pop.n_points, a + CHUNK)
        X = np.concatenate([np.asarray(pop.src[nm][a:b]) for nm in G.SRC_ORDER], axis=1)
        Z = G.transform(X, med, sc)
        feat[a:b] = torch.from_numpy(Z).to(dev).to(torch.float16)
    return feat


def train_jobs(pop: Pop, jobs: list) -> None:
    """jobs: dicts {k, fold, tr_idx, stats (med, sc), wdir}; 3 learners per job, one table per job."""
    import torch

    G.set_determinism()
    dev = torch.device('cuda')
    torch.cuda.set_per_process_memory_fraction(0.75, 0)
    data = G.Data(pop.n_on, pop.offs[:-1], dev)
    for jb in jobs:
        tgt, w = G.target_y(pop.TT, pop.Y, pop.rows_of, jb['tr_idx'])
        data.clear()
        med, sc = jb['stats']
        data.set_table(build_table(pop, med, sc, dev), tgt, w)
        for b in range(BAGS):
            path = Path(jb['wdir']) / weights_name(jb['k'], b, jb['fold'])
            if path.exists():
                log(f'{path.name} exists, skipping')
                continue
            net, info = G.train_one(data, S_RECIPE, jb['k'], b, jb['tr_idx'], pop.rows_of, G.E_MAX, dev, log)
            meta = dict(s=S_RECIPE, k=jb['k'], b=b, fold=jb['fold'], n_train_series=int(len(jb['tr_idx'])),
                        torch=torch.__version__, **info)
            G.save_learner(path, G.state_dict_np(net), med, sc, meta)
            log(f'k{jb["k"]} fold={jb["fold"]} b{b}: best epoch {info["best_ep"]}, inner loss '
                f'{info["best_inner"]:.5f} ({info["secs"]:.0f}s) -> {path.name}')
            del net
            torch.cuda.empty_cache()
    data.clear()


def cmd_fit_full(a) -> None:
    pop = Pop(a.features_dir, a.data_dir)
    tr = np.arange(pop.n_series)
    stats = compute_stats(pop, {'full': pop.rows_of_series(tr)})['full']
    wdir = a.out_dir / 'model_full_reads_two_rebuilds_s0'
    train_jobs(pop, [dict(k=K_FULL, fold=None, tr_idx=tr, stats=stats, wdir=wdir)])


def cmd_fit_oof(a) -> None:
    pop = Pop(a.features_dir, a.data_dir)
    masks = fold_masks(pop.n_series, 0, FOLDS)
    stats = compute_stats(pop, {f: pop.rows_of_series(np.flatnonzero(~masks[f])) for f in range(FOLDS)})
    jobs = [dict(k=K_OOF, fold=f, tr_idx=np.flatnonzero(~masks[f]), stats=stats[f], wdir=a.out_dir / 'oof_weights')
            for f in range(FOLDS)]
    train_jobs(pop, jobs)


_W: dict = {}


def _winit(features_dir: str, wdir: str) -> None:
    _W['pop'] = Pop(Path(features_dir), None)
    _W['bags'] = {f: [G.load_learner(Path(wdir) / weights_name(K_OOF, b, f)) for b in range(BAGS)]
                  for f in range(FOLDS)}


def _wjob(item):
    i, f = item
    return i, G.series_logit(_W['pop'].X32(i), _W['bags'][f])


def cmd_score_oof(a) -> None:
    pop = Pop(a.features_dir, None)
    masks = fold_masks(pop.n_series, 0, FOLDS)
    fold_of = np.full(pop.n_series, -1)
    for f in range(FOLDS):
        fold_of[masks[f]] = f
    oof = np.full(pop.n_points, np.nan)
    items = [(i, int(fold_of[i])) for i in range(pop.n_series)]
    t0 = time.time()
    with mp.get_context('spawn').Pool(a.workers, initializer=_winit,
                                      initargs=(str(a.features_dir), str(a.out_dir / 'oof_weights'))) as pool:
        for n, (i, lg) in enumerate(pool.imap_unordered(_wjob, items, chunksize=8), 1):
            oof[pop.offs[i]:pop.offs[i + 1]] = lg
            if n % 500 == 0:
                log(f'scored {n}/{pop.n_series} series ({time.time() - t0:.0f}s)')
    assert np.isfinite(oof).all()
    np.save(a.out_dir / 'oof_full_reads_two_rebuilds_s0.npy', oof)


def cmd_map(a) -> None:
    pop = Pop(a.features_dir, None)
    oof = np.load(a.out_dir / 'oof_full_reads_two_rebuilds_s0.npy')
    assert oof.dtype == np.float64 and oof.shape == (pop.n_points,) and np.all(np.isfinite(oof))
    maps = fit_step_maps(oof, pop.TT, N_MIN)
    path = a.out_dir / 'maps_reads_two_rebuilds_train10k_s0.npz'
    save_step_maps(path, maps)
    again = load_step_maps(path)
    assert np.array_equal(again.values, maps.values) and np.array_equal(again.offsets, maps.offsets)
    log(f'map -> {path}')


def cmd_head(a) -> None:
    mdir = a.out_dir / 'model_full_reads_two_rebuilds_s0'
    bags = [weights_name(K_FULL, b) for b in range(BAGS)]
    head = dict(format='reads_two_rebuilds_head/v1', input_set='reads_two_rebuilds', quant='fp16', s=S_RECIPE,
                k=K_FULL, sources=list(G.SRC_ORDER), bags=bags,
                reduction='logit = ((0.0 + l_b0) + l_b1) + l_b2) / 3, before the one map',
                trace_names=json.loads(LAYOUT.read_text())['trace_names'],
                weights_sha256={n: sha_file(mdir / n) for n in bags})
    (mdir / 'head.json').write_text(json.dumps(head, indent=2) + '\n', encoding='utf-8')


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument('step', choices=['fit-full', 'fit-oof', 'score-oof', 'map', 'head'])
    ap.add_argument('--data-dir', type=Path, default=os.environ.get('DATA_DIR'))
    ap.add_argument('--features-dir', type=Path)
    ap.add_argument('--out-dir', type=Path, required=True)
    ap.add_argument('--workers', type=int, default=4)
    a = ap.parse_args()
    if a.step != 'head' and a.features_dir is None:
        ap.error('--features-dir is required')
    if a.step in ('fit-full', 'fit-oof'):
        if a.data_dir is None:
            ap.error('set --data-dir or DATA_DIR (labels)')
        a.data_dir = Path(a.data_dir)
    a.out_dir.mkdir(parents=True, exist_ok=True)
    {'fit-full': cmd_fit_full, 'fit-oof': cmd_fit_oof, 'score-oof': cmd_score_oof, 'map': cmd_map,
     'head': cmd_head}[a.step](a)


if __name__ == '__main__':
    main()
