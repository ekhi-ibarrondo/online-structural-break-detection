"""Train reads_three_others: one LightGBM on the 468 streaming features, its tree export and its anchor CDF map.

    python -m features.extract --data-dir "$DATA_DIR" --out-dir "$OUT_DIR/features"
    python -m reads_three_others.train --data-dir "$DATA_DIR" --features-dir "$OUT_DIR/features" \\
        --out-dir "$OUT_DIR/reads_three_others"

Rows: every 16th online step (t = 0, 16, 32, ...) of the 10,000 training series
(319,490 rows); label = 1 once the break has occurred.  No weights.  LightGBM
4.7.0, 350 rounds, fixed parameters below (two threads, deterministic mode).
Outputs: ``model_reads_three_others/`` (feat/thr/left/right/value/roots/meta .npy
+ ``head.json``), ``map_reads_three_others.npz`` (anchor CDF of the training raw
scores) and the native LightGBM model ``model.txt``.

The original fit ran on Windows; LightGBM histogram construction is not
guaranteed to give bit-identical trees on another platform.
"""
from __future__ import annotations

import os

for _key in ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS", "NUMEXPR_NUM_THREADS"):
    os.environ[_key] = "1"

import argparse  # noqa: E402
import hashlib  # noqa: E402
import json  # noqa: E402
import time  # noqa: E402
from pathlib import Path  # noqa: E402

import numpy as np  # noqa: E402

from .anchor_maps import cdf_maps, load_anchor_cdf, save_anchor_cdf  # noqa: E402
from .export import export_booster, save_model  # noqa: E402

PARAMS = dict(objective='binary', metric='None', learning_rate=.04, num_leaves=31,
              min_data_in_leaf=300, max_bin=63, num_threads=2, deterministic=True,
              force_col_wise=True, seed=2509695, feature_fraction=.9, lambda_l2=10.,
              verbosity=-1, histogram_pool_size=64, bin_construct_sample_cnt=50000)
ROUNDS = 350
STRIDE = 16
BLOCKS = (("surprises", 182), ("pattern_removed", 110), ("summary", 36), ("traces", 22), ("context", 4),
          ("slow_and_fast", 96), ("order_not_size", 18))
LAYOUT = Path(__file__).resolve().parent / "layout.json"


def array_sha(a):
    a = np.ascontiguousarray(a)
    return hashlib.sha256(memoryview(a).cast('B')).hexdigest()


def load_tau(data_dir: Path) -> dict:
    """tau_index per series id (-1 = no break)."""
    import pyarrow.parquet as pq

    t = pq.read_table(data_dir / "y_train_index.parquet", columns=["id", "tau_index"])
    return dict(zip(t.column("id").to_numpy().tolist(), t.column("tau_index").to_numpy().tolist(), strict=True))


def build_rows(features_dir: Path, taus: dict):
    """Stride-16 rows of every series, in ascending id order: x (float32, C order), y, t, ids."""
    with np.load(features_dir / "index.npz") as z:
        ids, lengths, offsets = z["ids"], z["lengths"], z["offsets"]
    assert np.all(np.diff(ids) > 0)
    blocks = [np.load(features_dir / f"{name}.npy", mmap_mode="r") for name, _ in BLOCKS]
    n_rows = int(np.sum((lengths + STRIDE - 1) // STRIDE))
    x = np.empty((n_rows, 468), dtype=np.float32, order='C')
    y = np.empty(n_rows, np.int8)
    times = np.empty(n_rows, np.int16)
    row_ids = np.empty(n_rows, np.int64)
    j = 0
    for sid, length, off in zip(ids, lengths, offsets[:-1], strict=True):
        t = np.arange(0, int(length), STRIDE, dtype=np.int16)
        n = len(t)
        rows = int(off) + t.astype(np.int64)
        x[j:j + n] = np.concatenate([b[rows] for b in blocks], axis=1)
        tau = int(taus[int(sid)])
        times[j:j + n], row_ids[j:j + n] = t, sid
        y[j:j + n] = ((t >= tau) & (tau >= 0)).astype(np.int8)
        j += n
    assert j == n_rows and x.flags.c_contiguous and x.dtype == np.float32
    return x, y, times, row_ids, ids


def run(data_dir: Path, features_dir: Path, out: Path) -> dict:
    import lightgbm as lgb

    if lgb.__version__ != '4.7.0':
        print(f"warning: LightGBM {lgb.__version__} (the original fit used 4.7.0)", flush=True)
    out.mkdir(parents=True, exist_ok=False)
    start = time.perf_counter()
    x, y, times, row_ids, ids = build_rows(features_dir, load_tau(data_dir))
    n_rows = len(y)
    training_feature_sha = array_sha(x)
    dataset = lgb.Dataset(x, label=y, weight=None, free_raw_data=True)
    booster = lgb.train(dict(PARAMS), dataset, num_boost_round=ROUNDS)
    assert booster.current_iteration() == ROUNDS
    booster.save_model(str(out / 'model.txt'))
    native = booster.predict(x, raw_score=True, num_threads=2)
    exported = export_booster(booster, 468)
    save_model(exported, out / 'model_reads_three_others')
    layout = json.loads(LAYOUT.read_text())
    (out / 'model_reads_three_others' / 'head.json').write_text(json.dumps(layout, indent=2) + '\n', encoding='utf8')
    maps = cdf_maps(np.asarray(native, np.float64), times)
    del dataset, x, y, native
    save_anchor_cdf(out / 'map_reads_three_others.npz', maps)
    load_anchor_cdf(out / 'map_reads_three_others.npz')
    np.save(out / 'training_ids.npy', ids)
    np.savez(out / 'stride_rows.npz', ids=row_ids, times=times)
    report = dict(params=PARAMS, rounds=ROUNDS, stride=STRIDE, training_rows=n_rows,
                  training_feature_array_sha256=training_feature_sha,
                  feature_blocks=[dict(name=n, width=w, dtype='float32') for n, w in BLOCKS],
                  lightgbm=lgb.__version__, numpy=np.__version__, seconds=time.perf_counter() - start)
    (out / 'report.json').write_text(json.dumps(report, indent=2) + '\n', encoding='utf8')
    print(json.dumps({'training_rows': n_rows, 'seconds': round(report['seconds'], 1)}), flush=True)
    return report


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument('--data-dir', type=Path, default=os.environ.get('DATA_DIR'))
    parser.add_argument('--features-dir', type=Path, required=True)
    parser.add_argument('--out-dir', type=Path, required=True)
    args = parser.parse_args()
    if args.data_dir is None:
        parser.error('set --data-dir or DATA_DIR')
    run(Path(args.data_dir), args.features_dir, args.out_dir)


if __name__ == '__main__':
    main()
