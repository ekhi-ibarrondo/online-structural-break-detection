"""Shared driver for the members fitted on the 10,000 training series (surprises, pattern_removed,
slow_and_fast, order_not_size).

Every member goes through the same four steps, each one a subcommand of the
member's ``train`` module::

    python -m surprises.train features            # per-step input features of all series
    python -m surprises.train oof --seed 0        # 5-fold out-of-fold scores, fold_masks(10000, seed, 5)
    python -m surprises.train fit --seed 0        # model fitted on all 10,000 series
    python -m surprises.train map --seed 0        # per-step CDF map fitted on the out-of-fold scores

Series are always in ascending id order (the loader order).  Paths come from
``--data-dir`` / ``--out-dir`` or the ``DATA_DIR`` / ``OUT_DIR`` environment
variables.  BLAS is pinned to one thread: the fits are deterministic on a
fixed platform.
"""
from __future__ import annotations

import os

for _key in ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS", "NUMEXPR_NUM_THREADS",
             "VECLIB_MAXIMUM_THREADS"):
    os.environ.setdefault(_key, "1")

import argparse  # noqa: E402
import importlib  # noqa: E402
import json  # noqa: E402
import multiprocessing as mp  # noqa: E402
import shutil  # noqa: E402
import time  # noqa: E402
from pathlib import Path  # noqa: E402

import numpy as np  # noqa: E402

from common.data import load_train  # noqa: E402
from common.folds import fold_masks  # noqa: E402
from common.step_maps import N_MIN, fit_step_maps, load_step_maps, save_step_maps  # noqa: E402

FOLDS = 5
N_SERIES = 10_000


def log(msg: str) -> None:
    print(f"[{time.strftime('%Y-%m-%dT%H:%M:%SZ', time.gmtime())}] {msg}", flush=True)


def write_json(path: Path, obj) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(obj, indent=1, default=_json_default) + "\n", encoding="utf-8")


def _json_default(o):
    if isinstance(o, np.integer):
        return int(o)
    if isinstance(o, np.floating):
        return float(o)
    if isinstance(o, np.ndarray):
        return o.tolist()
    if isinstance(o, Path):
        return str(o)
    raise TypeError(f"not serialisable: {type(o)}")


# ---------------------------------------------------------------------------------------------- population
class Population:
    """The training series in loader order, with per-point arrays.

    ``t0`` is the 0-based online step, ``row_series`` the series index of each
    point and ``labels`` the per-step target (int8).
    """

    def __init__(self, series: list):
        self.series = series
        self.n_series = len(series)
        self.ids = np.asarray([s.id for s in series], dtype=np.int64)
        self.lengths = np.asarray([len(s.x_online) for s in series], dtype=np.int64)
        self.n_hist = np.asarray([len(s.x_historical) for s in series], dtype=np.int64)
        self.offsets = np.concatenate(([0], np.cumsum(self.lengths))).astype(np.int64)
        self.n_points = int(self.offsets[-1])
        self.t0 = np.concatenate([np.arange(n, dtype=np.int64) for n in self.lengths])
        self.row_series = np.repeat(np.arange(self.n_series, dtype=np.int64), self.lengths)
        self.labels = np.concatenate([np.asarray(s.labels, dtype=np.int8) for s in series])


def load_population(data_dir: Path, cache_dir: Path, limit: int = 0) -> Population:
    series = load_train(data_dir, cache_dir=cache_dir)
    if not limit and len(series) != N_SERIES:
        raise SystemExit(f"expected {N_SERIES} training series, found {len(series)}")
    return Population(series[:limit] if limit else series)


# ---------------------------------------------------------------------------------------------- features
_W: dict = {}


def _winit(module_name: str, paths: dict) -> None:
    mod = importlib.import_module(module_name)
    _W["fn"] = mod.make_extractor()
    _W["mm"] = {k: np.load(p, mmap_mode="r+") for k, p in paths.items()}


def _wjob(job):
    fn, mm = _W["fn"], _W["mm"]
    for a, x_hist, x_online in job:
        b = a + x_online.shape[0]
        out = fn(x_hist, x_online)
        for name, arr in out.items():
            mm[name][a:b] = arr
    for m in mm.values():
        m.flush()
    return len(job)


def extract_features(module_name: str, outputs: dict, pop: Population, feat_dir: Path, workers: int) -> None:
    """Run ``module.make_extractor()`` on every series, in parallel, into one memmap per output."""
    feat_dir.mkdir(parents=True, exist_ok=True)
    paths = {}
    for name, (ncol, dtype) in outputs.items():
        p = feat_dir / f"{name}.npy"
        mm = np.lib.format.open_memmap(p, mode="w+", dtype=dtype, shape=(pop.n_points, ncol))
        mm[:] = np.nan
        mm.flush()
        del mm
        paths[name] = str(p)
    jobs, cur = [], []
    for i, s in enumerate(pop.series):
        cur.append((int(pop.offsets[i]), np.asarray(s.x_historical, dtype=np.float64),
                    np.asarray(s.x_online, dtype=np.float64)))
        if len(cur) == 10:
            jobs.append(cur)
            cur = []
    if cur:
        jobs.append(cur)
    t0 = time.time()
    done = 0
    with mp.get_context("spawn").Pool(workers, initializer=_winit, initargs=(module_name, paths)) as pool:
        for n in pool.imap_unordered(_wjob, jobs):
            done += n
            if done % 1000 < n:
                log(f"features: {done}/{pop.n_series} series, {time.time() - t0:.0f}s")
    for name in outputs:
        a = np.load(paths[name], mmap_mode="r")
        if not np.all(np.isfinite(a)):
            raise RuntimeError(f"{name}: non-finite or missing features")
    write_json(feat_dir / "meta.json", {"ids": pop.ids, "n_points": pop.n_points, "outputs": list(outputs),
                                        "seconds": time.time() - t0})


def load_features(feat_dir: Path, outputs: dict, pop: Population) -> dict:
    meta = json.loads((feat_dir / "meta.json").read_text(encoding="utf-8"))
    if meta["ids"] != pop.ids.tolist():
        raise SystemExit(f"{feat_dir}: features were extracted for another population")
    return {name: np.load(feat_dir / f"{name}.npy", mmap_mode="r") for name in outputs}


# ---------------------------------------------------------------------------------------------- member interface
class Member:
    """Interface implemented by each member's ``train`` module."""

    name = ""
    module = ""                     # module that defines ``make_extractor()``
    outputs: dict = {}              # feature arrays: name -> (columns, dtype)
    model_seeds = (0, 1, 2)         # seeds that get a full-data model

    def learner(self, k: int, f: int):
        raise NotImplementedError

    def fit(self, pop: Population, F: dict, train_mask: np.ndarray, k: int, f: int, work: Path):
        raise NotImplementedError

    def predict(self, model, pop: Population, F: dict, rows: np.ndarray) -> np.ndarray:
        raise NotImplementedError

    def export(self, model, out_dir: Path, meta: dict) -> None:
        raise NotImplementedError

    def cleanup(self, model) -> None:
        wd = model.get("wd") if isinstance(model, dict) else None
        if wd is not None and Path(wd).exists():
            shutil.rmtree(wd, ignore_errors=True)


def run_oof(m: Member, pop: Population, F: dict, k: int, out: Path) -> Path:
    """5-fold out-of-fold scores with ``fold_masks(n_series, k, 5)`` and learner ``m.learner(k, f)``."""
    masks = fold_masks(pop.n_series, k, FOLDS)
    oof = np.full(pop.n_points, np.nan, dtype=np.float64)
    for f in range(FOLDS):
        t0 = time.time()
        train_mask = ~masks[f]
        model = m.fit(pop, F, train_mask, k, f, out / "work" / f"oof_s{k}_f{f}")
        rows = np.flatnonzero(masks[f][pop.row_series])
        oof[rows] = m.predict(model, pop, F, rows)
        m.cleanup(model)
        log(f"{m.name} oof s{k} fold {f}: learner {m.learner(k, f)}, {int(train_mask.sum())} train series, "
            f"{rows.size} test rows, {time.time() - t0:.0f}s")
    if not np.all(np.isfinite(oof)):
        raise RuntimeError("incomplete or non-finite out-of-fold scores")
    path = out / f"oof_{m.name}_s{k}.npy"
    np.save(path, oof)
    return path


def run_fit(m: Member, pop: Population, F: dict, k: int, out: Path) -> Path:
    """Model fitted on every series with learner ``m.learner(k, 0)``."""
    t0 = time.time()
    model = m.fit(pop, F, np.ones(pop.n_series, dtype=bool), k, 0, out / "work" / f"fit_s{k}")
    mdir = out / f"model_full_{m.name}_s{k}"
    if mdir.exists():
        shutil.rmtree(mdir)
    meta = {"member": m.name, "seed": k, "learner": m.learner(k, 0), "n_train_series": pop.n_series,
            "fit_seconds": time.time() - t0}
    m.export(model, mdir, meta)
    m.cleanup(model)
    log(f"{m.name} full model s{k} -> {mdir} ({time.time() - t0:.0f}s)")
    return mdir


def run_map(m: Member, pop: Population, k: int, out: Path) -> Path:
    """Per-step CDF map of the out-of-fold scores (all rows, ``N_MIN = 500``)."""
    oof = np.load(out / f"oof_{m.name}_s{k}.npy")
    if oof.shape != (pop.n_points,) or not np.all(np.isfinite(oof)):
        raise SystemExit("out-of-fold scores missing, incomplete or for another population")
    maps = fit_step_maps(oof, pop.t0, N_MIN)
    path = out / f"maps_{m.name}_train10k_s{k}.npz"
    save_step_maps(path, maps)
    again = load_step_maps(path)
    if not (np.array_equal(again.values, maps.values) and np.array_equal(again.offsets, maps.offsets)):
        raise RuntimeError(f"{path}: reloaded map differs from the fitted one")
    return path


def main(m: Member) -> None:
    ap = argparse.ArgumentParser(description=f"Train the {m.name} member.")
    ap.add_argument("step", choices=["features", "oof", "fit", "map", "all"])
    ap.add_argument("--seed", type=int, default=0, help="protocol seed k (folds and learner seed)")
    ap.add_argument("--data-dir", type=Path, default=os.environ.get("DATA_DIR"))
    ap.add_argument("--out-dir", type=Path, default=os.environ.get("OUT_DIR"))
    ap.add_argument("--workers", type=int, default=4, help="processes for feature extraction")
    ap.add_argument("--limit", type=int, default=0, help="smoke test: first N series only")
    a = ap.parse_args()
    if a.data_dir is None or a.out_dir is None:
        ap.error("set --data-dir/--out-dir or DATA_DIR/OUT_DIR")
    out = Path(a.out_dir) / m.name
    out.mkdir(parents=True, exist_ok=True)
    pop = load_population(Path(a.data_dir), Path(a.out_dir) / "cache", a.limit)
    feat_dir = out / "features"
    steps = ["features", "oof", "fit", "map"] if a.step == "all" else [a.step]
    seeds = (0, 1, 2) if a.step == "all" else (a.seed,)
    if "features" in steps:
        extract_features(m.module, m.outputs, pop, feat_dir, a.workers)
    F = load_features(feat_dir, m.outputs, pop) if set(steps) & {"oof", "fit"} else None
    for k in seeds:
        if "oof" in steps:
            log(f"{m.name}: out-of-fold scores -> {run_oof(m, pop, F, k, out)}")
        if "fit" in steps and (a.step == "fit" or k in m.model_seeds):
            run_fit(m, pop, F, k, out)
        if "map" in steps:
            log(f"{m.name}: map -> {run_map(m, pop, k, out)}")
