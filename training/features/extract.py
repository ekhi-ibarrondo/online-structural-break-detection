"""Streaming input features of reads_two_rebuilds and reads_three_others for every point of the training series.

    python -m features.extract --data-dir "$DATA_DIR" --out-dir "$OUT_DIR/features" --workers 8

Each series is replayed point by point exactly as at inference time (causal,
no access to the online length).  Per point, 468 float32 columns in seven
blocks, each written to its own array ``<block>.npy`` of shape (n_points, width):

    surprises 182 | pattern_removed 110 | summary 36 | traces 22 | context 4 | slow_and_fast 96 | order_not_size 18

reads_two_rebuilds reads the first five blocks (354 columns); reads_three_others
reads all seven.  ``index.npz`` holds the series ids (ascending), online lengths
and row offsets.  Size on disk: about 9.4 GB for the full training set.
"""
from __future__ import annotations

import os

for _key in ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS", "NUMBA_NUM_THREADS"):
    os.environ[_key] = "1"
os.environ["NUMBA_CPU_NAME"] = "generic"
os.environ["NUMBA_DISABLE_INTEL_SVML"] = "1"

import argparse  # noqa: E402
import json  # noqa: E402
import multiprocessing as mp  # noqa: E402
import time  # noqa: E402
from pathlib import Path  # noqa: E402

import numpy as np  # noqa: E402

from common.data import load_train  # noqa: E402

LAYOUT = json.loads((Path(__file__).resolve().parents[1] / "reads_three_others" / "layout.json").read_text())
BLOCKS = (("surprises", 182), ("pattern_removed", 110), ("summary", 36), ("traces", 22), ("context", 4),
          ("slow_and_fast", 96), ("order_not_size", 18))
WIDTH = sum(w for _, w in BLOCKS)
assert WIDTH == LAYOUT["width"] == 468


class Extractor:
    """Per-series feature state of the seven blocks."""

    def __init__(self):
        from common.inference_features import StreamInputs
        from order_not_size.scorer import Scorer
        from slow_and_fast.features import FeatureEngine

        self.inputs = StreamInputs()
        self.trace_keys = tuple(n[2:] for n in LAYOUT["trace_names"])
        self.engine = FeatureEngine()
        self.columns = Scorer(None)

    def series(self, history, online):
        from pattern_removed.features import FeatureState
        from surprises.features import StreamState, fit_hist, stream_step

        inputs = self.inputs
        h = np.asarray(history, dtype=np.float64)
        rows = np.empty((len(online), WIDTH), dtype=np.float32)
        summary = inputs.summary(h)
        traces = inputs.traces(h, self.trace_keys)
        context = np.asarray(inputs.context(h), np.float32)
        engine_state = self.engine.start(h)
        column_state = self.columns.start(h)
        sstate, pstate = StreamState(fit_hist(h)), FeatureState(h)
        for t, value in enumerate(online):
            x = float(value)
            r = rows[t]
            r[:182] = stream_step(sstate, x)
            r[182:292] = np.asarray(pstate.step(x), np.float32)
            r[292:328] = np.asarray(summary.step(x), np.float32)
            r[328:350] = np.asarray(traces.step(x), np.float64).astype(np.float32)
            r[350:354] = context
            r[354:450] = self.engine.step_features(engine_state, x).astype(np.float32)
            r[450:468] = self.columns.step_columns(column_state, x).astype(np.float32)
            if not np.isfinite(r).all():
                raise ValueError("non-finite feature")
        return rows


_W: dict = {}


def _winit(paths: dict) -> None:
    _W["ex"] = Extractor()
    _W["mm"] = {k: np.load(p, mmap_mode="r+") for k, p in paths.items()}


def _wjob(job):
    for a, history, online in job:
        rows = _W["ex"].series(history, online)
        b = a + rows.shape[0]
        c = 0
        for name, w in BLOCKS:
            _W["mm"][name][a:b] = rows[:, c:c + w]
            c += w
    for m in _W["mm"].values():
        m.flush()
    return len(job)


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--data-dir", type=Path, default=os.environ.get("DATA_DIR"))
    ap.add_argument("--out-dir", type=Path, required=True)
    ap.add_argument("--cache-dir", type=Path, default=None, help="loader cache (default: <out-dir>/cache)")
    ap.add_argument("--workers", type=int, default=4)
    ap.add_argument("--limit", type=int, default=0, help="smoke test: first N series only")
    a = ap.parse_args()
    if a.data_dir is None:
        ap.error("set --data-dir or DATA_DIR")
    series = load_train(Path(a.data_dir), cache_dir=a.cache_dir or a.out_dir / "cache")
    if a.limit:
        series = series[:a.limit]
    ids = np.asarray([s.id for s in series], dtype=np.int64)
    lengths = np.asarray([len(s.x_online) for s in series], dtype=np.int64)
    offsets = np.concatenate(([0], np.cumsum(lengths))).astype(np.int64)
    n = int(offsets[-1])
    a.out_dir.mkdir(parents=True, exist_ok=True)
    paths = {}
    for name, w in BLOCKS:
        p = a.out_dir / f"{name}.npy"
        mm = np.lib.format.open_memmap(p, mode="w+", dtype=np.float32, shape=(n, w))
        mm[:] = np.nan
        mm.flush()
        del mm
        paths[name] = str(p)
    jobs, cur = [], []
    for i, s in enumerate(series):
        cur.append((int(offsets[i]), s.x_historical, s.x_online))
        if len(cur) == 10:
            jobs.append(cur)
            cur = []
    if cur:
        jobs.append(cur)
    del series
    t0 = time.time()
    done = 0
    with mp.get_context("spawn").Pool(a.workers, initializer=_winit, initargs=(paths,)) as pool:
        for k in pool.imap_unordered(_wjob, jobs):
            done += k
            if done % 500 < k:
                print(f"{done}/{len(ids)} series, {time.time() - t0:.0f}s", flush=True)
    for name, _ in BLOCKS:
        if not np.isfinite(np.load(paths[name], mmap_mode="r")).all():
            raise RuntimeError(f"{name}: missing or non-finite rows")
    np.savez(a.out_dir / "index.npz", ids=ids, lengths=lengths, offsets=offsets)
    print(json.dumps({"series": len(ids), "points": n, "seconds": round(time.time() - t0, 1)}))


if __name__ == "__main__":
    main()
