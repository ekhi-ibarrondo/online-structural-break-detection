"""Build the float-grid member: raw membership scores of every training point and their CDF map.

    python -m float_grid.build_map --data-dir "$DATA_DIR" --out-dir "$OUT_DIR/float_grid"

The member has no supervised training: a discrete grid of float values is
fitted on each series' history and every online point is scored causally by
its membership log-likelihood ratio.  Only ``X_train.parquet`` is read (no
labels).  The per-step CDF map is fitted on the raw scores of all 10,000
series (``N_MIN = 500``).  Outputs: ``model_full_float_grid_s0/float_grid.json``
(fixed constants), ``maps_float_grid_train10k_s0.npz`` and the raw scores.

Exact reproduction requires Linux/glibc libm (``log`` is called from the
compiled kernels) and the pinned numba/llvmlite pair.
"""

import argparse
import hashlib
import json
import os
import platform
from pathlib import Path
from time import perf_counter

import numpy as np
import pyarrow as pa
import pyarrow.parquet as pq

from common import step_maps

from . import grid
from .grid.config import write_model_config

# sha256 of the int64 little-endian ids of the 10,000 training series, ascending.
TRAIN_IDS_SHA256 = "9e1c19b3fdc185411bd1a987deb6a9fda2531116aac060a84c4d878854d1c099"


def sha256(path):
    digest = hashlib.sha256()
    with Path(path).open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def read_series(path):
    """Read only the feature table, ordered by integer id and time; never open labels."""
    table = pq.ParquetFile(path).read(columns=["id", "time", "value", "period"], use_threads=False)
    if table.schema.field("value").type != pa.float32():
        raise ValueError("Require float32 values")
    ids, times, values, periods = [table[k].to_numpy() for k in ("id", "time", "value", "period")]
    if not np.issubdtype(ids.dtype, np.integer) or not np.issubdtype(times.dtype, np.integer):
        raise ValueError("Series identifiers and times must be integers")
    order = np.lexsort((times, ids))
    ids, times, values, periods = (a[order] for a in (ids, times, values, periods))
    if np.any((ids[1:] == ids[:-1]) & (times[1:] == times[:-1])):
        raise ValueError("Require unique rows")
    bounds = np.r_[0, np.flatnonzero(ids[1:] != ids[:-1]) + 1, len(ids)] if len(ids) else [0]
    for lo, hi in zip(bounds[:-1], bounds[1:]):
        part = periods[lo:hi]
        n = int(np.count_nonzero(part == 1))
        if not np.all(part[:n] == 1) or not np.all(part[n:] == 2):
            raise ValueError("Each series must contain history followed by online rows")
        yield ids[lo], values[lo:lo + n], values[lo + n:hi]


def build(args):
    started = perf_counter()
    source = args.data_dir / "X_train.parquet"
    args.out_dir.mkdir(parents=True, exist_ok=True)
    chunks, times, ids, lengths = [], [], [], []
    detected = 0
    modes = {"support": 0, "band": 0, "none": 0}
    for identifier, history, online in read_series(source):
        if not isinstance(identifier, (int, np.integer)):
            raise ValueError("Series identifiers must be integers")
        ids.append(identifier)
        lengths.append(len(online))
        chunks.append(grid.batch_series(history, online))
        times.append(np.arange(len(online), dtype=np.int64))
        summary = grid.fit_summary(history)
        detected += int(summary["detected"])
        modes[summary["mode"]] += 1
    ids = np.asarray(ids, dtype=np.int64)
    lengths = np.asarray(lengths, dtype=np.int64)
    ids_hash = hashlib.sha256(ids.astype("<i8").tobytes()).hexdigest()
    if not args.allow_other_ids and ids_hash != TRAIN_IDS_SHA256:
        raise ValueError("Series identifier digest does not match the 10,000 training series")
    if not chunks:
        raise ValueError("Input contains no series")
    raw, times = np.concatenate(chunks), np.concatenate(times)
    np.save(args.out_dir / "raw_scores.npy", raw)
    np.savez(args.out_dir / "raw_index.npz", ids=ids, lengths=lengths, times=times)
    score_seconds = perf_counter() - started
    maps = step_maps.fit_step_maps(raw, times, args.n_min)
    map_path = args.out_dir / args.map_name
    step_maps.save_step_maps(map_path, maps)
    loaded = step_maps.load_step_maps(map_path)
    assert maps.values.tobytes() == loaded.values.tobytes()
    assert maps.offsets.tobytes() == loaded.offsets.tobytes()
    rng = np.random.default_rng(58)
    rows = rng.choice(len(raw), size=min(1000, len(raw)), replace=False)
    evaluated = step_maps.evaluate_rows(loaded, raw[rows], times[rows])
    assert np.all((evaluated > 0.0) & (evaluated < 1.0))
    config_path = args.out_dir / "model_full_float_grid_s0" / "float_grid.json"
    write_model_config(config_path)
    report = {
        "series": len(ids),
        "points": len(raw),
        "detected": detected,
        "modes": modes,
        "ids_sha256": ids_hash,
        "sha256": {p.relative_to(args.out_dir).as_posix(): sha256(p)
                   for p in (args.out_dir / "raw_scores.npy", args.out_dir / "raw_index.npz", map_path, config_path)},
        "input_sha256": sha256(source),
        "labels_read": False,
        "python": platform.python_version(),
        "numpy": np.__version__,
        "pyarrow": pa.__version__,
        "runtime": grid.runtime_report(),
        "seconds": {"scores": score_seconds, "total": perf_counter() - started},
    }
    (args.out_dir / "report.json").write_text(
        json.dumps(report, sort_keys=True, indent=1) + "\n", encoding="utf-8", newline="\n"
    )
    print(json.dumps({key: report[key] for key in ("series", "points", "detected", "ids_sha256")}, sort_keys=True))
    return report


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--data-dir", type=Path, default=os.environ.get("DATA_DIR"))
    parser.add_argument("--out-dir", type=Path, required=True)
    parser.add_argument("--map-name", default="maps_float_grid_train10k_s0.npz")
    parser.add_argument("--n-min", type=int, default=step_maps.N_MIN)
    parser.add_argument("--allow-other-ids", action="store_true", help="skip the training-ids check (smoke tests)")
    args = parser.parse_args()
    if args.data_dir is None:
        parser.error("set --data-dir or DATA_DIR")
    args.data_dir = Path(args.data_dir)
    if Path(args.map_name).name != args.map_name or not args.map_name.endswith(".npz"):
        parser.error("map-name must be a local .npz filename")
    if args.n_min < 1:
        parser.error("n-min must be positive")
    build(args)


if __name__ == "__main__":
    main()
