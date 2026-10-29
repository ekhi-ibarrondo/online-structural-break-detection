"""Score the local test set of the challenge and print its TS-AUC.

The data comes from the competition (``crunch setup`` downloads it); it is not part of
this repository. ``--data-dir`` must contain X_test.reduced.parquet and y_test.reduced.parquet.

usage: python scripts/evaluate.py --data-dir data [--weights weights] [--limit N] [--save scores.npz]
"""

from __future__ import annotations

import argparse
import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import breakdetect  # noqa: E402
from breakdetect.metrics import ts_auc  # noqa: E402


def load_series(data_dir: Path):
    """(id, history, online, labels) per series, in id order; period 1 is the history, 2 the online part."""
    x = pd.read_parquet(data_dir / "X_test.reduced.parquet")
    y = pd.read_parquet(data_dir / "y_test.reduced.parquet")
    for sid, g in x.groupby(level="id", sort=True):
        g = g.sort_index(level="time")
        history = g.loc[g["period"] == 1, "value"].to_numpy(np.float64)
        online = g.loc[g["period"] == 2, "value"]
        labels = y.loc[sid].reindex(online.index.get_level_values("time"))["target"].to_numpy()
        yield sid, history, online.to_numpy(np.float64), labels


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--data-dir", type=Path, required=True)
    parser.add_argument("--weights", type=Path, default=None)
    parser.add_argument("--limit", type=int, default=None, help="score only the first N series")
    parser.add_argument("--save", type=Path, default=None, help="save the scores of every point (npz)")
    args = parser.parse_args()

    t0 = time.perf_counter()
    model = breakdetect.load_model(args.weights)
    print(f"model loaded in {time.perf_counter() - t0:.1f}s")
    ids, scores, labels = [], [], []
    t1 = time.perf_counter()
    for i, (sid, history, online, lab) in enumerate(load_series(args.data_dir)):
        if args.limit is not None and i >= args.limit:
            break
        ids.append(sid)
        scores.append(model.score(history, online))
        labels.append(lab)
    n_points = sum(len(s) for s in scores)
    print(f"{len(scores)} series, {n_points} points scored in {time.perf_counter() - t1:.1f}s")
    value = ts_auc(scores, labels)
    print(f"TS-AUC = {value!r} ({100.0 * value:.4f} points)")
    if args.save is not None:
        np.savez(args.save, ids=np.asarray(ids), scores=np.concatenate(scores), labels=np.concatenate(labels))


if __name__ == "__main__":
    main()
