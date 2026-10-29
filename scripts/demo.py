"""Score a few synthetic series (no competition data needed) and print their TS-AUC.

usage: python scripts/demo.py [--series 12] [--seed 0] [--weights weights]
"""

from __future__ import annotations

import argparse
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import breakdetect  # noqa: E402
from breakdetect.metrics import labels_from_break, ts_auc  # noqa: E402
from breakdetect.synthetic import generate  # noqa: E402


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--series", type=int, default=12)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--weights", type=Path, default=None)
    args = parser.parse_args()

    series = generate(n_series=args.series, seed=args.seed, online_range=(50, 300))
    t0 = time.perf_counter()
    model = breakdetect.load_model(args.weights)
    print(f"model loaded in {time.perf_counter() - t0:.1f}s")
    scores, labels = [], []
    for s in series:
        scores.append(model.score(s.history, s.online))
        labels.append(labels_from_break(s.break_index, len(s.online)))
        print(f"{s.kind:>10}  break at {s.break_index!s:>5}  last score {scores[-1][-1]:.3f}")
    value = ts_auc(scores, labels)
    print(f"TS-AUC = {value:.4f} ({100.0 * value:.2f} points) on {len(series)} synthetic series")


if __name__ == "__main__":
    main()
