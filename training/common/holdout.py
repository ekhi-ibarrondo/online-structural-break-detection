"""Internal 2,000-series holdout and the 8,000-series "search" split.

The three sequence networks (broken_or_not, which_one_broke, blind_to_size)
were trained on the 8,000 training series that are NOT in a fixed internal
holdout of 2,000 series (the holdout was reserved for model selection).  The
holdout is a deterministic
stratified draw: strata = label (break / no break) x quintile of the
historical length, proportional allocation with largest remainder, one
``numpy`` generator seeded with ``HOLDOUT_SEED`` consumed in stratum order.

Usage::

    python -m common.holdout --data-dir "$DATA_DIR" --out-dir "$OUT_DIR/search_split"

writes ``holdout_ids.json`` and the three ``*.search.parquet`` files that
``networks/`` reads, after checking the search ids against ``SEARCH_IDS_SHA256``.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
from pathlib import Path

import numpy as np

__all__ = ["HOLDOUT_SEED", "N_HOLDOUT", "SEARCH_IDS_SHA256", "build_holdout_ids", "ids_sha256"]

N_HOLDOUT = 2000
HOLDOUT_SEED = 20260821
_QUANTILES = (0.2, 0.4, 0.6, 0.8)
# sha256 of the comma-joined ascending ids of the 8,000 search series.
SEARCH_IDS_SHA256 = "c06d4fc1515165211df478a69b5ec792725e8ea659404b85f668dc90504285c5"


def ids_sha256(ids) -> str:
    return hashlib.sha256(",".join(str(int(i)) for i in ids).encode("ascii")).hexdigest()


def build_holdout_ids(
    ids: np.ndarray,
    n_hist: np.ndarray,
    has_break: np.ndarray,
    n_holdout: int = N_HOLDOUT,
    seed: int = HOLDOUT_SEED,
) -> list[int]:
    """Deterministic stratified draw; returns the sorted holdout id list.

    Strata: label (0/1) x n_hist quintile (edges = np.quantile at
    ``_QUANTILES`` over the full population).  Allocation: proportional with
    largest remainder (ties broken by stratum order: label asc, quintile
    asc), summing to exactly ``n_holdout``.  Within each stratum: ids sorted
    ascending, then ``rng.permutation`` (one generator consumed in stratum order).
    """
    ids = np.asarray(ids)
    n_hist = np.asarray(n_hist, dtype=np.float64)
    label = (np.asarray(has_break)).astype(np.int64)
    n = ids.shape[0]
    if not (0 < n_holdout < n):
        raise ValueError(f"n_holdout={n_holdout} out of range for {n} series")
    if len(set(ids.tolist())) != n:
        raise ValueError("duplicate series ids")
    edges = np.quantile(n_hist, _QUANTILES)
    qbin = np.searchsorted(edges, n_hist, side="right")  # 0..4
    strata = [(lab, qb) for lab in (0, 1) for qb in range(5)]
    masks = {key: (label == key[0]) & (qbin == key[1]) for key in strata}
    sizes = {key: int(m.sum()) for key, m in masks.items()}
    if sum(sizes.values()) != n:
        raise RuntimeError("strata do not partition the population")
    # largest-remainder allocation
    quota = {key: n_holdout * sizes[key] / n for key in strata}
    alloc = {key: int(np.floor(quota[key])) for key in strata}
    leftover = n_holdout - sum(alloc.values())
    order = sorted(
        range(len(strata)),
        key=lambda i: (-(quota[strata[i]] - alloc[strata[i]]), i),
    )
    for i in order[:leftover]:
        alloc[strata[i]] += 1
    rng = np.random.default_rng(seed)
    chosen: list[int] = []
    for key in strata:
        members = np.sort(ids[masks[key]])
        k = alloc[key]
        if k > members.shape[0]:
            raise RuntimeError(f"stratum {key} smaller than its allocation")
        perm = rng.permutation(members.shape[0])
        chosen.extend(int(x) for x in members[perm[:k]])
    if len(chosen) != n_holdout or len(set(chosen)) != n_holdout:
        raise RuntimeError("allocation did not produce n_holdout unique ids")
    return sorted(chosen)


def series_summary(data_dir: Path) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """(ids, historical length, has_break) of every training series, ids ascending."""
    import pyarrow.parquet as pq

    x = pq.read_table(data_dir / "X_train.parquet", columns=["id", "period"])
    xid = x.column("id").to_numpy().astype(np.int64)
    hist = x.column("period").to_numpy() == 1
    ids, inv = np.unique(xid, return_inverse=True)
    n_hist = np.bincount(inv, weights=hist.astype(np.float64)).astype(np.int64)
    yi = pq.read_table(data_dir / "y_train_index.parquet", columns=["id", "tau_index"])
    tau = dict(zip(yi.column("id").to_numpy().tolist(), yi.column("tau_index").to_numpy().tolist(), strict=True))
    has_break = np.array([int(tau[int(i)]) >= 0 for i in ids])
    return ids, n_hist, has_break


def write_search_split(data_dir: Path, out_dir: Path) -> dict:
    import pyarrow.parquet as pq

    ids, n_hist, has_break = series_summary(data_dir)
    hold = build_holdout_ids(ids, n_hist, has_break)
    search = np.setdiff1d(ids, np.asarray(hold, np.int64))
    got = ids_sha256(search)
    if got != SEARCH_IDS_SHA256:
        raise SystemExit(f"search ids sha256 {got} != expected {SEARCH_IDS_SHA256}")
    out_dir.mkdir(parents=True, exist_ok=True)
    keep = [int(i) for i in search]
    for name in ("X_train", "y_train", "y_train_index"):
        table = pq.read_table(data_dir / f"{name}.parquet", filters=[("id", "in", keep)])
        pq.write_table(table, out_dir / f"{name}.search.parquet")
    payload = {"seed": HOLDOUT_SEED, "n": len(hold), "ids_sha256": ids_sha256(hold), "ids": hold}
    (out_dir / "holdout_ids.json").write_text(json.dumps(payload, indent=1) + "\n", encoding="utf-8")
    return {"n_search": len(keep), "n_holdout": len(hold), "search_ids_sha256": got}


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--data-dir", type=Path, default=os.environ.get("DATA_DIR"), required="DATA_DIR" not in os.environ)
    ap.add_argument("--out-dir", type=Path, required=True)
    a = ap.parse_args()
    print(json.dumps(write_search_split(Path(a.data_dir), a.out_dir)))


if __name__ == "__main__":
    main()
