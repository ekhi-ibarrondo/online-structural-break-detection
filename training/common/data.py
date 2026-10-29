"""Loader for the competition training data.

pandas/pyarrow are used only here (imported lazily inside the loader).
Series are returned in ascending id order; every training script relies on
that order.

Layout of the parquet files:

- ``X_*.parquet``: MultiIndex (id, time), columns ``value`` (float32) and
  ``period`` (int64; 1 = historical, 2 = online).
- ``y_*.parquet``: MultiIndex (id, time), column ``target`` (0/1) — the exact
  per-step online label (1 = the break already occurred at/before that time).
- ``y_*_index.parquet``: index id, columns ``tau_index`` (0-indexed break
  position within the online segment; -1 = no break) and ``tau``.
"""

from __future__ import annotations

import hashlib
import inspect
import pickle
from dataclasses import dataclass, field
from pathlib import Path

import numpy as np

__all__ = ["Series", "load_train"]

_CACHE_VERSION = 2
_DEFAULT_CACHE_DIR = Path(__file__).resolve().parent.parent / "cache"


def _fingerprint(paths: list[Path]) -> list[tuple[str, int, int]]:
    """(resolved path, mtime_ns, size) per source file — cache invalidation key."""
    out: list[tuple[str, int, int]] = []
    for p in paths:
        st = p.stat()
        out.append((str(p.resolve()), int(st.st_mtime_ns), int(st.st_size)))
    return out


@dataclass
class Series:
    """One series: fixed historical segment + point-by-point online segment."""

    id: int
    # float64; no break by construction.  The loader never rescales.
    x_historical: np.ndarray
    x_online: np.ndarray  # float64, revealed point by point
    tau_index: int | None = None  # 0-indexed break position in online; None = no break
    labels: np.ndarray | None = None  # int8 per online step, len == len(x_online)
    meta: dict | None = field(default=None, repr=False)


def _load_split(
    x_path: Path, y_path: Path, y_index_path: Path
) -> list[Series]:
    import pandas as pd  # pandas/pyarrow allowed only in the loader

    x_df = pd.read_parquet(x_path)
    if list(x_df.index.names) != ["id", "time"]:
        raise ValueError(f"{x_path.name}: expected MultiIndex (id, time), got {x_df.index.names}")
    if not x_df.index.is_monotonic_increasing:
        x_df = x_df.sort_index()
    ids = x_df.index.get_level_values("id").to_numpy()
    times = x_df.index.get_level_values("time").to_numpy()
    values = x_df["value"].to_numpy(dtype=np.float64)
    if not np.isfinite(values).all():
        raise ValueError(f"{x_path.name}: non-finite values in 'value' column")
    period = x_df["period"].to_numpy(dtype=np.int64)

    y_df = pd.read_parquet(y_path)
    if list(y_df.index.names) != ["id", "time"]:
        raise ValueError(f"{y_path.name}: expected MultiIndex (id, time), got {y_df.index.names}")
    if not y_df.index.is_monotonic_increasing:
        y_df = y_df.sort_index()
    y_ids = y_df.index.get_level_values("id").to_numpy()
    y_times = y_df.index.get_level_values("time").to_numpy()
    targets = y_df["target"].to_numpy(dtype=np.int8)

    yi_df = pd.read_parquet(y_index_path)
    tau_by_id = dict(
        zip(
            yi_df.index.to_numpy().tolist(),
            yi_df["tau_index"].to_numpy().tolist(),
            strict=True,
        )
    )

    uids, first = np.unique(ids, return_index=True)
    x_bounds = np.append(first, ids.shape[0])
    y_uids, y_first = np.unique(y_ids, return_index=True)
    y_bounds = np.append(y_first, y_ids.shape[0])
    if not np.array_equal(uids, y_uids):
        raise ValueError("X and y cover different series ids")

    series: list[Series] = []
    for k in range(uids.shape[0]):
        sid = int(uids[k])
        a, b = int(x_bounds[k]), int(x_bounds[k + 1])
        per = period[a:b]
        split = int(np.searchsorted(per, 2))
        if not (np.all(per[:split] == 1) and np.all(per[split:] == 2)):
            raise ValueError(f"series {sid}: period column is not 1..1,2..2 in time order")
        x_hist = values[a : a + split].copy()
        x_online = values[a + split : b].copy()
        if x_hist.size == 0 or x_online.size == 0:
            raise ValueError(f"series {sid}: empty historical or online segment")

        ya, yb = int(y_bounds[k]), int(y_bounds[k + 1])
        labels = targets[ya:yb].copy()
        if labels.shape[0] != x_online.shape[0]:
            raise ValueError(
                f"series {sid}: len(labels)={labels.shape[0]} != "
                f"len(x_online)={x_online.shape[0]}"
            )
        if not np.array_equal(times[a + split : b], y_times[ya:yb]):
            raise ValueError(
                f"series {sid}: online time index of X does not match y"
            )
        tau_raw = int(tau_by_id[sid])
        tau_index = None if tau_raw < 0 else tau_raw
        series.append(
            Series(
                id=sid,
                x_historical=x_hist,
                x_online=x_online,
                tau_index=tau_index,
                labels=labels,
            )
        )
    return series


# Hash of the loader source: a changed loader invalidates the cache.
_LOADER_HASH = hashlib.sha256(inspect.getsource(_load_split).encode()).hexdigest()


def _load_cached(
    data_dir: str | Path,
    x_name: str,
    y_name: str,
    y_index_name: str,
    cache_name: str,
    cache_dir: str | Path | None,
    use_cache: bool,
) -> list[Series]:
    data_dir = Path(data_dir)
    cache_path = Path(cache_dir) if cache_dir is not None else _DEFAULT_CACHE_DIR
    cache_file = cache_path / cache_name
    sources = [data_dir / x_name, data_dir / y_name, data_dir / y_index_name]
    fingerprint = _fingerprint(sources)
    loader_hash = _LOADER_HASH
    if use_cache and cache_file.exists():
        # The cache is written only by this module from the local parquet
        # files; never point ``cache_dir`` at files you did not create.
        with open(cache_file, "rb") as f:
            payload = pickle.load(f)
        if (
            payload.get("version") == _CACHE_VERSION
            and payload.get("files") == fingerprint
            and payload.get("loader") == loader_hash
        ):
            return payload["series"]
    series = _load_split(*sources)
    if use_cache:
        cache_path.mkdir(parents=True, exist_ok=True)
        with open(cache_file, "wb") as f:
            pickle.dump(
                {
                    "version": _CACHE_VERSION,
                    "files": fingerprint,
                    "loader": loader_hash,
                    "series": series,
                },
                f,
                protocol=5,
            )
    return series


def load_train(
    data_dir: str | Path,
    cache_dir: str | Path | None = None,
    use_cache: bool = True,
) -> list[Series]:
    """Load the real training series (cached to ``cache/train.pkl``)."""
    return _load_cached(
        data_dir,
        "X_train.parquet",
        "y_train.parquet",
        "y_train_index.parquet",
        "train.pkl",
        cache_dir,
        use_cache,
    )

