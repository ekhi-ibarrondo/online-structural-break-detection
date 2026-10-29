"""Flat LightGBM forest of the reads_three_others member, validated at load time."""

from __future__ import annotations

from pathlib import Path

import numpy as np

from ...kernels.forest import forest_one

ARRAY_NAMES = ("feat", "thr", "left", "right", "value", "roots", "meta")


def validate_model(model: dict) -> None:
    """Reject malformed graphs before they reach the compiled (bounds-check free) kernel."""
    if set(model) != set(ARRAY_NAMES):
        raise ValueError("forest array names")
    for key, a in model.items():
        dtype = np.int64 if key == "meta" else np.float64 if key in ("thr", "value") else np.int32
        if not isinstance(a, np.ndarray) or a.dtype != dtype or a.ndim != 1:
            raise ValueError(f"forest dtype/shape: {key}")
        if not np.isfinite(a).all():
            raise ValueError(f"forest nonfinite: {key}")
    meta = model["meta"]
    if meta.shape != (3,) or meta[0] < 1 or meta[1] < 0 or meta[2] < 1:
        raise ValueError("forest meta")
    n = len(model["feat"])
    roots = model["roots"]
    if any(len(model[k]) != n for k in ARRAY_NAMES[:5]) or len(roots) != meta[0]:
        raise ValueError("forest lengths")
    if n == 0 or any(np.any((model[k] < 0) | (model[k] >= n)) for k in ("roots", "left", "right")):
        raise ValueError("forest child/root bounds")
    visited, max_depth = set(), 0
    for root in roots:
        stack = [(int(root), 0)]
        while stack:
            i, depth = stack.pop()
            if i in visited:
                raise ValueError("forest cycle/shared node/duplicate root")
            visited.add(i)
            li, ri = int(model["left"][i]), int(model["right"][i])
            if li == i:
                if ri != i:
                    raise ValueError("forest leaf")
                max_depth = max(max_depth, depth)
            else:
                if not 0 <= model["feat"][i] < meta[2] or ri == i or li == ri:
                    raise ValueError("forest feature/children")
                stack.extend(((ri, depth + 1), (li, depth + 1)))
    if len(visited) != n or max_depth != meta[1]:
        raise ValueError("forest unreachable node/depth")


class Forest:
    """Read-only forest arrays; inputs are rounded to float32 like LightGBM does."""

    def __init__(self, model: dict) -> None:
        validate_model(model)
        self.model = {k: np.array(v, copy=True, order="C") for k, v in model.items()}
        for a in self.model.values():
            a.flags.writeable = False
        self.n_features = int(self.model["meta"][2])
        self.arrays = tuple(self.model[k] for k in ARRAY_NAMES[:6])

    @classmethod
    def load(cls, directory) -> Forest:
        """Read the seven node arrays from ``directory``."""
        return cls({k: np.load(Path(directory) / f"{k}.npy", allow_pickle=False) for k in ARRAY_NAMES})

    def predict_one(self, x) -> float:
        """Raw score of one feature vector (rounded to float32 first)."""
        a = np.asarray(x)
        if a.ndim != 1 or a.shape[-1] != self.n_features:
            raise ValueError("forest feature shape")
        with np.errstate(over="ignore", invalid="ignore"):
            a = np.ascontiguousarray(a, dtype=np.float32).astype(np.float64)
        if not np.isfinite(a).all():
            raise ValueError("forest requires finite inputs after the float32 cast")
        return float(forest_one(a, *self.arrays))
