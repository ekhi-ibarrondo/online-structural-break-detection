"""Series-level fold assignment used by every cross-validated fit."""
from __future__ import annotations

import numpy as np

__all__ = ["fold_masks"]


def fold_masks(n_series: int, seed: int, folds: int) -> list[np.ndarray]:
    rng = np.random.default_rng(seed)
    perm = rng.permutation(n_series)
    out = []
    for f in range(folds):
        m = np.zeros(n_series, dtype=bool)
        m[perm[f::folds]] = True
        out.append(m)
    return out
