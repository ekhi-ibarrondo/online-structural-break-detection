"""Compiled evaluation of a flat LightGBM forest on one float32-rounded feature vector."""

from __future__ import annotations

import numba as nb
import numpy as np

from .runtime import kernel

_I = nb.types.Array(nb.int32, 1, "C", readonly=True)
_F = nb.types.Array(nb.float64, 1, "C", readonly=True)
# LightGBM treats |x| <= 1e-35f as zero before descending the trees (float32 constant).
ZERO_THRESHOLD = float(np.float32(1e-35))


@kernel(nb.float64(_F, _I, _F, _I, _I, _F, _I))
def forest_one(x, feat, threshold, left, right, leaf, roots):
    """Sum of the leaf values reached by ``x`` in every tree, in tree order."""
    score = 0.0
    for root in roots:
        node = root
        while left[node] != node:
            v = x[feat[node]]
            if abs(v) <= ZERO_THRESHOLD:
                v = 0.0
            if v <= threshold[node]:
                node = left[node]
            else:
                node = right[node]
        score += leaf[node]
    return score
