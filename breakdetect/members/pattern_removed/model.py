"""Frozen pattern_removed learner: per-cell PIT of the detector columns, then gradient-boosted trees."""

from __future__ import annotations

import numpy as np

from .features import pit_logit


class NumpyTrees:
    """Fixed-depth trees stored as (n_trees, nodes) arrays; probability by a clipped logistic."""

    def __init__(self, path) -> None:
        with np.load(path, allow_pickle=False) as q:
            self.feature = q["feature"]
            self.threshold = q["threshold"]
            self.left = q["left"]
            self.right = q["right"]
            self.value = q["value"]
            self.depth = int(q["depth"])
        self.rows = np.arange(len(self.feature))

    def predict_one(self, x):
        """Probability of one PIT-transformed feature vector."""
        nodes = np.zeros(len(self.rows), dtype=np.int64)
        for _ in range(self.depth):
            r = self.rows
            f = self.feature[r, nodes]
            nodes = np.where(x[f] <= self.threshold[r, nodes], self.left[r, nodes], self.right[r, nodes])
        raw = float(np.sum(self.value[self.rows, nodes]))
        return 1 / (1 + np.exp(-max(-700.0, min(700.0, raw))))


class FoldPIT:
    """Quantile knots per (online step bucket, history length bucket) for the 18 detector columns."""

    def __init__(self, path) -> None:
        with np.load(path, allow_pickle=False) as q:
            self.knots = q["knots"]
            self.hist_edges = q["hist_edges"]
            self.t_edges = q["t_edges"]

    def apply_one(self, x, t, n_hist):
        """Replace columns 92.. by their PIT logit in the cell of (t, n_hist), in float32."""
        ti = int(np.searchsorted(self.t_edges, t, side="left"))
        hi = int(np.searchsorted(self.hist_edges, n_hist, side="right"))
        x = np.asarray(x, dtype=np.float32).astype(np.float64)
        x[92:] = pit_logit(x[92:], self.knots[ti, hi])
        return x.astype(np.float32)
