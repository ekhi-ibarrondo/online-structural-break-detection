"""Model pieces of pattern_removed: per-cell PIT of the detector columns, numpy tree ensemble and its exporter."""

from __future__ import annotations
import numpy as np
from .features import PROBS, pit_logit


class NumpyTrees:
    def __init__(self, path):
        with np.load(path, allow_pickle=False) as q:
            self.feature = q["feature"]
            self.threshold = q["threshold"]
            self.left = q["left"]
            self.right = q["right"]
            self.value = q["value"]
            self.depth = int(q["depth"])
        self.rows = np.arange(len(self.feature))

    def predict_one(self, x):
        nodes = np.zeros(len(self.rows), dtype=np.int64)
        for _ in range(self.depth):
            r = self.rows
            f = self.feature[r, nodes]
            nodes = np.where(x[f] <= self.threshold[r, nodes], self.left[r, nodes], self.right[r, nodes])
        raw = float(np.sum(self.value[self.rows, nodes]))
        return 1 / (1 + np.exp(-max(-700.0, min(700.0, raw))))

    def predict_batch(self, x, chunk=4096):
        out = np.empty(len(x), dtype=np.float64)
        for start in range(0, len(x), chunk):
            xx = x[start : start + chunk]
            nodes = np.zeros((len(xx), len(self.rows)), dtype=np.int64)
            rr = self.rows[None, :]
            points = np.arange(len(xx))[:, None]
            for _ in range(self.depth):
                ff = self.feature[rr, nodes]
                nodes = np.where(
                    xx[points, ff] <= self.threshold[rr, nodes], self.left[rr, nodes], self.right[rr, nodes]
                )
            raw = np.sum(self.value[rr, nodes], axis=1)
            out[start : start + len(xx)] = 1 / (1 + np.exp(-np.clip(raw, -700, 700)))
        return out


class FoldPIT:
    def __init__(self, path):
        with np.load(path, allow_pickle=False) as q:
            self.knots = q["knots"]
            self.hist_edges = q["hist_edges"]
            self.t_edges = q["t_edges"]

    def apply_one(self, x, t, n_hist):
        ti = int(np.searchsorted(self.t_edges, t, side="left"))
        hi = int(np.searchsorted(self.hist_edges, n_hist, side="right"))
        x = np.asarray(x, dtype=np.float32).astype(np.float64)
        x[92:] = pit_logit(x[92:], self.knots[ti, hi])
        return x.astype(np.float32)

    def apply_batch(self, x, t, n_hist):
        out = np.asarray(x, dtype=np.float32).astype(np.float64)
        tb = np.searchsorted(self.t_edges, t, side="left")
        hb = np.searchsorted(self.hist_edges, n_hist, side="right")
        for ti in range(len(self.t_edges) + 1):
            for hi in range(len(self.hist_edges) + 1):
                ix = np.flatnonzero((tb == ti) & (hb == hi))
                for j in range(0, len(ix), 32768):
                    ii = ix[j : j + 32768]
                    out[ii, 92:] = pit_logit(out[ii, 92:], np.broadcast_to(self.knots[ti, hi], (len(ii), 18, 33)))
        return out.astype(np.float32)


def quantiles(a, ix):
    return np.stack([np.quantile(a[ix, 92 + j], PROBS) for j in range(18)])


def export_dump(dump, path):
    """LightGBM ``dump_model()`` -> padded per-tree arrays read by ``NumpyTrees``."""
    trees = dump["tree_info"]
    nt = len(trees)
    nmax = 1

    def count(n):
        return 1 if "leaf_value" in n else 1 + count(n["left_child"]) + count(n["right_child"])

    nmax = max(count(q["tree_structure"]) for q in trees)
    f = np.zeros((nt, nmax), dtype=np.int32)
    th = np.zeros((nt, nmax))
    left = np.zeros((nt, nmax), dtype=np.int32)
    right = left.copy()
    val = th.copy()
    maxdepth = 0
    for ti, tree in enumerate(trees):
        nodes = []

        def rec(n, depth):
            nonlocal maxdepth
            maxdepth = max(maxdepth, depth)
            idx = len(nodes)
            nodes.append(n)
            if "leaf_value" in n:
                val[ti, idx] = n["leaf_value"]
                left[ti, idx] = idx
                right[ti, idx] = idx
            else:
                if n.get("decision_type", "<=") != "<=":
                    raise ValueError("Only continuous <= trees are allowed")
                f[ti, idx] = n["split_feature"]
                th[ti, idx] = float(n["threshold"])
                left[ti, idx] = rec(n["left_child"], depth + 1)
                right[ti, idx] = rec(n["right_child"], depth + 1)
            return idx

        rec(tree["tree_structure"], 0)
    if maxdepth > 5:
        raise ValueError("Tree depth above 5")
    np.savez(path, feature=f, threshold=th, left=left, right=right, value=val, depth=np.array(maxdepth, dtype=np.int64))
