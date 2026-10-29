"""Numpy-only GBM inference (padded complete binary trees) + LightGBM export (offline only).

Arrays (model.npz):
  feat  (T, 2^D-1) int32   split feature per internal node (heap layout, children 2i+1 / 2i+2)
  thr   (T, 2^D-1) float64 split threshold; rule: go right iff x[feat] > thr
  leaf  (T, 2^D)   float64 leaf values
  meta: depth D, n_features, base score (float), sigmoid flag
Pass-through nodes (leaf reached above depth D) use thr=+inf so traversal always goes left.
"""
from __future__ import annotations

import numpy as np


class NumpyGBM:
    def __init__(self, npz_path):
        z = np.load(npz_path)
        self.feat = np.ascontiguousarray(z["feat"].astype(np.int64))
        self.thr = np.ascontiguousarray(z["thr"].astype(np.float64))
        self.leaf = np.ascontiguousarray(z["leaf"].astype(np.float64))
        self.depth = int(z["depth"][0])
        self.n_features = int(z["n_features"][0])
        self.base = float(z["base"][0])
        self.T = self.feat.shape[0]
        self.n_int = self.feat.shape[1]
        self.feat_flat = self.feat.ravel()
        self.thr_flat = self.thr.ravel()
        self.leaf_flat = self.leaf.ravel()
        self.tree_off_int = (np.arange(self.T, dtype=np.int64) * self.n_int)
        self.tree_off_leaf = (np.arange(self.T, dtype=np.int64) * self.leaf.shape[1])
        self._node = np.zeros(self.T, dtype=np.int64)

    def raw_single(self, x):
        """Raw (pre-sigmoid) score for one feature vector x (n_features,)."""
        node = self._node
        node[:] = 0
        off = self.tree_off_int
        ff = self.feat_flat
        tt = self.thr_flat
        for _ in range(self.depth):
            idx = off + node
            go = x[ff[idx]] > tt[idx]
            node *= 2
            node += 1
            node += go
        leaf_idx = node - self.n_int
        return float(self.leaf_flat[self.tree_off_leaf + leaf_idx].sum()) + self.base

    def raw_batch(self, X, chunk=4096):
        X = np.asarray(X, dtype=np.float64)
        out = np.empty(X.shape[0])
        for a in range(0, X.shape[0], chunk):
            xb = X[a:a + chunk]
            nb = xb.shape[0]
            node = np.zeros((nb, self.T), dtype=np.int64)
            rows = np.arange(nb)[:, None]
            for _ in range(self.depth):
                idx = self.tree_off_int[None, :] + node
                f = self.feat_flat[idx]
                go = xb[rows, f] > self.thr_flat[idx]
                node = 2 * node + 1 + go
            leaf_idx = node - self.n_int
            out[a:a + nb] = self.leaf_flat[self.tree_off_leaf[None, :] + leaf_idx].sum(axis=1) + self.base
        return out


def sigmoid(v):
    return 1.0 / (1.0 + np.exp(-v))


# ----------------------------------------------------------------------------- export (offline)
def export_lightgbm(booster, depth, n_features, path):
    """Convert a LightGBM booster (binary objective) into the padded numpy layout."""
    dump = booster.dump_model()
    trees = dump["tree_info"]
    T = len(trees)
    n_int = (1 << depth) - 1
    n_leaf = 1 << depth
    feat = np.zeros((T, n_int), dtype=np.int32)
    thr = np.full((T, n_int), np.inf, dtype=np.float64)
    leaf = np.zeros((T, n_leaf), dtype=np.float64)
    for ti, tr in enumerate(trees):
        def visit(node, pos, d):
            if "leaf_value" in node:          # leaf: pass-through down to depth D via leftmost path
                v = float(node["leaf_value"])
                q = pos
                for _ in range(d, depth):
                    feat[ti, q] = 0
                    thr[ti, q] = np.inf
                    q = 2 * q + 1
                leaf[ti, q - n_int] = v
                return
            if d >= depth:
                raise ValueError("tree deeper than export depth")
            if node.get("decision_type", "<=") != "<=":
                raise ValueError("unsupported decision type")
            if node.get("missing_type", "None") not in ("None", "Zero", "NaN"):
                raise ValueError("unsupported missing type")
            feat[ti, pos] = int(node["split_feature"])
            thr[ti, pos] = float(node["threshold"])
            visit(node["left_child"], 2 * pos + 1, d + 1)
            visit(node["right_child"], 2 * pos + 2, d + 1)
        visit(tr["tree_structure"], 0, 0)
    np.savez(path, feat=feat, thr=thr, leaf=leaf, depth=np.array([depth]),
             n_features=np.array([n_features]), base=np.array([0.0]))
