"""Gradient-boosted trees padded to complete binary trees (heap layout), evaluated in numpy."""

from __future__ import annotations

import numpy as np


class NumpyGBM:
    """Trees of depth D: ``feat``/``thr`` per internal node, ``leaf`` per leaf; go right iff x > thr."""

    def __init__(self, npz_path):
        z = np.load(npz_path, allow_pickle=False)
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
        self.tree_off_int = np.arange(self.T, dtype=np.int64) * self.n_int
        self.tree_off_leaf = np.arange(self.T, dtype=np.int64) * self.leaf.shape[1]
        self._node = np.zeros(self.T, dtype=np.int64)

    def raw_single(self, x):
        """Raw (pre-sigmoid) score of one feature vector."""
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
