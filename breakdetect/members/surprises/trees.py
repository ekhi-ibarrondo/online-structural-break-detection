"""Gradient-boosted trees exported to flat numpy arrays, evaluated one feature vector at a time."""

from __future__ import annotations

import os

import numpy as np

ARRAY_NAMES = ("feat", "thr", "left", "right", "value", "roots", "meta")


def load_model(model_dir: str) -> dict:
    """Node arrays of all trees (leaves point to themselves) and ``meta`` = [n_trees, depth, n_features]."""
    return {k: np.load(os.path.join(model_dir, f"{k}.npy"), allow_pickle=False) for k in ARRAY_NAMES}


def predict_raw_single(model: dict, x: np.ndarray) -> float:
    """Raw score: every tree descends ``depth`` levels in lockstep, then leaf values are summed in order."""
    feat, thr, left, right = model["feat"], model["thr"], model["left"], model["right"]
    node = model["roots"]
    for _ in range(int(model["meta"][1])):
        go = x[feat[node]] <= thr[node]
        node = np.where(go, left[node], right[node])
    return float(np.cumsum(model["value"][node])[-1])
