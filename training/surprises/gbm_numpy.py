"""Export of a LightGBM booster to flat numpy arrays + numpy-only evaluation.

Node arrays (all trees concatenated):
  feat  int32   split feature (0 for leaves)
  thr   float64 split threshold (x <= thr goes left; 0 for leaves)
  left  int32   left child (self for leaves)
  right int32   right child (self for leaves)
  value float64 leaf value (0 for internal nodes)
  roots int32   root node index per tree
  meta  int64   [n_trees, max_depth, n_features]

Raw score = sequential sum over trees of the reached leaf values;
probability = 1 / (1 + exp(-raw)) (LightGBM binary objective, sigmoid=1).
"""
from __future__ import annotations

import json
import os

import numpy as np

ARRAY_NAMES = ("feat", "thr", "left", "right", "value", "roots", "meta")


def export_booster(booster, n_features: int) -> dict:
    dump = booster.dump_model()
    feat, thr, left, right, value = [], [], [], [], []
    roots = []
    max_depth = 0

    def add(node, depth):
        nonlocal max_depth
        i = len(feat)
        if "leaf_value" in node:
            feat.append(0)
            thr.append(0.0)
            left.append(i)
            right.append(i)
            value.append(float(node["leaf_value"]))
            max_depth = max(max_depth, depth)
            return i
        if node.get("decision_type") != "<=":
            raise ValueError(f"unsupported decision type {node.get('decision_type')}")
        feat.append(int(node["split_feature"]))
        thr.append(float(node["threshold"]))
        left.append(-1)
        right.append(-1)
        value.append(0.0)
        li = add(node["left_child"], depth + 1)
        ri = add(node["right_child"], depth + 1)
        left[i] = li
        right[i] = ri
        return i

    for tree in dump["tree_info"]:
        roots.append(add(tree["tree_structure"], 0))
    return {
        "feat": np.asarray(feat, dtype=np.int32),
        "thr": np.asarray(thr, dtype=np.float64),
        "left": np.asarray(left, dtype=np.int32),
        "right": np.asarray(right, dtype=np.int32),
        "value": np.asarray(value, dtype=np.float64),
        "roots": np.asarray(roots, dtype=np.int32),
        "meta": np.asarray([len(roots), max_depth, n_features], dtype=np.int64),
    }


def save_model(model: dict, out_dir: str, extra: dict | None = None) -> None:
    os.makedirs(out_dir, exist_ok=True)
    for k in ARRAY_NAMES:
        np.save(os.path.join(out_dir, f"{k}.npy"), model[k])
    if extra:
        with open(os.path.join(out_dir, "model_info.json"), "w") as f:
            json.dump(extra, f, indent=2, sort_keys=True)


def load_model(model_dir: str) -> dict:
    model = {k: np.load(os.path.join(model_dir, f"{k}.npy")) for k in ARRAY_NAMES}
    model["is_leaf"] = model["left"] == np.arange(model["left"].shape[0], dtype=np.int32)
    return model


def predict_raw_batch(model: dict, XT: np.ndarray) -> np.ndarray:
    """Raw scores for XT of shape (n_features, n_rows), sequential over trees."""
    n = XT.shape[1]
    out = np.zeros(n, dtype=np.float64)
    feat, thr, left, right, value = (
        model["feat"], model["thr"], model["left"], model["right"], model["value"],
    )
    is_leaf = model["is_leaf"]
    all_idx = np.arange(n)
    for root in model["roots"]:
        stack = [(int(root), all_idx)]
        while stack:
            node, idx = stack.pop()
            if is_leaf[node]:
                out[idx] += value[node]
                continue
            go = XT[feat[node]][idx] <= thr[node]
            stack.append((int(left[node]), idx[go]))
            stack.append((int(right[node]), idx[~go]))
    return out


def predict_raw_single(model: dict, x: np.ndarray) -> float:
    """Raw score for one feature vector x (float64 values), all trees at once."""
    feat, thr, left, right = model["feat"], model["thr"], model["left"], model["right"]
    node = model["roots"]
    for _ in range(int(model["meta"][1])):
        go = x[feat[node]] <= thr[node]
        node = np.where(go, left[node], right[node])
    return float(np.cumsum(model["value"][node])[-1])


def sigmoid(raw):
    return 1.0 / (1.0 + np.exp(-raw))
