"""Export a fitted LightGBM booster to flat node arrays (feat/thr/left/right/value/roots/meta).

No LightGBM import in this module: the caller supplies the fitted Booster.
All iterations are exported (no early stopping).
"""
from pathlib import Path

import numpy as np

ARRAY_NAMES = ("feat", "thr", "left", "right", "value", "roots", "meta")


def export_booster(booster, n_features):
    dump = booster.dump_model(num_iteration=-1)
    objective = dump.get("objective", "")
    if (objective not in ("binary", "binary sigmoid:1") or dump.get("num_class") != 1
            or dump.get("num_tree_per_iteration") != 1 or dump.get("average_output", False)):
        raise ValueError("only ordinary binary tree raw scores are supported")
    if n_features != booster.num_feature() or n_features != dump["max_feature_idx"] + 1:
        raise ValueError("feature width differs from training")
    if booster.params.get("boosting_type", booster.params.get("boosting", "gbdt")) != "gbdt":
        raise ValueError("only tree is supported")
    if int(getattr(booster, "best_iteration", 0) or 0) > 0:
        raise ValueError("early-stopped model: the recipe uses a fixed number of trees")
    feat, thr, left, right, value, roots = [], [], [], [], [], []
    max_depth = 0

    def add(node, depth):
        nonlocal max_depth
        i = len(feat)
        if "leaf_value" in node:
            if any(k in node for k in ("leaf_const", "leaf_features", "leaf_coeff")):
                raise ValueError("linear leaves are unsupported")
            feat.append(0)
            thr.append(0.)
            left.append(i)
            right.append(i)
            value.append(float(node["leaf_value"]))
            max_depth = max(max_depth, depth)
            return i
        if node.get("decision_type") != "<=":
            raise ValueError("categorical/non-numeric split")
        # NaN routing is irrelevant for this finite-input runtime. Zero routing is not.
        if node.get("missing_type") not in ("None", "NaN"):
            raise ValueError("unsupported missing semantics (including zero routing)")
        feat.append(int(node["split_feature"]))
        thr.append(float(node["threshold"]))
        left.append(-1)
        right.append(-1)
        value.append(0.)
        left[i] = add(node["left_child"], depth + 1)
        right[i] = add(node["right_child"], depth + 1)
        return i

    for tree in dump["tree_info"]:
        if tree.get("is_linear", False):
            raise ValueError("linear tree")
        roots.append(add(tree["tree_structure"], 0))
    model = dict(feat=np.asarray(feat, np.int32), thr=np.asarray(thr, np.float64),
                 left=np.asarray(left, np.int32), right=np.asarray(right, np.int32),
                 value=np.asarray(value, np.float64), roots=np.asarray(roots, np.int32),
                 meta=np.asarray([len(roots), max_depth, n_features], np.int64))
    validate_model(model)
    return model


def save_model(model, directory):
    validate_model(model)
    path = Path(directory)
    path.mkdir(parents=True, exist_ok=False)
    for k in ARRAY_NAMES:
        np.save(path / f"{k}.npy", model[k], allow_pickle=False)


def validate_model(model):
    """Reject corrupt graphs (bad dtypes, cycles, shared nodes, out-of-range children)."""
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
