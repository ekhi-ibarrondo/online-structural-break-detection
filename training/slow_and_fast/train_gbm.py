"""Offline GBM training (LightGBM) with per-step asymmetric weights, and numpy export."""
from __future__ import annotations

import numpy as np

from .gbm_np import NumpyGBM, export_lightgbm

GBM_FIXED = dict(
    objective="binary",
    num_leaves=31,
    max_depth=7,
    feature_fraction=0.7,
    bagging_fraction=0.5,
    bagging_freq=1,
    max_bin=127,
    min_sum_hessian_in_leaf=1e-3,
    num_threads=1,
    deterministic=True,
    force_row_wise=True,
    verbose=-1,
)
EXPORT_DEPTH = 7
ROW_SUB = 4          # deterministic 1-in-4 step subsampling of training rows (all steps t<16 kept)
ROW_SUB_MIN_T = 16


def step_counts(labels_list, t_max):
    """n_pos(t), n_neg(t) over a list of label arrays (training fold), t up to t_max."""
    n_pos = np.zeros(t_max + 1)
    n_all = np.zeros(t_max + 1)
    for lab in labels_list:
        n = lab.shape[0]
        n_all[:n] += 1.0
        n_pos[:n] += lab
    return n_pos, n_all - n_pos


def build_training_rows(feats_list, labels_list, ids_list):
    """Stack subsampled rows with asymmetric per-step weights (pos x n_neg(t), neg x n_pos(t))."""
    t_max = max(lab.shape[0] for lab in labels_list) - 1
    n_pos, n_neg = step_counts(labels_list, t_max)
    X_parts, y_parts, w_parts = [], [], []
    for F, lab, sid in zip(feats_list, labels_list, ids_list):
        n = lab.shape[0]
        t = np.arange(n)
        keep = (t < ROW_SUB_MIN_T) | (((t + int(sid)) % ROW_SUB) == 0)
        lab_k = lab[keep].astype(np.float64)
        tk = t[keep]
        w = np.where(lab_k > 0.5, n_neg[tk], n_pos[tk])
        X_parts.append(F[keep])
        y_parts.append(lab_k)
        w_parts.append(w)
    X = np.concatenate(X_parts, axis=0)
    y = np.concatenate(y_parts)
    w = np.concatenate(w_parts)
    wm = w.mean()
    if wm > 0:
        w = w / wm
    return X, y, w


def train_gbm(X, y, w, knobs, seed):
    import lightgbm as lgb
    params = dict(GBM_FIXED)
    params.update(
        learning_rate=float(knobs["learning_rate"]),
        min_data_in_leaf=int(knobs["min_data_in_leaf"]),
        lambda_l2=float(knobs["lambda_l2"]),
        seed=int(seed),
    )
    ds = lgb.Dataset(X, label=y, weight=w, free_raw_data=True, params={"verbose": -1})
    booster = lgb.train(params, ds, num_boost_round=int(knobs["n_trees"]))
    return booster


def predict_raw(booster, X, n_trees=None):
    return booster.predict(X, raw_score=True, num_iteration=n_trees)


def export(booster, n_features, path):
    export_lightgbm(booster, EXPORT_DEPTH, n_features, path)
    return NumpyGBM(path)
