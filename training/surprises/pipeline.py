"""Fitting of surprises: one pooled LightGBM over the per-step features, exported to numpy.

Everything is deterministic given the seeds (single-threaded LightGBM).
"""
from __future__ import annotations

import json
import time

import numpy as np

from .features import N_FEATURES, feature_names
from .gbm_numpy import export_booster, predict_raw_batch, sigmoid

KNOB_NAMES = (
    "learning_rate",
    "num_iterations",
    "num_leaves",
    "min_data_in_leaf",
    "feature_fraction",
    "bagging_fraction",
    "lambda_l2",
    "max_depth",
    "row_stride",
    "max_bin",
)


def load_knobs(path: str) -> dict:
    with open(path) as f:
        knobs = json.load(f)
    if set(knobs) != set(KNOB_NAMES):
        raise ValueError(f"knobs.json keys {sorted(knobs)} != {sorted(KNOB_NAMES)}")
    return knobs


def lgb_params(knobs: dict, seed: int) -> dict:
    return {
        "objective": "binary",
        "learning_rate": float(knobs["learning_rate"]),
        "num_leaves": int(knobs["num_leaves"]),
        "min_data_in_leaf": int(knobs["min_data_in_leaf"]),
        "feature_fraction": float(knobs["feature_fraction"]),
        "bagging_fraction": float(knobs["bagging_fraction"]),
        "bagging_freq": 1 if float(knobs["bagging_fraction"]) < 1.0 else 0,
        "lambda_l2": float(knobs["lambda_l2"]),
        "max_depth": int(knobs["max_depth"]),
        "max_bin": int(knobs["max_bin"]),
        "num_threads": 1,
        "deterministic": True,
        "force_row_wise": True,
        "seed": 1000 + seed,
        "verbose": -1,
        "boost_from_average": True,
    }


def train_fold(X, y, t_idx, sid, train_series_mask, knobs, seed, fold, log=print):
    """Fit the pooled GBM on the training-fold rows; return the numpy model and the booster.

    Rows are subsampled with a fixed stride over (t + series index); each row is
    weighted by the opposite-class count at its step (positives by n_neg(t),
    negatives by n_pos(t)), normalised to mean 1.
    """
    import lightgbm as lgb

    is_train_row = train_series_mask[sid]
    yt = y.astype(np.float64)
    t_max = int(t_idx.max()) + 1
    # per-step class counts over ALL rows of the training-fold series
    n_pos = np.bincount(t_idx[is_train_row], weights=yt[is_train_row], minlength=t_max)
    n_all = np.bincount(t_idx[is_train_row], minlength=t_max).astype(np.float64)
    n_neg = n_all - n_pos
    stride = int(knobs["row_stride"])
    keep = is_train_row & (((t_idx.astype(np.int64) + sid) % stride) == 0)
    w = np.where(y == 1, n_neg[t_idx], n_pos[t_idx])
    keep &= w > 0
    rows = np.flatnonzero(keep)
    w_rows = w[rows]
    w_rows = w_rows / w_rows.mean()
    t0 = time.time()
    ds = lgb.Dataset(
        X[rows], label=y[rows].astype(np.float32), weight=w_rows.astype(np.float32),
        params={"max_bin": int(knobs["max_bin"]), "verbose": -1},
        free_raw_data=True,
    )
    booster = lgb.train(lgb_params(knobs, seed), ds, num_boost_round=int(knobs["num_iterations"]))
    model = export_booster(booster, N_FEATURES)
    model["is_leaf"] = model["left"] == np.arange(model["left"].shape[0], dtype=np.int32)
    log(
        f"  seed {seed} fold {fold}: trained on {rows.shape[0]} rows "
        f"({int(train_series_mask.sum())} series), {int(model['meta'][0])} trees, "
        f"max depth {int(model['meta'][1])}, {time.time()-t0:.1f}s"
    )
    # logging only: gain share of the top features
    gain = booster.feature_importance(importance_type="gain")
    if gain.sum() > 0:
        names = feature_names()
        top = np.argsort(gain)[::-1][:15]
        log("    top-15 gain: " + ", ".join(f"{names[i]}={gain[i]/gain.sum():.3f}" for i in top))
    return model, booster


def predict_rows(model, X, rows, chunk=200000):
    out = np.empty(rows.shape[0], dtype=np.float64)
    for a in range(0, rows.shape[0], chunk):
        b = min(a + chunk, rows.shape[0])
        XT = np.ascontiguousarray(X[rows[a:b]].T)
        out[a:b] = predict_raw_batch(model, XT)
    return sigmoid(out)
