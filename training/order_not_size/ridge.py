"""Fitting of order_not_size: conditional PIT of the 18 columns + closed-form weighted ridge (numpy only).

Conditional PIT on the label-0 training rows: 21 step buckets x 3 tertiles of
the historical length, at least 1500 rows per cell, fallback to the step
bucket then to the global knots, 29 knots.  Features: logits of the PITs plus
within-group products (115).  Weighted ridge in closed form (alpha = 1.0,
unpenalised intercept); score map sigmoid(4 (s - 0.5)).  No random seed.
"""
from __future__ import annotations

import hashlib
import json
import os
import time
from pathlib import Path

import numpy as np

from .scorer import (
    COLUMN_NAMES,
    FEATURE_NAMES,
    KNOT_LEVELS,
    MIN_CELL,
    N_COLS,
    N_FEATURES,
    N_KNOTS,
    N_TBUCKETS,
    PAIRS_I,
    PAIRS_J,
    RIDGE_ALPHA,
    T_EDGES,
    features_from_u,
    pit_transform,
    score_map,
    tail_scales,
)

T_EDGES_ARR = np.array(T_EDGES, dtype=np.int64)


def sha256_file(p: str | os.PathLike) -> str:
    h = hashlib.sha256()
    with open(p, "rb") as f:
        for chunk in iter(lambda: f.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def sha256_array(a: np.ndarray) -> str:
    return hashlib.sha256(np.ascontiguousarray(a).tobytes()).hexdigest()


def t_bucket_vec(m: np.ndarray) -> np.ndarray:
    return np.searchsorted(T_EDGES_ARR, m, side="right") - 1


def nh_edges_from_train(nh_train: np.ndarray) -> np.ndarray:
    return np.array([np.quantile(nh_train, 1.0 / 3.0), np.quantile(nh_train, 2.0 / 3.0)])


def nh_bucket_vec(nh: np.ndarray, edges: np.ndarray) -> np.ndarray:
    return (nh[:, None] >= edges[None, :]).sum(axis=1)


def fit_pit(cols_null: np.ndarray, tb_null: np.ndarray, hb_null: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """Knots (3, 21, 18, 29) resolved with the hierarchical fallback; source (3, 21): 0 cell, 1 t-marginal, 2 global."""
    knots = np.empty((3, N_TBUCKETS, N_COLS, N_KNOTS))
    source = np.full((3, N_TBUCKETS), -1, dtype=np.int64)
    glob = np.quantile(cols_null, KNOT_LEVELS, axis=0).T  # (18, 29)
    tmarg = {}
    for tb in range(N_TBUCKETS):
        sel = tb_null == tb
        if sel.sum() >= MIN_CELL:
            tmarg[tb] = np.quantile(cols_null[sel], KNOT_LEVELS, axis=0).T
        for hb in range(3):
            sel2 = sel & (hb_null == hb)
            if sel2.sum() >= MIN_CELL:
                knots[hb, tb] = np.quantile(cols_null[sel2], KNOT_LEVELS, axis=0).T
                source[hb, tb] = 0
            elif tb in tmarg:
                knots[hb, tb] = tmarg[tb]
                source[hb, tb] = 1
            else:
                knots[hb, tb] = glob
                source[hb, tb] = 2
    return knots, source


def apply_pit(cols: np.ndarray, tb: np.ndarray, hb: np.ndarray, knots: np.ndarray) -> np.ndarray:
    """u (N, 18) using exactly the inference transform per (hb, tb) cell and column."""
    u = np.empty_like(cols)
    slo_all, shi_all = tail_scales(knots)
    cell = hb * N_TBUCKETS + tb
    order = np.argsort(cell, kind="stable")
    cell_sorted = cell[order]
    bounds = np.flatnonzero(np.diff(cell_sorted)) + 1
    starts = np.concatenate([[0], bounds])
    ends = np.concatenate([bounds, [cell_sorted.shape[0]]])
    for a, b in zip(starts, ends):
        idx = order[a:b]
        c = int(cell_sorted[a])
        h, t = divmod(c, N_TBUCKETS)
        for j in range(N_COLS):
            K = knots[h, t, j][None, :]
            u[idx, j] = pit_transform(cols[idx, j], K, slo_all[h, t, j], shi_all[h, t, j])
    return u


def ridge_weights(y: np.ndarray, t: np.ndarray, t_max: int) -> np.ndarray:
    """positives x n_neg(t), negatives x n_pos(t) over the training rows; zero-weight rows get 0; mean 1."""
    n_all = np.bincount(t, minlength=t_max).astype(np.float64)
    n_pos = np.bincount(t, weights=y.astype(np.float64), minlength=t_max)
    n_neg = n_all - n_pos
    w = np.where(y == 1, n_neg[t], n_pos[t])
    pos = w > 0
    w = w / w[pos].mean()
    return w


def ridge_fit(u: np.ndarray, y: np.ndarray, w: np.ndarray, alpha: float = RIDGE_ALPHA,
              block: int = 200_000) -> tuple[np.ndarray, float, dict]:
    """Closed-form weighted ridge with unpenalized intercept; normal equations accumulated by blocks."""
    d = N_FEATURES + 1
    XtWX = np.zeros((d, d))
    XtWy = np.zeros(d)
    N = u.shape[0]
    for a in range(0, N, block):
        b = min(a + block, N)
        wb = w[a:b]
        keep = wb > 0
        if not keep.any():
            continue
        F = features_from_u(u[a:b][keep])
        X = np.concatenate([np.ones((F.shape[0], 1)), F], axis=1)
        Xw = X * wb[keep][:, None]
        XtWX += Xw.T @ X
        XtWy += Xw.T @ y[a:b][keep].astype(np.float64)
    D = np.eye(d)
    D[0, 0] = 0.0
    beta = np.linalg.solve(XtWX + alpha * D, XtWy)
    info = {"n_rows_used": int((w > 0).sum()), "trace_XtWX": float(np.trace(XtWX))}
    return beta[1:], float(beta[0]), info


def predict(u: np.ndarray, coef: np.ndarray, intercept: float, block: int = 200_000) -> np.ndarray:
    out = np.empty(u.shape[0])
    for a in range(0, u.shape[0], block):
        b = min(a + block, u.shape[0])
        F = features_from_u(u[a:b])
        out[a:b] = score_map(intercept + F @ coef)
    return out


def export_model(out_dir: str | os.PathLike, knots: np.ndarray, source: np.ndarray, nh_edges: np.ndarray,
                 coef: np.ndarray, intercept: float, meta: dict) -> dict:
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    np.savez(out_dir / "model.npz", knots=knots, pit_source=source, nh_edges=nh_edges, coef=coef,
             intercept=np.array(intercept), pairs_i=PAIRS_I, pairs_j=PAIRS_J,
             knot_levels=KNOT_LEVELS, t_edges=T_EDGES_ARR)
    meta = dict(meta)
    meta["feature_order"] = FEATURE_NAMES
    meta["column_order"] = COLUMN_NAMES
    meta["model_npz_sha256"] = sha256_file(out_dir / "model.npz")
    with open(out_dir / "meta.json", "w") as f:
        json.dump(meta, f, indent=1, sort_keys=True)
    meta["meta_json_sha256"] = sha256_file(out_dir / "meta.json")
    return meta


class Timer:
    def __init__(self):
        self.t0 = time.time()
        self.marks = {}

    def mark(self, name: str) -> float:
        t = time.time() - self.t0
        self.marks[name] = round(t, 3)
        return t
