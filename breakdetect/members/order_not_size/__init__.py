"""The "Order, not size" member: ridge on ranks/PIT."""

from __future__ import annotations

import math
import os

import numpy as np

from .columns import PAIRS_I, PAIRS_J, U_CLIP, ColumnEngine, pit_transform, t_bucket, tail_scales


class OrderNotSize:
    """Conditional PIT of the 18 columns, logits plus within-group products, then a ridge score."""

    feature_inputs: tuple[str, ...] = ()

    def __init__(self, model_dirs) -> None:
        (model_dir,) = model_dirs
        z = np.load(os.path.join(model_dir, "model.npz"), allow_pickle=False)
        self.knots = np.ascontiguousarray(z["knots"], dtype=np.float64)
        self.nh_edges = z["nh_edges"].astype(np.float64)
        self.coef = np.ascontiguousarray(z["coef"], dtype=np.float64)
        self.intercept = float(z["intercept"])
        if not np.array_equal(z["pairs_i"], PAIRS_I) or not np.array_equal(z["pairs_j"], PAIRS_J):
            raise ValueError("feature pair order mismatch between model and code")
        self.slo_all, self.shi_all = tail_scales(self.knots)
        self.columns = ColumnEngine()

    def start(self, history):
        st = self.columns.start(history)
        st.hb = int(np.sum(self.nh_edges <= st.n))
        return st

    def step(self, st, x: float, inputs=()) -> tuple[list[float], None]:
        """Score of the point, mapped to (0, 1) by a logistic centred at 0.5."""
        cols = self.columns.step_columns(st, x)
        tb = t_bucket(st.m)
        if tb != st.tb_cur:
            st.tb_cur = tb
            st.K_cur = self.knots[st.hb, tb]
            st.slo_cur = self.slo_all[st.hb, tb]
            st.shi_cur = self.shi_all[st.hb, tb]
        u = pit_transform(cols, st.K_cur, st.slo_cur, st.shi_cur)
        uc = np.clip(u, U_CLIP, 1.0 - U_CLIP)
        lg = np.log(uc / (1.0 - uc))
        f = np.concatenate([lg, lg[PAIRS_I] * lg[PAIRS_J]])
        s = self.intercept + float(f @ self.coef)
        return [1.0 / (1.0 + math.exp(-4.0 * (s - 0.5)))], None
