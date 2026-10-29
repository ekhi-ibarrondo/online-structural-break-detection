"""Train the order_not_size member: detector columns, out-of-fold scores, the full-data model and CDF maps.

    python -m order_not_size.train all
    python -m order_not_size.train oof --seed 2

The 18 columns come from ``Scorer(None).step_columns`` (numba kernels pinned to
a generic CPU target, so they are identical across machines).  The fit has no
random seed (PIT knots + closed-form ridge): the three protocol seeds only
change the folds, so one full-data model (seed 0) serves the three maps.
Outputs (under ``$OUT_DIR/order_not_size/``): ``model_full_order_not_size_s0/``
and ``maps_order_not_size_train10k_s{k}.npz`` for k = 0, 1, 2.
"""
from __future__ import annotations

from pathlib import Path

import numpy as np

from common import driver


def make_extractor():
    from .scorer import Scorer

    engine = Scorer(None)

    def extract(x_hist, x_online):
        st = engine.start(x_hist)
        cols = np.empty((x_online.shape[0], 18))
        for t, v in enumerate(x_online):
            cols[t] = engine.step_columns(st, float(v))
        return {"cols": cols}

    return extract


def _load(path):
    with np.load(path, allow_pickle=False) as z:
        return {k: z[k] for k in z.files}


class OrderNotSize(driver.Member):
    name = "order_not_size"
    module = "order_not_size.train"
    outputs = {"cols": (18, np.float64)}
    model_seeds = (0,)

    def __init__(self):
        from . import ridge

        self.ridge = ridge

    def learner(self, k, f):
        return None

    def _buckets(self, pop, nh_edges):
        ridge = self.ridge
        tb = ridge.t_bucket_vec(pop.t0 + 1)
        hb = ridge.nh_bucket_vec(pop.n_hist[pop.row_series].astype(np.float64), nh_edges)
        return tb, hb

    def fit(self, pop, F, train_mask, k, f, work):
        ridge = self.ridge
        cols = F["cols"]
        tr_rows = train_mask[pop.row_series]
        nh_edges = ridge.nh_edges_from_train(pop.n_hist[train_mask].astype(np.float64))
        tb, hb = self._buckets(pop, nh_edges)
        null = tr_rows & (pop.labels == 0)
        knots, source = ridge.fit_pit(cols[null], tb[null], hb[null])
        u_tr = ridge.apply_pit(cols[tr_rows], tb[tr_rows], hb[tr_rows], knots)
        y_tr = pop.labels[tr_rows].astype(np.int64)
        t_tr = pop.t0[tr_rows]
        w = ridge.ridge_weights(y_tr, t_tr, int(pop.t0.max()) + 1)
        coef, intercept, rinfo = ridge.ridge_fit(u_tr, y_tr, w)
        return {"knots": knots, "source": source, "nh_edges": nh_edges, "coef": coef, "intercept": intercept,
                "info": {"ridge": rinfo, "pit_source_counts_cell_tmarg_global":
                         np.bincount(source.ravel(), minlength=3).tolist(), "nh_edges": nh_edges.tolist()}}

    def _predict(self, knots, nh_edges, coef, intercept, pop, F, rows):
        tb, hb = self._buckets(pop, nh_edges)
        u = self.ridge.apply_pit(F["cols"][rows], tb[rows], hb[rows], knots)
        return self.ridge.predict(u, coef, intercept)

    def predict(self, model, pop, F, rows):
        return self._predict(model["knots"], model["nh_edges"], model["coef"], model["intercept"], pop, F, rows)

    def export(self, model, out_dir, meta):
        self.ridge.export_model(out_dir, model["knots"], model["source"], model["nh_edges"], model["coef"],
                             model["intercept"], dict(meta, **model["info"]))

    def batch_from_export(self, out_dir, pop, F, rows):
        """Scores of an exported model (``model.npz``) on the given rows."""
        z = _load(Path(out_dir) / "model.npz")
        return self._predict(np.ascontiguousarray(z["knots"], dtype=np.float64), z["nh_edges"].astype(np.float64),
                             np.ascontiguousarray(z["coef"], dtype=np.float64), float(z["intercept"]), pop, F, rows)


if __name__ == "__main__":
    driver.main(OrderNotSize())
