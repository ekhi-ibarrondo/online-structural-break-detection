"""Train the surprises member: features, out-of-fold scores, full-data models and CDF maps.

    python -m surprises.train all                 # features + seeds 0, 1, 2
    python -m surprises.train oof --seed 1        # one step, one seed

Learner seed: LightGBM ``seed = 1000 + k`` for protocol seed k (the fold does
not enter the seed).  150 boosting rounds, no early stopping, no validation set.
Outputs (under ``$OUT_DIR/surprises/``): ``model_full_surprises_s{k}/`` and
``maps_surprises_train10k_s{k}.npz`` for k = 0, 1, 2.
"""
from __future__ import annotations

import json
from pathlib import Path

import numpy as np

from common import driver

HERE = Path(__file__).resolve().parent


def make_extractor():
    from .features import batch_features, fit_hist

    def extract(x_hist, x_online):
        return {"X": batch_features(fit_hist(x_hist), x_online)}

    return extract


class Surprises(driver.Member):
    name = "surprises"
    module = "surprises.train"
    outputs = {"X": (182, np.float32)}

    def __init__(self):
        from . import gbm_numpy, pipeline
        from .features import N_FEATURES, feature_names

        self.PL, self.G = pipeline, gbm_numpy
        self.n_feat = N_FEATURES
        self.fnames = feature_names()
        self.knobs = pipeline.load_knobs(str(HERE / "knobs.json"))

    def learner(self, k, f):
        return 1000 + int(k)

    def fit(self, pop, F, train_mask, k, f, work):
        X, y = F["X"], pop.labels
        t, sid = pop.t0.astype(np.int32), pop.row_series.astype(np.int32)
        model, booster = self.PL.train_fold(X, y, t, sid, train_mask, self.knobs, int(k), int(f), log=driver.log)
        samp = np.flatnonzero(train_mask[sid])
        samp = samp[np.linspace(0, samp.size - 1, num=min(4000, samp.size), dtype=np.int64)]
        Xs = np.ascontiguousarray(X[samp])
        raw_np = self.G.predict_raw_batch(model, np.ascontiguousarray(Xs.T))
        ref = booster.predict(Xs, raw_score=True, num_threads=1)
        driver.log(f"  numpy export vs LightGBM, max |diff| = {float(np.max(np.abs(raw_np - ref))):.3e}")
        return {"model": model, "booster": booster}

    def predict(self, model, pop, F, rows):
        return self.PL.predict_rows(model["model"], F["X"], rows)

    def export(self, model, out_dir, meta):
        import lightgbm

        extra = dict(meta, knobs=self.knobs, n_features=self.n_feat, feature_names=self.fnames,
                     lgb_params=self.PL.lgb_params(self.knobs, int(meta["seed"])),
                     lightgbm_version=lightgbm.__version__, numpy_version=np.__version__)
        self.G.save_model(model["model"], str(out_dir), extra=extra)
        (Path(out_dir) / "meta.json").write_text(json.dumps(extra, indent=1) + "\n", encoding="utf-8")

    def cleanup(self, model):
        return None


if __name__ == "__main__":
    driver.main(Surprises())
