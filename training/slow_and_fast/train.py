"""Train the slow_and_fast member: spectral features, out-of-fold scores, full-data models and CDF maps.

    python -m slow_and_fast.train all
    python -m slow_and_fast.train fit --seed 2

Learner seed: LightGBM ``seed = 1000 + 10 k + f`` (protocol seed k, fold f; the
full-data model uses f = 0).  200 boosting rounds, no early stopping, no
validation set.  Out-of-fold scores are the LightGBM raw scores through a
sigmoid.  Outputs (under ``$OUT_DIR/slow_and_fast/``): ``model_full_slow_and_fast_s{k}/``
and ``maps_slow_and_fast_train10k_s{k}.npz`` for k = 0, 1, 2.
"""
from __future__ import annotations

import json
import math
import shutil
from pathlib import Path

import numpy as np

from common import driver

HERE = Path(__file__).resolve().parent


def make_extractor():
    from .features import FeatureEngine, extract_features

    engine = FeatureEngine()

    def extract(x_hist, x_online):
        return {"X": extract_features(x_hist, x_online, engine)}

    return extract


class SlowAndFast(driver.Member):
    name = "slow_and_fast"
    module = "slow_and_fast.train"
    outputs = {"X": (96, np.float64)}

    def __init__(self):
        from . import features, train_gbm
        from .gbm_np import NumpyGBM

        self.FF, self.TG, self.NumpyGBM = features, train_gbm, NumpyGBM
        self.knobs = json.loads((HERE / "knobs.json").read_text())

    def learner(self, k, f):
        return 1000 + 10 * int(k) + int(f)

    def fit(self, pop, F, train_mask, k, f, work):
        X = F["X"]
        off = pop.offsets
        idx = np.flatnonzero(train_mask)
        feats = [X[off[i]:off[i + 1]] for i in idx]
        labels = [np.asarray(pop.series[i].labels).astype(np.int8) for i in idx]
        ids = [pop.series[i].id for i in idx]
        Xt, y, w = self.TG.build_training_rows(feats, labels, ids)
        booster = self.TG.train_gbm(Xt, y, w, self.knobs, seed=self.learner(k, f))
        del Xt, y, w, feats
        # numerical check of the numpy export against LightGBM on training rows
        if work.exists():
            shutil.rmtree(work)
        work.mkdir(parents=True)
        npm = self.TG.export(booster, self.FF.N_FEAT, str(work / "model.npz"))
        rows_tr = np.flatnonzero(train_mask[pop.row_series])[:20000]
        samp = np.ascontiguousarray(X[rows_tr], dtype=np.float64)
        chk = float(np.max(np.abs(npm.raw_batch(samp) - self.TG.predict_raw(booster, samp))))
        driver.log(f"  numpy export vs LightGBM, max |diff| = {chk:.3e}")
        return {"booster": booster, "wd": work}

    def predict(self, model, pop, F, rows):
        Xt = np.ascontiguousarray(F["X"][rows])
        raw = self.TG.predict_raw(model["booster"], Xt, n_trees=int(self.knobs["n_trees"]))
        return 1.0 / (1.0 + np.exp(-raw))

    def export(self, model, out_dir, meta):
        import lightgbm

        out_dir.mkdir(parents=True, exist_ok=True)
        self.TG.export(model["booster"], self.FF.N_FEAT, str(out_dir / "model.npz"))
        info = dict(meta, knobs=self.knobs, n_features=self.FF.N_FEAT, feature_names=self.FF.feature_names(),
                    lightgbm_version=lightgbm.__version__)
        (out_dir / "meta.json").write_text(json.dumps(info, indent=1) + "\n", encoding="utf-8")

    def batch_from_export(self, out_dir, pop, F, rows):
        """Scores of an exported model with the exact per-point formula of the inference scorer."""
        npm = self.NumpyGBM(str(Path(out_dir) / "model.npz"))
        raw = npm.raw_batch(np.asarray(F["X"][rows], dtype=np.float64))
        out = np.empty(raw.shape[0], dtype=np.float64)
        for i, r in enumerate(raw.tolist()):
            if r >= 0.0:
                out[i] = 1.0 / (1.0 + math.exp(-r))
            else:
                z = math.exp(r)
                out[i] = z / (1.0 + z)
        return out


if __name__ == "__main__":
    driver.main(SlowAndFast())
