"""Train the pattern_removed member: features, out-of-fold scores, full-data models and CDF maps.

    python -m pattern_removed.train all
    python -m pattern_removed.train fit --seed 1

Fit (per training set):
  1. PIT of the 18 detector columns (features 92..109): 33 quantile knots of
     the label-0 training rows per cell (6 step buckets x 4 historical-length
     buckets); a cell with fewer than 256 rows falls back to its step bucket,
     then to the global knots.
  2. Rows weighted by the opposite-class count at their step (mean 1).
  3. LightGBM, 128 trees grown in 4 stages of 32 (each stage continues the
     saved model of the previous one), exported to numpy after every stage.
Learner seed: ``1729 + 100 k + f`` (protocol seed k, fold f; the full-data
model uses f = 0).  Outputs (under ``$OUT_DIR/pattern_removed/``):
``model_full_pattern_removed_s{k}/`` and ``maps_pattern_removed_train10k_s{k}.npz``
for k = 0, 1, 2.
"""
from __future__ import annotations

import json
import shutil
import time
from pathlib import Path

import numpy as np

from common import driver

HERE = Path(__file__).resolve().parent
CONFIG = json.loads((HERE / "config.json").read_text())


def make_extractor():
    from .features import FeatureState

    def extract(x_hist, x_online):
        state = FeatureState(x_hist)
        return {"X": np.stack([state.step(x) for x in x_online])}

    return extract


class PatternRemoved(driver.Member):
    name = "pattern_removed"
    module = "pattern_removed.train"
    outputs = {"X": (110, np.float32)}

    def __init__(self):
        from . import model
        from .features import FEATURE_NAMES

        self.M = model
        self.FEATURE_NAMES = FEATURE_NAMES

    def learner(self, k, f):
        return 1729 + 100 * int(k) + int(f)

    def fit(self, pop, F, train_mask, k, f, work):
        import lightgbm as lgb

        x = F["X"]
        steps = (pop.t0 + 1).astype(np.int32)
        labels = pop.labels
        rs = pop.row_series.astype(np.int32)
        n_hist = pop.n_hist
        mask = ~train_mask
        wd = work
        if wd.exists():
            shutil.rmtree(wd)
        ed = wd / "model"
        ed.mkdir(parents=True)
        # ---- conditional PIT of the detector columns (label-0 training rows) ----
        train = ~mask[rs]
        null = train & (labels == 0)
        t_edges = np.array([4, 16, 64, 256, 1024])
        h_edges = np.quantile(n_hist[~mask], [.25, .5, .75])
        nh = n_hist[rs]
        tb = np.searchsorted(t_edges, steps, side="left")
        hb = np.searchsorted(h_edges, nh, side="right")
        knots = np.empty((6, 4, 18, 33))
        counts = np.zeros((6, 4), dtype=np.int64)
        global_q = self.M.quantiles(x, np.flatnonzero(null))
        for ti in range(6):
            nix = np.flatnonzero(null & (tb == ti))
            sq = self.M.quantiles(x, nix) if len(nix) >= 256 else global_q
            for hi in range(4):
                ix = np.flatnonzero(null & (tb == ti) & (hb == hi))
                counts[ti, hi] = len(ix)
                knots[ti, hi] = self.M.quantiles(x, ix) if len(ix) >= 256 else sq
        np.savez(ed / "pit.npz", knots=knots, hist_edges=h_edges, t_edges=t_edges, counts=counts)
        pit = self.M.FoldPIT(ed / "pit.npz")
        # ---- pair-count weights ----
        ix = np.flatnonzero(train)
        yi = labels[ix]
        tt = steps[ix]
        pos = np.bincount(tt, weights=yi)
        allc = np.bincount(tt, minlength=len(pos))
        neg = allc - pos
        weights = np.where(yi == 1, neg[tt], pos[tt]).astype(np.float64)
        meanweight = float(weights[weights > 0].mean())
        weights /= meanweight
        xxpath = wd / "train_features.npy"
        xx = np.lib.format.open_memmap(xxpath, mode="w+", dtype=np.float32, shape=(len(ix), 110))
        for a in range(0, len(ix), 65536):
            ii = ix[a:a + 65536]
            xx[a:a + len(ii)] = pit.apply_batch(x[ii], steps[ii], nh[ii])
        xx.flush()
        assert np.isfinite(xx).all()
        params = dict(CONFIG["lgbm"])
        params["seed"] = self.learner(k, f)
        ds = lgb.Dataset(xx, label=yi, weight=weights, params=params, feature_name=self.FEATURE_NAMES,
                         free_raw_data=True)
        ds.construct()
        ds.save_binary(str(wd / "train.bin"))
        del ds, xx
        # ---- 4 stages of 32 trees, each continuing the saved model of the previous stage ----
        init = None
        err = None
        t0 = time.perf_counter()

        def cb(env):
            if (env.iteration + 1) % 32 == 0:
                driver.log(f"    pattern_removed: iteration {env.iteration + 1}, {time.perf_counter() - t0:.0f}s")

        for stage in range(4):
            dsb = lgb.Dataset(str(wd / "train.bin"), params=params, free_raw_data=False)
            dsb.construct()
            # The binary dataset keeps the exact bins/labels/weights; the raw feature
            # memmap is attached only so LightGBM can compute the initial predictions.
            dsb.data = np.load(xxpath, mmap_mode="r")
            model = lgb.train(params, dsb, num_boost_round=32, init_model=init, keep_training_booster=True,
                              callbacks=[cb])
            assert model.current_iteration() == 32 * (stage + 1)
            cp = wd / f"stage{stage}.txt"
            model.save_model(str(cp))
            exp = wd / f"stage{stage}.npz"
            self.M.export_dump(model.dump_model(), exp)
            raw = np.load(xxpath, mmap_mode="r")[:2048]
            ref = model.predict(raw)
            got = self.M.NumpyTrees(exp).predict_batch(raw)
            err = float(np.max(np.abs(ref - got)))
            assert err < 1e-12, err
            init = str(cp)
            del dsb, model
        shutil.copyfile(wd / "stage3.npz", ed / "trees.npz")
        driver.log(f"  pattern_removed seed {params['seed']}: {len(ix)} training rows, "
                   f"numpy export max |diff| = {err:.3e}")
        return {"wd": wd, "dir": ed}

    def _predict_dir(self, d, pop, F, rows):
        pit = self.M.FoldPIT(Path(d) / "pit.npz")
        trees = self.M.NumpyTrees(Path(d) / "trees.npz")
        x = F["X"]
        steps = (pop.t0 + 1).astype(np.int32)
        rs = pop.row_series.astype(np.int32)
        p = np.empty(len(rows), dtype=np.float64)
        for a in range(0, len(rows), 16384):
            ii = rows[a:a + 16384]
            xx = pit.apply_batch(x[ii], steps[ii], pop.n_hist[rs[ii]])
            p[a:a + len(ii)] = trees.predict_batch(xx)
        assert np.isfinite(p).all() and np.all((p >= 0) & (p <= 1))
        return p

    def predict(self, model, pop, F, rows):
        return self._predict_dir(model["dir"], pop, F, rows)

    def export(self, model, out_dir, meta):
        out_dir.mkdir(parents=True, exist_ok=True)
        shutil.copyfile(model["dir"] / "pit.npz", out_dir / "pit.npz")
        shutil.copyfile(model["dir"] / "trees.npz", out_dir / "trees.npz")
        info = dict(meta, trees=128, features=self.FEATURE_NAMES, lgbm=dict(CONFIG["lgbm"], seed=meta["learner"]),
                    feature_precision="raw_float32_PIT_float64_final_float32")
        (out_dir / "model.json").write_text(json.dumps(info, indent=1) + "\n", encoding="utf-8")


if __name__ == "__main__":
    driver.main(PatternRemoved())
