"""The "Slow and fast" member: spectral gradient boosting."""

from __future__ import annotations

import math
import os

import numpy as np

from .features import N_FEAT, FeatureEngine
from .trees import NumpyGBM


class SlowAndFast:
    """Three learner seeds evaluated on one shared spectral feature state per series."""

    feature_inputs: tuple[str, ...] = ()

    def __init__(self, model_dirs) -> None:
        self.engine = FeatureEngine()
        self.models = tuple(NumpyGBM(os.path.join(d, "model.npz")) for d in model_dirs)
        for gbm in self.models:
            if gbm.n_features != N_FEAT:
                raise ValueError(f"model expects {gbm.n_features} features, engine produces {N_FEAT}")

    def start(self, history):
        return self.engine.start(np.asarray(history, dtype=np.float64))

    def step(self, state, x: float, inputs=()) -> tuple[list[float], None]:
        """Probability of each seed."""
        f = self.engine.step_features(state, float(x))
        raws = []
        for gbm in self.models:
            raw = gbm.raw_single(f)
            if raw >= 0.0:
                raws.append(1.0 / (1.0 + math.exp(-raw)))
            else:
                z = math.exp(raw)
                raws.append(z / (1.0 + z))
        return raws, None
