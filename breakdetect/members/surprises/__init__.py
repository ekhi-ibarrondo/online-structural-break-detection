"""The "Surprises" member: gradient boosting on AR-whitened windows."""

from __future__ import annotations

import numpy as np

from .features import N_FEATURES, StreamState, fit_hist, stream_step
from .trees import load_model, predict_raw_single


class Surprises:
    """Three learner seeds on one shared feature state; the feature vector is also exposed to heads."""

    feature_inputs: tuple[str, ...] = ()

    def __init__(self, model_dirs) -> None:
        self.models = tuple(load_model(d) for d in model_dirs)
        if any(int(m["meta"][2]) != N_FEATURES for m in self.models):
            raise ValueError("model / feature dimension mismatch")

    def start(self, history) -> StreamState:
        return StreamState(fit_hist(np.asarray(history, dtype=np.float64)))

    def step(self, state: StreamState, x: float, inputs=()) -> tuple[list[float], np.ndarray]:
        """Probability of each seed and the float32 feature vector of the point."""
        f32 = stream_step(state, float(x))
        raws = []
        for model in self.models:
            raw = predict_raw_single(model, f32.astype(np.float64))
            p = 1.0 / (1.0 + np.exp(np.array([-raw])))[0]
            raws.append(float(p))
        return raws, f32
