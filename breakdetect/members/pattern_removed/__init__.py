"""The "Pattern removed" member: gradient boosting on seasonal residuals, 9 transforms x 5 windows + 18 detectors."""

from __future__ import annotations

from pathlib import Path

import numpy as np

from .features import FeatureState
from .model import FoldPIT, NumpyTrees


class PatternRemoved:
    """Three learner seeds on one shared feature state; the feature vector is also exposed to heads."""

    feature_inputs: tuple[str, ...] = ()

    def __init__(self, model_dirs) -> None:
        self.heads = tuple((NumpyTrees(Path(d) / "trees.npz"), FoldPIT(Path(d) / "pit.npz")) for d in model_dirs)

    def start(self, history) -> FeatureState:
        return FeatureState(history)

    def step(self, state: FeatureState, x: float, inputs=()) -> tuple[list[float], np.ndarray]:
        """Probability of each seed and the float64 feature vector of the point."""
        v = state.step(x)
        raws = [float(trees.predict_one(pit.apply_one(v, state.t, state.n_hist))) for trees, pit in self.heads]
        return raws, v
