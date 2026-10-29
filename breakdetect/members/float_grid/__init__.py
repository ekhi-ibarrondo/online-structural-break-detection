"""float_grid: float-grid detector.

In this dataset the history of about 2% of the series lies on a float32 grid, and the
points after the break do not. The history fixes that grid; each online point is scored by how much more
likely it is to lie on the grid under the fitted support than by chance. There is no
supervised training: the model file only stores the configuration constants.
"""

from __future__ import annotations

import json
import math
from pathlib import Path

import numpy as np

from ...kernels import float_grid as k

FORMAT = "float_grid/v2"
SCORE_NAME = "membership_llr"


def _array(values) -> np.ndarray:
    """1-D float64 array of the float32-rounded values."""
    try:
        with np.errstate(over="ignore", invalid="ignore"):
            arr = np.asarray(values, dtype=np.float32)
    except OverflowError:
        objects = np.asarray(values, dtype=object)
        if objects.ndim != 1:
            raise ValueError("Expected a one-dimensional sequence") from None
        return np.asarray([_scalar(value) for value in objects], dtype=np.float64)
    if arr.ndim != 1:
        raise ValueError("Expected a one-dimensional sequence")
    return np.ascontiguousarray(arr, dtype=np.float64)


def _scalar(value) -> float:
    """A value rounded to float32 (out-of-range integers become +-inf)."""
    if np.ndim(value) != 0:
        raise ValueError("Expected a scalar")
    try:
        number = float(value)
    except OverflowError:
        if not isinstance(value, (int, np.integer)):
            raise
        number = -math.inf if value < 0 else math.inf
    with np.errstate(over="ignore", invalid="ignore"):
        return float(np.float32(number))


class SupportStream:
    """Fitted value support and running score of one series."""

    __slots__ = ("_params", "_support", "_support_ulp", "_t")

    def __init__(self, x_hist) -> None:
        history = _array(x_hist)
        self._support = np.empty(len(history), np.float64)
        self._support_ulp = np.empty(len(history), np.float64)
        self._params = np.empty(k.PARAM_COUNT, np.float64)
        bands = np.empty((len(k.BAND_LIMITS), k.BAND_COLUMNS), np.float64)
        k.fit_cascade(history, self._support, self._support_ulp, self._params, bands)
        self._t = 0

    def update(self, x_t) -> float:
        """Score of the next online point."""
        result = k.step(_scalar(x_t), self._t, self._support, self._support_ulp, self._params)
        self._t += 1
        return float(result[0])


class FloatGrid:
    """Scorer wrapper: validates the configuration constants against the compiled ones."""

    feature_inputs: tuple[str, ...] = ()

    def __init__(self, model_dirs) -> None:
        (model_dir,) = model_dirs
        config = json.loads((Path(model_dir) / "float_grid.json").read_text(encoding="utf-8"))
        if config.get("format") != FORMAT or config.get("score") != SCORE_NAME:
            raise ValueError("Model format or score does not match")
        if config.get("constants") != k.constants_hex():
            raise ValueError("Model constants do not match exactly")

    def start(self, history) -> SupportStream:
        return SupportStream(history)

    def step(self, state: SupportStream, x: float, inputs=()) -> tuple[list[float], None]:
        return [state.update(x)], None
