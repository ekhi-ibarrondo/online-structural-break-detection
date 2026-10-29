"""The 22 detector traces read by the reads_two_rebuilds and reads_three_others members, computed point by point."""

from __future__ import annotations

import numpy as np

from .bridge import BridgeTrace
from .garch import GarchResidualTrace, GarchScoreTrace
from .meandep import MeanDepTrace
from .score import ScoreTrace
from .shape import ShapeTrace
from .variance import VarianceTrace

GARCH_SCORE_NAMES = ("garch_q", "garch_hr", "garch_cusum", "garch_lvl")


def _build(history: list[float]) -> list[tuple[tuple[str, ...], object]]:
    """Every trace with the names of the values it emits, in a fixed order."""
    traces: list[tuple[tuple[str, ...], object]] = []
    for name in ("focus_var", "sanso_k2", "mix_sr_var"):
        traces.append(((name,), VarianceTrace((name,), history)))
    for name in ("hermite", "pot", "qms"):
        traces.append(((name,), ShapeTrace(name, history)))
    for name in ("focus_mean", "ar_local", "sign_dep"):
        traces.append(((name,), MeanDepTrace(name, history)))
    traces.append((("ar_split",), MeanDepTrace("ar_local", history, split=True)))
    traces.append((GARCH_SCORE_NAMES, GarchScoreTrace(history)))
    traces.append((("garch_resid",), GarchResidualTrace(history)))
    for name in ("bn_fs", "bn_hs"):
        traces.append(((name,), ScoreTrace(name, history)))
    for name in ("br_var_w", "br_rankvar_w"):
        traces.append(((name,), BridgeTrace(name, history)))
    traces.append((("mix_sr_mean",), MeanDepTrace("mix_sr_mean", history)))
    for name in ("bn_f", "bn_h"):
        traces.append(((name,), ScoreTrace(name, history)))
    return traces


class TraceSet:
    """Per-series state of all traces; ``step`` returns their values in the order of ``keys``."""

    def __init__(self, history, keys) -> None:
        self._traces = _build(np.asarray(history, np.float64).tolist())
        self._keys = list(keys)

    def step(self, x: float) -> list[float]:
        values: dict[str, float] = {}
        for names, trace in self._traces:
            out = trace.step(x)
            if len(names) == 1:
                values[names[0]] = out
            else:
                values.update(zip(names, out, strict=True))
        return [values[k] for k in self._keys]
