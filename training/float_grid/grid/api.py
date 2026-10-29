"""Independent per-series state and float32 wire conversion."""

import math

import numpy as np

from . import kernels as k
from .constants import BAND_LIMITS, MAX_CHANCE_RATE, MIN_FIT_FRACTION, MIN_STEP_ULPS

SCORE_NAME = "membership_llr"
FORMAT = "float_grid/v2"
TRANSCENDENTALS = {"log": {"where": "step", "per_support_point": 1, "per_band_point": 1, "per_none_point": 0,
                           "per_fit": 0}}


def _array(values):
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


def _scalar(value):
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
    """A support fit and recurrence owned exclusively by one series."""

    __slots__ = ("_support", "_support_ulp", "_params", "_bands", "_t")

    def __init__(self, x_hist):
        history = _array(x_hist)
        self._support = np.empty(len(history), np.float64)
        self._support_ulp = np.empty(len(history), np.float64)
        self._params = np.empty(k.PARAM_COUNT, np.float64)
        self._bands = np.empty((len(BAND_LIMITS), k.BAND_COLUMNS), np.float64)
        k.fit_cascade(history, self._support, self._support_ulp, self._params, self._bands)
        self._t = 0

    @classmethod
    def init(cls, x_hist):
        return cls(x_hist)

    def update(self, x_t):
        result = k.step(_scalar(x_t), self._t, self._support, self._support_ulp, self._params)
        self._t += 1
        return float(result[0])

    @property
    def t(self):
        return self._t

    @property
    def detected(self):
        return bool(self._params[k.DETECTED])

    @property
    def mode(self):
        return ("none", "support", "band")[int(self._params[k.MODE])]

    @property
    def band_limit(self):
        return float(self._params[k.BAND_LIMIT])

    @property
    def step(self):
        return float(self._params[k.STEP])

    @property
    def rel(self):
        return float(self._params[k.REL])

    @property
    def fit_fraction(self):
        return float(self._params[k.FRACTION])

    @property
    def chance_rate(self):
        return float(self._params[k.CHANCE])

    @property
    def off_rate_h0(self):
        return float(self._params[k.OFF_RATE])

    @property
    def reason(self):
        if self.mode == "band":
            return "band_detected"
        return k.REASONS[int(self._params[k.REASON])]


def batch_series(x_hist, x_online):
    state = SupportStream(x_hist)
    online = _array(x_online)
    output = np.empty(len(online), np.float64)
    k.batch_steps(online, state._support, state._support_ulp, state._params, output)
    return output


def _fit_fields(params):
    fraction = float(params[k.FRACTION])
    chance = float(params[k.CHANCE])
    umed = float(params[k.UMED])
    n = int(params[k.GAP_COUNT])
    ok_count = int(params[k.OK_COUNT])
    fraction_margin = fraction - MIN_FIT_FRACTION
    chance_margin = MAX_CHANCE_RATE - chance
    step_ratio = float(params[k.STEP]) / (MIN_STEP_ULPS * umed)
    step_margin = step_ratio - 1.0
    changes = None
    if math.isfinite(fraction) and n:
        # Search integer counts using the detector's exact comparison, including ties.
        lo, hi = 0, n
        while lo < hi:
            mid = (lo + hi) // 2
            if mid / n >= MIN_FIT_FRACTION:
                hi = mid
            else:
                lo = mid + 1
        changes = ok_count - lo + 1 if fraction >= MIN_FIT_FRACTION else lo - ok_count
    return {
        "detected": bool(params[k.DETECTED]),
        "step": float(params[k.STEP]),
        "rel": float(params[k.REL]),
        "fit_fraction": fraction,
        "chance_rate": chance,
        "off_rate_h0": float(params[k.OFF_RATE]),
        "reason": k.REASONS[int(params[k.REASON])],
        "n_support": int(params[k.COUNT]),
        "n_fit_gaps": n,
        "n_ok_gaps": ok_count,
        "median_gap_ulp": umed,
        "fraction_margin": fraction_margin,
        "chance_margin": chance_margin,
        "step_margin": step_margin,
        "step_ratio": step_ratio,
        "fraction_flip_count": changes,
        "minimum_relative_margin": min(
            fraction_margin / MIN_FIT_FRACTION, chance_margin / MAX_CHANCE_RATE, step_margin
        ),
    }


def _summary(state):
    result = _fit_fields(state._params)
    bands = []
    for row in state._bands:
        if math.isnan(row[k.BAND_STATUS]):
            break
        evaluated = bool(row[k.BAND_STATUS])
        band = _fit_fields(row) if evaluated else {"reason": "few_band_values"}
        band.update({"limit": float(row[k.BAND_LIMIT]), "n_values": int(row[k.BAND_VALUE_COUNT]),
                     "evaluated": evaluated, "ok": bool(row[k.DETECTED]) if evaluated else False,
                     "k": int(row[k.BAND_K]) if evaluated else 0})
        bands.append(band)
    result.update({"mode": state.mode, "band_limit": state.band_limit, "reason": state.reason,
                   "band_k": int(state._params[k.BAND_K]), "bands": bands,
                   "band_minimum_relative_margin": result["minimum_relative_margin"]
                   if state.mode == "band" else math.nan})
    return result


def fit_summary(x_hist):
    return _summary(SupportStream(x_hist))


def _minimum(values):
    finite = values[np.isfinite(values)]
    return float(finite.min()) if finite.size else math.nan


def diagnostics(x_hist, x_online):
    state = SupportStream(x_hist)
    summary = _summary(state)
    online = _array(x_online)
    output = np.empty((len(online), 9), np.float64)
    k.diagnostic_steps(online, state._support, state._support_ulp, state._params, output)
    margins = np.empty(len(online), np.float64)
    for i in range(len(online)):
        tol = float(output[i, 7])
        margins[i] = float(output[i, 6]) / math.ulp(tol) if math.isfinite(tol) else math.nan
    return {
        "fit": summary,
        "scores": output[:, 0].copy(),
        "membership": output[:, 1].astype(bool),
        "q": output[:, 2].copy(),
        "lam": output[:, 3].copy(),
        "R": output[:, 4].copy(),
        "log_argument": output[:, 5].copy(),
        "membership_margin_ulps": margins,
        "rounding_margin": output[:, 8].copy(),
        "in_band": np.isfinite(online) & (np.abs(online) < state.band_limit)
        if state.mode == "band" else np.isfinite(online),
        "minimum_membership_margin_ulps": _minimum(margins),
        "minimum_absolute_membership_margin_ulps": _minimum(np.abs(margins)),
        "minimum_rounding_margin": _minimum(output[:, 8]),
        "points_within_four_ulps": int(np.count_nonzero(np.abs(margins) < 4.0)),
    }
