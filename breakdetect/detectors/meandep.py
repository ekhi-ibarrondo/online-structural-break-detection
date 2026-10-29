"""Mean and dependence traces: FOCuS mean, mean-shift mixture, whitened AR score, sign dependence.

The null lanes rotate the standardized history and add a location offset per lane, sized
by the long-run variance of the levels: the online stream is centred with a mean estimated
out of sample, an error that in-sample rotations would not show. The ``split`` variant fits
the AR model on the first half of the history and rotates the (re-centred) second half.
"""

from __future__ import annotations

import math

import numpy as np

from ..kernels import meandep as kernels
from .common import (
    FOCUS_MAX_MAIN,
    FOCUS_MAX_NULL,
    PROBIT_B16,
    PROBIT_B32,
    WINSOR_C,
    ar_residuals,
    bartlett_omega2,
    hand_cholesky,
    null_substreams,
    standardize,
    yule_walker,
)

MEAN_SR_GRID = (-2.0, -1.25, -0.8, -0.5, 0.5, 0.8, 1.25, 2.0)
AR_ORDER = 2
SIGN_LAGS = (1,)
SIGN_K = 0.25
_KINDS = {
    "focus_mean": kernels.KIND_FOCUS_MEAN,
    "mix_sr_mean": kernels.KIND_MIX_MEAN,
    "ar_local": kernels.KIND_AR,
    "sign_dep": kernels.KIND_SIGN,
}


def gs_whitener(coefs) -> list[list[float]]:
    """Cholesky factor of the inverse AR Fisher information (Gohberg-Semencul formula)."""
    a = [float(v) for v in coefs]
    p = len(a)
    if not (1 <= p <= 6):
        raise ValueError(f"need 1 <= p <= 6 AR coefficients, got {p}")
    col1 = [1.0] + [-v for v in a[: p - 1]]
    col2 = [-v for v in reversed(a)]
    g = [[0.0] * p for _ in range(p)]
    for i in range(p):
        for j in range(i + 1):
            acc = 0.0
            for k in range(j + 1):
                acc += col1[i - k] * col1[j - k] - col2[i - k] * col2[j - k]
            g[i][j] = acc
            g[j][i] = acc
    return hand_cholesky(g)


def _sign_p_hats(eps: np.ndarray, lags) -> list[float]:
    """Historical rate of 1(eps_t * eps_{t-lag} > 0) per lag, clamped inside (0, 1)."""
    out = []
    for lag in lags:
        if eps.shape[0] <= lag + 1:
            raise ValueError("historical residual segment shorter than a lag window")
        prod = eps[lag:] * eps[:-lag]
        m = prod.shape[0]
        rate = float(np.count_nonzero(prod > 0.0)) / m
        lo, hi = 1.0 / (m + 1.0), m / (m + 1.0)
        out.append(min(max(rate, lo), hi))
    return out


def _lane_value(src: list[float], n: int, idx: int, g: float, cap: float) -> float:
    """Null-lane value at circular position ``idx``: offset, then winsorized."""
    v = src[idx % n] + g
    if v > cap:
        return cap
    if v < -cap:
        return -cap
    return v


def _seed_eps(src: list[float], n: int, start: int, coefs, m: int, g: float, cap: float) -> list[float]:
    """AR residuals at the ``m`` positions preceding ``start`` (most recent first)."""
    p = len(coefs)
    out = []
    for i in range(1, m + 1):
        pos = start - i
        eps = _lane_value(src, n, pos, g, cap)
        for j in range(p):
            eps -= coefs[j] * _lane_value(src, n, pos - 1 - j, g, cap)
        out.append(eps)
    return out


def _location_offsets(n_sub: int, omega2: float, n_full: int) -> list[float]:
    """Location offset of each null lane: probit quantiles times sqrt(omega2 / N), clipped to 0.75."""
    quantiles = PROBIT_B16 if n_sub == 16 else PROBIT_B32
    scale = math.sqrt(omega2 / n_full)
    out = []
    for q in quantiles:
        d = q * scale
        if d > 0.75:
            d = 0.75
        elif d < -0.75:
            d = -0.75
        out.append(d)
    return out


class MeanDepTrace:
    """One mean or dependence detector, z-scored against offset null lanes."""

    def __init__(self, name: str, history, split: bool = False) -> None:
        kind = _KINDS[name]
        mu_h, sd_h, a_hist = standardize(history)
        n_full = int(a_hist.shape[0])
        cap = WINSOR_C * float(np.max(np.abs(a_hist)))
        omega2 = bartlett_omega2(a_hist)
        if split:
            if n_full < 16:
                raise ValueError("split calibration needs >= 16 historical points")
            half = n_full // 2
            ref_hist = a_hist[:half] - float(a_hist[:half].mean())
            null_arr = a_hist[half:] - float(a_hist[half:].mean())
        else:
            ref_hist = a_hist
            null_arr = a_hist
        n_null = int(null_arr.shape[0])
        starts, _ = null_substreams(n_null, 1.0, injected=False)
        n_sub = len(starts)
        n_streams = 1 + n_sub
        offsets = _location_offsets(n_sub, omega2, n_full)
        null_src = null_arr.tolist()
        hist_list = a_hist.tolist()

        p = 1
        coefs: tuple[float, ...] = ()
        ar_lt = np.zeros((1, 1), dtype=np.float64)
        ar_coefs = np.zeros(1, dtype=np.float64)
        ar_inv_sigma = 0.0
        sg_lags = np.ones(1, dtype=np.int64)
        sg_phats = np.zeros(1, dtype=np.float64)
        sg_sdinv = np.zeros(1, dtype=np.float64)
        sg_k = 0.0
        sg_ring = np.zeros((n_streams, 1), dtype=np.float64)
        if name in ("ar_local", "sign_dep"):
            p = AR_ORDER
            coefs, sigma2 = yule_walker(ref_hist, p)
            sigma = math.sqrt(sigma2)
            eps_ref = ar_residuals(ref_hist, coefs) / sigma
            ar_coefs = np.asarray([float(v) for v in coefs], dtype=np.float64)
            if name == "ar_local":
                ell = gs_whitener(coefs)
                ar_lt = np.zeros((p, p), dtype=np.float64)
                for i in range(p):
                    ar_lt[i, : p - i] = [ell[j][i] for j in range(i, p)]
                ar_inv_sigma = float(1.0 / sigma)
            else:
                p_hats = _sign_p_hats(eps_ref, SIGN_LAGS)
                max_lag = max(SIGN_LAGS)
                sg_lags = np.asarray(SIGN_LAGS, dtype=np.int64)
                sg_phats = np.asarray(p_hats, dtype=np.float64)
                sg_sdinv = np.asarray([1.0 / math.sqrt(v * (1.0 - v)) for v in p_hats], dtype=np.float64)
                sg_k = float(SIGN_K)
                sg_ring = np.zeros((n_streams, max_lag), dtype=np.float64)
                sg_ring[0, :] = _seed_eps(hist_list, n_full, n_full, coefs, max_lag, 0.0, cap)
                for b in range(n_sub):
                    sg_ring[1 + b, :] = _seed_eps(null_src, n_null, starts[b], coefs, max_lag, offsets[b], cap)
        ar_ph = np.zeros((n_streams, p), dtype=np.float64)
        if name in ("ar_local", "sign_dep"):
            ar_ph[0, :] = [float(hist_list[n_full - 1 - j]) for j in range(p)]
            for b in range(n_sub):
                ar_ph[1 + b, :] = [_lane_value(null_src, n_null, starts[b] - 1 - j, offsets[b], cap) for j in range(p)]

        n_units = 1
        cap_m = cap_n = 2
        if name == "focus_mean":
            cap_m, cap_n = FOCUS_MAX_MAIN, FOCUS_MAX_NULL
        elif name == "ar_local":
            n_units = p
            cap_m, cap_n = FOCUS_MAX_MAIN, FOCUS_MAX_NULL
        m_grid = 1
        mx_delta = np.zeros(1, dtype=np.float64)
        mx_c2 = np.zeros(1, dtype=np.float64)
        mx_logw = np.zeros(1, dtype=np.float64)
        mx_add = 0.0
        if name == "mix_sr_mean":
            weights = [1.0 / len(MEAN_SR_GRID)] * len(MEAN_SR_GRID)
            m_grid = len(MEAN_SR_GRID)
            mx_delta = np.asarray(MEAN_SR_GRID, dtype=np.float64)
            mx_c2 = np.asarray([0.5 * v * v for v in MEAN_SR_GRID], dtype=np.float64)
            mx_logw = np.asarray([math.log(v) for v in weights], dtype=np.float64)
            mx_add = float(-math.log1p(-0.0))

        self._args = (
            float(mu_h),
            float(sd_h),
            float(cap),
            int(kind),
            np.zeros(1, dtype=np.int64),
            np.ascontiguousarray(null_arr, dtype=np.float64),
            np.asarray(starts, dtype=np.int64),
            np.asarray(offsets, dtype=np.float64),
            np.zeros(n_sub, dtype=np.float64),
            np.zeros((n_units, 2, cap_m), dtype=np.int64),
            np.zeros((n_units, 2, cap_m), dtype=np.float64),
            np.zeros((n_units, 2, cap_m), dtype=np.float64),
            np.tile(np.array([0, 1, 1], dtype=np.int64), (n_units, 1)),
            np.zeros(n_units, dtype=np.float64),
            np.zeros((n_sub * n_units, 2, cap_n), dtype=np.int64),
            np.zeros((n_sub * n_units, 2, cap_n), dtype=np.float64),
            np.zeros((n_sub * n_units, 2, cap_n), dtype=np.float64),
            np.tile(np.array([0, 1, 1], dtype=np.int64), (n_sub * n_units, 1)),
            np.zeros(n_sub * n_units, dtype=np.float64),
            np.full((n_streams, m_grid), -700.0, dtype=np.float64),
            mx_delta,
            mx_c2,
            mx_logw,
            mx_add,
            ar_ph,
            ar_lt,
            ar_coefs,
            ar_inv_sigma,
            sg_ring,
            np.zeros((n_streams, len(sg_lags)), dtype=np.float64),
            np.zeros((n_streams, len(sg_lags)), dtype=np.float64),
            sg_lags,
            sg_phats,
            sg_sdinv,
            sg_k,
        )

    def step(self, x: float) -> float:
        """Consume one online point and return the standardized statistic."""
        return kernels.meandep_step(float(x), *self._args)
