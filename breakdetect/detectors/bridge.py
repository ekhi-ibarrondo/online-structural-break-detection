"""Brownian-bridge variance traces on AR-whitened residuals.

The online residuals are prewhitened with the AR model of the history, transformed into a
squared z (``br_var_w``) or the mid-rank of |z| in the history (``br_rankvar_w``), and the
weighted maximum deviation of their partial sums from a straight line is reported.
"""

from __future__ import annotations

import math

import numpy as np

from .common import ar_residuals, standardize, yule_walker

MAX_AR_P = 5
MIN_HIST_FIT = 32
EPS = 1e-12
SD_FLOOR = 1e-6
Z2_CLIP = 50.0
W_FLOOR = 0.02
BRIDGE_EXP = 0.25
BUF_START = 64


def _bridge_stat(c1: np.ndarray, m: int, s_h: float) -> float:
    """max_k |S_k - (k/m) S_m| / w(k/m), with w(u) = max(u (1 - u), 0.02)^(1/4), scaled by sqrt(m) s_h."""
    k = np.arange(1.0, m + 1.0)
    q = float(c1[m - 1]) / m
    frac = k / m
    w = np.maximum(frac * (1.0 - frac), W_FLOOR) ** BRIDGE_EXP
    dev = np.abs(c1[:m] - q * k) / w
    return float(dev.max()) / (math.sqrt(m) * s_h)


def _rank_u(sorted_ref: np.ndarray, v) -> np.ndarray:
    """Mid-rank PIT (lo + 0.5 ties + 0.5) / (n + 1) against a sorted reference."""
    n = sorted_ref.shape[0]
    lo = np.searchsorted(sorted_ref, v, side="left")
    hi = np.searchsorted(sorted_ref, v, side="right")
    return (lo + 0.5 * (hi - lo) + 0.5) / (n + 1.0)


class BridgeTrace:
    """Weighted Brownian-bridge statistic of a whitened variance transform."""

    def __init__(self, name: str, history) -> None:
        self._rank = name == "br_rankvar_w"
        x_h = np.asarray(history, dtype=np.float64)
        if x_h.ndim != 1 or x_h.size == 0:
            raise ValueError("empty or non-1D historical segment")
        n_hist = int(x_h.shape[0])
        if n_hist < MIN_HIST_FIT:
            raise ValueError(f"historical segment must have >= {MIN_HIST_FIT} points")
        self._mu, self._sd, a_h = standardize(x_h)
        p = min(MAX_AR_P, max(1, n_hist // 20))
        coefs_t, _ = yule_walker(a_h, p)
        self._coefs = [float(v) for v in coefs_t]
        e_h = ar_residuals(a_h, self._coefs)
        self._n_ref = int(e_h.shape[0])
        mu_e = float(e_h.mean())
        sd_e = float(e_h.std())
        if not (math.isfinite(mu_e) and math.isfinite(sd_e)):
            raise ValueError("non-finite residual statistics")
        self._mu_e = mu_e
        self._sd_e = max(sd_e, EPS)
        zh = (e_h - self._mu_e) / self._sd_e
        self._sorted_ref = None
        if self._rank:
            ah = np.abs(zh)
            self._sorted_ref = np.sort(ah)
            gh = _rank_u(self._sorted_ref, ah)
        else:
            gh = np.minimum(zh * zh, Z2_CLIP)
        self._s_h = max(float(gh.std()), SD_FLOOR)
        self._mean_h = float(gh.mean())
        if not (math.isfinite(self._s_h) and math.isfinite(self._mean_h)):
            raise ValueError("non-finite transform statistics")
        self._phi = [float(a_h[n_hist - 1 - j]) for j in range(p)]
        self._buf = np.empty(BUF_START, dtype=np.float64)
        self._t = 0

    def step(self, x: float) -> float:
        """Consume one online point and return the bridge statistic."""
        a = (float(x) - self._mu) / self._sd
        if not math.isfinite(a):
            raise ValueError("non-finite online point")
        phi = self._phi
        e = a
        for j in range(len(self._coefs)):
            e -= self._coefs[j] * phi[j]
        for j in range(len(phi) - 1, 0, -1):
            phi[j] = phi[j - 1]
        phi[0] = a

        z = (e - self._mu_e) / self._sd_e
        if self._rank:
            v = abs(z)
            lo = int(np.searchsorted(self._sorted_ref, v, side="left"))
            hi = int(np.searchsorted(self._sorted_ref, v, side="right"))
            g = (lo + 0.5 * (hi - lo) + 0.5) / (self._n_ref + 1.0)
        else:
            g = min(z * z, Z2_CLIP)

        t = self._t
        buf = self._buf
        if t >= buf.shape[0]:
            grown = np.empty(buf.shape[0] * 2, dtype=np.float64)
            grown[:t] = buf[:t]
            self._buf = buf = grown
        prev = float(buf[t - 1]) if t > 0 else 0.0
        buf[t] = prev + g
        m = t + 1
        self._t = m
        if m == 1:
            return abs(g - self._mean_h) / self._s_h
        return _bridge_stat(buf, m, self._s_h)
