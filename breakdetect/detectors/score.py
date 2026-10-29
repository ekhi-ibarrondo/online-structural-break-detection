"""Whitened AR(p) score traces (local-approach statistics).

The AR model is fitted on the history; each online point gives a (p + 1)-vector of scores
(lag and variance directions) that is centred, whitened with the historical score covariance
(plain or Bartlett long-run) and accumulated. The statistic is either the squared norm of the
cumulative sum divided by t, or the best such value over a grid of recent suffixes.
"""

from __future__ import annotations

import math

import numpy as np

from ..kernels import bn as kernels
from .common import SIGMA2_FLOOR, ar_residuals, hand_cholesky, standardize, yule_walker

MAX_AR_P = 5
RIDGE_REL = 1e-6
RIDGE_ESCALATION = (1.0, 1e2, 1e4)
MIN_HIST_FIT = 32
SUFFIX_CAPACITY = 1024
VARIANTS = {"bn_f": (False, False), "bn_h": (True, False), "bn_fs": (False, True), "bn_hs": (True, True)}


def suffix_grid(n: int) -> list[int]:
    """Suffix lengths 1..16, then geometric with ratio 1.25, up to ``n``."""
    vals = list(range(1, min(n, 16) + 1))
    cur = 16.0
    while vals and vals[-1] < n:
        cur *= 1.25
        nxt = max(vals[-1] + 1, round(cur))
        if nxt > n:
            break
        vals.append(nxt)
    return sorted(set(vals))


def _bartlett_lrv(xc: np.ndarray) -> np.ndarray:
    """Bartlett long-run covariance of centred rows."""
    m = xc.shape[0]
    lag = min(m - 1, max(1, math.floor(4.0 * (m / 100.0) ** (2.0 / 9.0))))
    cov = xc.T @ xc / m
    for j in range(1, lag + 1):
        gamma = xc[j:].T @ xc[:-j] / m
        cov = cov + (1.0 - j / (lag + 1.0)) * (gamma + gamma.T)
    return cov


def _whitener(sigma: np.ndarray) -> list[list[float]]:
    """Cholesky factor of the score covariance, with an escalating ridge if needed."""
    d = sigma.shape[0]
    tr = 0.0
    for i in range(d):
        tr += float(sigma[i, i])
    lam_base = RIDGE_REL * (tr / d)
    err: ValueError | None = None
    for factor in RIDGE_ESCALATION:
        lam = lam_base * factor
        g = [[float(sigma[i, j]) + (lam if i == j else 0.0) for j in range(d)] for i in range(d)]
        try:
            return hand_cholesky(g)
        except ValueError as exc:
            err = exc
    raise ValueError("score covariance not factorizable after ridge escalation") from err


class ScoreTrace:
    """One whitened AR score statistic: ``hac`` selects the long-run covariance, ``suffix`` the scan."""

    def __init__(self, name: str, history) -> None:
        hac, suffix = VARIANTS[name]
        x_h = np.asarray(history, dtype=np.float64)
        if x_h.ndim != 1 or x_h.size == 0:
            raise ValueError("empty or non-1D historical segment")
        n_hist = int(x_h.shape[0])
        if n_hist < MIN_HIST_FIT:
            raise ValueError(f"historical segment must have >= {MIN_HIST_FIT} points")
        mu_h, sd_h, a_hist = standardize(x_h)

        p = min(MAX_AR_P, max(1, n_hist // 20))
        coefs_t, _ = yule_walker(a_hist, p)
        coefs = [float(v) for v in coefs_t]
        e_hist = ar_residuals(a_hist, coefs)
        sigma2 = max(float(np.mean(e_hist * e_hist)), SIGMA2_FLOOR)
        inv_s2 = 1.0 / sigma2

        m = n_hist - p
        lags = np.empty((m, p), dtype=np.float64)
        for j in range(1, p + 1):
            lags[:, j - 1] = a_hist[p - j : n_hist - j]
        scores = np.empty((m, p + 1), dtype=np.float64)
        scores[:, :p] = e_hist[:, None] * lags * inv_s2
        scores[:, p] = e_hist * e_hist * inv_s2 - 1.0
        center = scores.mean(axis=0)
        xc = scores - center
        ell = _whitener(_bartlett_lrv(xc) if hac else xc.T @ xc / m)

        d = p + 1
        if suffix:
            cs_hist = np.zeros((SUFFIX_CAPACITY + 1, d), dtype=np.float64)
            grid = np.asarray(suffix_grid(SUFFIX_CAPACITY), dtype=np.int64)
        else:
            cs_hist = np.zeros((1, d), dtype=np.float64)
            grid = np.zeros(0, dtype=np.int64)
        self._suffix = suffix
        self._args = [
            float(mu_h),
            float(sd_h),
            np.asarray(coefs, dtype=np.float64),
            float(inv_s2),
            np.asarray([float(v) for v in center], dtype=np.float64),
            np.ascontiguousarray(np.asarray(ell, dtype=np.float64)),
            np.asarray([float(a_hist[n_hist - 1 - j]) for j in range(p)], dtype=np.float64),
            np.zeros(d, dtype=np.float64),
            np.array([0], dtype=np.int64),
            1 if suffix else 0,
            cs_hist,
            grid,
            np.zeros(d, dtype=np.float64),
            np.zeros(d, dtype=np.float64),
        ]

    def _grow(self, t_next: int) -> None:
        """Double the cumulative-sum history (and its suffix grid) to hold step ``t_next``."""
        cs_hist = self._args[10]
        cap = int(cs_hist.shape[0]) - 1
        while cap < t_next:
            cap *= 2
        grown = np.zeros((cap + 1, cs_hist.shape[1]), dtype=np.float64)
        grown[: cs_hist.shape[0]] = cs_hist
        self._args[10] = grown
        self._args[11] = np.asarray(suffix_grid(cap), dtype=np.int64)

    def step(self, x: float) -> float:
        """Consume one online point and return the statistic."""
        xf = float(x)
        t = int(self._args[8][0])
        if self._suffix and t + 1 > self._args[10].shape[0] - 1:
            self._grow(t + 1)
        stat, status = kernels.bn_step(xf, *self._args)
        if status == 1:
            raise ValueError("non-finite online point")
        return stat
