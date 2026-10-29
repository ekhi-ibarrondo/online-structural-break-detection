"""GARCH(1,1) traces: score statistics of the fitted model and variance detectors on its residuals."""

from __future__ import annotations

import math

import numpy as np

from ..kernels import garch as kernels
from .common import standardize
from .variance import DETECTORS, VarianceTrace

ALPHA_GRID = (0.005, 0.015, 0.03, 0.05, 0.08, 0.12, 0.17, 0.24, 0.33, 0.45)
BETA_GRID = (0.10, 0.30, 0.50, 0.65, 0.75, 0.82, 0.86, 0.89, 0.92, 0.94, 0.96, 0.975)
_REFINE_ALPHA_FACTORS = (0.75, 0.875, 1.0, 1.143, 1.333)
_REFINE_BETA_OFFSETS = (-0.04, -0.02, 0.0, 0.02, 0.04)
_ALPHA_MIN, _ALPHA_MAX = 0.005, 0.60
_BETA_MIN, _BETA_MAX = 0.05, 0.985
PERSIST_MAX = 0.999
U_CAP = 50.0
KAPPA_HR = 0.45
FIT_MAX = 3000
MIN_HIST_FIT = 32
FALLBACK_AB = (0.05, 0.90)
RIDGE_REL = 1e-8
RIDGE_ESCALATION = (1.0, 1e2, 1e4)


def _coarse_candidates() -> tuple[np.ndarray, np.ndarray]:
    """Every (alpha, beta) pair of the coarse grid with alpha + beta <= 0.999."""
    alphas, betas = [], []
    for a in ALPHA_GRID:
        for b in BETA_GRID:
            if a + b <= PERSIST_MAX:
                alphas.append(a)
                betas.append(b)
    return np.asarray(alphas, dtype=np.float64), np.asarray(betas, dtype=np.float64)


_COARSE_ALPHAS, _COARSE_BETAS = _coarse_candidates()


def _refine_candidates(a0: float, b0: float) -> tuple[np.ndarray, np.ndarray]:
    """Local grid around the best coarse candidate."""
    alphas, betas = [], []
    seen: set[tuple[float, float]] = set()
    for f in _REFINE_ALPHA_FACTORS:
        a = min(max(a0 * f, _ALPHA_MIN), _ALPHA_MAX)
        for db in _REFINE_BETA_OFFSETS:
            b = min(max(b0 + db, _BETA_MIN), _BETA_MAX)
            if a + b <= PERSIST_MAX and (a, b) not in seen:
                seen.add((a, b))
                alphas.append(a)
                betas.append(b)
    return np.asarray(alphas, dtype=np.float64), np.asarray(betas, dtype=np.float64)


def _nll(a_fit: np.ndarray, alphas: np.ndarray, betas: np.ndarray) -> np.ndarray:
    """Mean negative log-likelihood per candidate, non-finite values replaced by inf."""
    out = np.empty(alphas.shape[0], dtype=np.float64)
    kernels.nll_candidates(np.ascontiguousarray(a_fit, dtype=np.float64), alphas, betas, out)
    return np.where(np.isfinite(out), out, np.inf)


def fit_garch11(a_hist: np.ndarray) -> tuple[float, float]:
    """Unit-variance GARCH(1,1) by a coarse and a refined likelihood grid (last 3000 points)."""
    n = a_hist.shape[0]
    if n < MIN_HIST_FIT:
        return FALLBACK_AB
    a_fit = a_hist[-FIT_MAX:] if n > FIT_MAX else a_hist
    with np.errstate(all="ignore"):
        nll_c = _nll(a_fit, _COARSE_ALPHAS, _COARSE_BETAS)
        if not np.isfinite(nll_c).any():
            return FALLBACK_AB
        i0 = int(np.argmin(nll_c))
        a0, b0 = float(_COARSE_ALPHAS[i0]), float(_COARSE_BETAS[i0])
        ra, rb = _refine_candidates(a0, b0)
        nll_r = _nll(a_fit, ra, rb)
        i1 = int(np.argmin(nll_r))
        if not math.isfinite(float(nll_r[i1])):
            return a0, b0
        return float(ra[i1]), float(rb[i1])


def _chol3(g: list[list[float]], lam: float) -> tuple[float, ...] | None:
    """Packed Cholesky factor of g + lam I, or None if it is not positive definite."""
    a11 = g[0][0] + lam
    if a11 <= 0.0:
        return None
    l11 = math.sqrt(a11)
    l21 = g[1][0] / l11
    l31 = g[2][0] / l11
    a22 = g[1][1] + lam - l21 * l21
    if a22 <= 0.0:
        return None
    l22 = math.sqrt(a22)
    l32 = (g[2][1] - l31 * l21) / l22
    a33 = g[2][2] + lam - l31 * l31 - l32 * l32
    if a33 <= 0.0:
        return None
    l33 = math.sqrt(a33)
    return (l11, l21, l22, l31, l32, l33)


class GarchScoreTrace:
    """Score-test statistics of the GARCH(1,1) fitted on the history: q, hr, CUSUM and level."""

    def __init__(self, history) -> None:
        mu_h, sd_h, a_hist = standardize(history)
        alpha, beta = fit_garch11(a_hist)
        n_hist = int(a_hist.shape[0])
        hist_arr = np.ascontiguousarray(a_hist, dtype=np.float64)
        scores = np.empty((n_hist, 3), dtype=np.float64)
        center = np.empty(3, dtype=np.float64)
        s2, dw, da, db = kernels.hist_score_pass(hist_arr, alpha, beta, U_CAP, scores, center)
        g_arr = np.empty((3, 3), dtype=np.float64)
        kernels.score_covariance(scores, center, g_arr)
        g = g_arr.tolist()
        tr = g[0][0] + g[1][1] + g[2][2]
        lam_base = RIDGE_REL * (tr / 3.0) + 1e-300
        chol = None
        lam = lam_base
        for factor in RIDGE_ESCALATION:
            lam = lam_base * factor
            chol = _chol3(g, lam)
            if chol is not None:
                break
        self._args = (
            np.array([0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, s2, dw, da, db], dtype=np.float64),
            np.array([0, 0], dtype=np.int64),
            np.zeros(6, dtype=np.float64) if chol is None else np.asarray(chol, dtype=np.float64),
            0 if chol is None else 1,
            float(g[0][0] + lam),
            center,
            n_hist,
            KAPPA_HR,
            U_CAP,
            float(alpha),
            float(beta),
            float(1.0 - alpha - beta),
            float(mu_h),
            float(sd_h),
        )

    def step(self, x: float) -> tuple[float, float, float, float]:
        """Consume one online point and return (q, hr, cusum, level)."""
        return kernels.garch_step(float(x), *self._args)


class GarchResidualTrace:
    """The three variance detectors run on GARCH(1,1)-standardized residuals."""

    def __init__(self, history) -> None:
        mu_h, sd_h, a_hist = standardize(history)
        alpha, beta = fit_garch11(a_hist)
        hist_arr = np.ascontiguousarray(a_hist, dtype=np.float64)
        r_hist = np.empty(hist_arr.shape[0], dtype=np.float64)
        s2_end = kernels.resid_hist(hist_arr, alpha, beta, r_hist)
        self._rs = np.array([s2_end, mu_h, sd_h, alpha, beta, 1.0 - alpha - beta], dtype=np.float64)
        self._variance = VarianceTrace(DETECTORS, r_hist)

    def step(self, x: float) -> float:
        """Consume one online point and return the mean variance z of its residual."""
        return self._variance.step(kernels.resid_step(float(x), self._rs))
