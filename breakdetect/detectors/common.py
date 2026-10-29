"""Shared pieces of the detector traces: standardization, null sub-streams, long-run variances.

Most detectors are calibrated per series: alongside the online stream they run the same
statistic on B circular rotations of the standardized history (the null lanes) and report
the main statistic as a z-score against the null lanes at the same step.
"""

from __future__ import annotations

import math

import numpy as np

SD_GUARD = 1e-8
WINSOR_C = 1.5
FOCUS_MAX_MAIN = 1024
FOCUS_MAX_NULL = 64
SIGMA2_FLOOR = 1e-12

# Phi^{-1}((b + 1/2) / B) for B = 32 and B = 16, rescaled to unit sample sd (ddof=1).
PROBIT_B32 = (
    -2.1623698782161442,
    -1.6825498643087764,
    -1.4233891289391367,
    -1.234709494810729,
    -1.0817654396985126,
    -0.950515996457287,
    -0.8337865296963051,
    -0.7273719699283163,
    -0.6285684347430053,
    -0.5355135490028449,
    -0.446852048162788,
    -0.3615502946378853,
    -0.278785689898466,
    -0.1978764707515733,
    -0.11823437549699102,
    -0.03933060148386764,
    0.03933060148386764,
    0.11823437549699102,
    0.1978764707515733,
    0.278785689898466,
    0.3615502946378853,
    0.446852048162788,
    0.5355135490028449,
    0.6285684347430053,
    0.7273719699283163,
    0.8337865296963051,
    0.950515996457287,
    1.0817654396985126,
    1.234709494810729,
    1.4233891289391367,
    1.6825498643087764,
    2.1623698782161442,
)
PROBIT_B16 = (
    -1.8766143451956163,
    -1.3278337050343063,
    -1.0175173750282311,
    -0.7822082396158557,
    -0.5834482903635855,
    -0.4052479351112911,
    -0.2389699177118871,
    -0.07899680096204025,
    0.07899680096204025,
    0.2389699177118871,
    0.4052479351112911,
    0.5834482903635855,
    0.7822082396158557,
    1.0175173750282311,
    1.3278337050343063,
    1.8766143451956163,
)


def standardize(history) -> tuple[float, float, np.ndarray]:
    """Mean, sample sd (ddof=1, floored) and the z-scored history."""
    x_h = np.asarray(history, dtype=np.float64)
    if x_h.ndim != 1 or x_h.size == 0:
        raise ValueError("empty or non-1D historical segment")
    mu_h = float(x_h.mean())
    sd_h = float(x_h.std(ddof=1)) if x_h.size > 1 else 1.0
    if not (math.isfinite(mu_h) and math.isfinite(sd_h)):
        raise ValueError("non-finite historical statistics")
    sd_h = max(sd_h, SD_GUARD)
    return mu_h, sd_h, (x_h - mu_h) / sd_h


def null_substreams(n_hist: int, omega4: float, injected: bool) -> tuple[list[int], list[float]]:
    """Start offsets and scale factors of the B circular null sub-streams.

    Sub-stream b reads ``factor_b * hist[(start_b + i) mod N]`` at online step i. With
    ``injected`` the factors spread the variance by ``sqrt(omega4 / N)``, the sampling error
    of a variance estimated from N history points, which the online stream carries and an
    in-sample rotation does not.
    """
    if n_hist < 1:
        raise ValueError("empty historical segment")
    n_sub = 16 if n_hist < 200 else 32
    starts = [(b * n_hist) // n_sub for b in range(n_sub)]
    if not injected:
        return starts, [1.0] * n_sub
    quantiles = PROBIT_B16 if n_sub == 16 else PROBIT_B32
    scale = math.sqrt(omega4 / n_hist)
    factors: list[float] = []
    for q in quantiles:
        d = q * scale
        if d > 0.75:
            d = 0.75
        elif d < -0.75:
            d = -0.75
        factors.append(math.sqrt(1.0 + d))
    return starts, factors


def bartlett_omega4(a_hist: np.ndarray) -> float:
    """Bartlett long-run variance of the squares a_t^2 (bandwidth floor(4 (T/100)^(2/9)))."""
    a = np.asarray(a_hist, dtype=np.float64)
    if a.ndim != 1 or a.shape[0] < 1:
        raise ValueError("a_hist must be a non-empty 1-D array")
    t = a.shape[0]
    a2 = a * a
    d = a2 - a2.mean()
    omega = float(d @ d) / t
    m = min(math.floor(4.0 * (t / 100.0) ** (2.0 / 9.0)), t - 1)
    for lag in range(1, m + 1):
        w = 1.0 - lag / (m + 1.0)
        omega += 2.0 * w * float(d[lag:] @ d[:-lag]) / t
    return max(omega, 1e-6)


def bartlett_omega4_prewhitened(a_hist: np.ndarray) -> float:
    """AR(1)-prewhitened variant of :func:`bartlett_omega4` (recoloured by 1 / (1 - phi)^2)."""
    a = np.asarray(a_hist, dtype=np.float64)
    if a.ndim != 1 or a.shape[0] < 1:
        raise ValueError("a_hist must be a non-empty 1-D array")
    if a.shape[0] < 8:
        return bartlett_omega4(a)
    a2 = a * a
    d = a2 - a2.mean()
    r0 = float(d @ d)
    if r0 <= 0.0:
        return 1e-6
    rhos = [float(d[k:] @ d[:-k]) / r0 for k in range(1, 7)]
    den = sum(rhos[:-1])
    num = sum(rhos[1:])
    significant = max(0.10, 4.5 * math.sqrt(5.0 / d.shape[0]))
    if den > significant and num > 0.0 and rhos[0] > 0.0:
        phi = min(num / den, 0.97)
    else:
        phi = max(min(rhos[0], 0.97), 0.0)
    e = d[1:] - phi * d[:-1]
    e = e - e.mean()
    t = e.shape[0]
    omega_e = float(e @ e) / t
    m = min(math.floor(4.0 * (t / 100.0) ** (2.0 / 9.0)), t - 1)
    for lag in range(1, m + 1):
        w = 1.0 - lag / (m + 1.0)
        omega_e += 2.0 * w * float(e[lag:] @ e[:-lag]) / t
    omega = omega_e / ((1.0 - phi) * (1.0 - phi))
    if not math.isfinite(omega):
        fb = bartlett_omega4(a)
        return fb if math.isfinite(fb) else 1e-6
    return max(omega, 1e-6)


def bartlett_omega2(a_hist: np.ndarray) -> float:
    """Bartlett long-run variance of the levels a_t."""
    a = np.asarray(a_hist, dtype=np.float64)
    if a.ndim != 1 or a.shape[0] < 1:
        raise ValueError("a_hist must be a non-empty 1-D array")
    t = a.shape[0]
    d = a - a.mean()
    omega = float(d @ d) / t
    m = min(math.floor(4.0 * (t / 100.0) ** (2.0 / 9.0)), t - 1)
    for lag in range(1, m + 1):
        w = 1.0 - lag / (m + 1.0)
        omega += 2.0 * w * float(d[lag:] @ d[:-lag]) / t
    if not math.isfinite(omega):
        return 1e-6
    return max(omega, 1e-6)


def yule_walker(a_hist: np.ndarray, p: int) -> tuple[tuple[float, ...], float]:
    """AR(p) by Levinson-Durbin on the biased autocovariances (reflection coefficients clamped to 0.98)."""
    if not (1 <= p <= 6):
        raise ValueError(f"p must be in [1, 6], got {p}")
    a = np.asarray(a_hist, dtype=np.float64)
    if a.ndim != 1 or a.shape[0] < max(8, 2 * p):
        raise ValueError(f"a_hist must be 1-D with >= max(8, 2p) points, got {a.shape}")
    n = a.shape[0]
    d = a - a.mean()
    r = [float(d[k:] @ d[:-k]) / n if k else float(d @ d) / n for k in range(p + 1)]
    if not all(math.isfinite(v) for v in r) or r[0] <= SIGMA2_FLOOR:
        raise ValueError("degenerate or non-finite autocovariances")
    coefs: list[float] = []
    e = r[0]
    for m in range(1, p + 1):
        acc = r[m]
        for j in range(m - 1):
            acc -= coefs[j] * r[m - 1 - j]
        kappa = acc / e
        if kappa > 0.98:
            kappa = 0.98
        elif kappa < -0.98:
            kappa = -0.98
        new = [coefs[j] - kappa * coefs[m - 2 - j] for j in range(m - 1)]
        new.append(kappa)
        coefs = new
        e *= 1.0 - kappa * kappa
    sigma2 = max(e, SIGMA2_FLOOR)
    return tuple(coefs), sigma2


def ar_residuals(src: np.ndarray, coefs) -> np.ndarray:
    """Residuals eps_i = src[i] - sum_j a_j src[i-1-j] for i in [p, n)."""
    a = np.asarray(src, dtype=np.float64)
    p = len(coefs)
    n = a.shape[0]
    if n <= p:
        raise ValueError("source shorter than the AR order")
    eps = a[p:].copy()
    for j, cj in enumerate(coefs):
        eps -= cj * a[p - 1 - j : n - 1 - j]
    return eps


def hand_cholesky(g: list[list[float]]) -> list[list[float]]:
    """Lower Cholesky factor in plain Python floats (no BLAS, fixed operation order)."""
    p = len(g)
    ell = [[0.0] * p for _ in range(p)]
    for i in range(p):
        for j in range(i + 1):
            acc = g[i][j]
            for k in range(j):
                acc -= ell[i][k] * ell[j][k]
            if i == j:
                if acc <= SIGMA2_FLOOR:
                    raise ValueError("whitener matrix is not positive definite")
                ell[i][j] = math.sqrt(acc)
            else:
                ell[i][j] = acc / ell[j][j]
    return ell
