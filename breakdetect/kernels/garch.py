"""Compiled pieces of the GARCH(1,1) traces: grid likelihood, score statistics, residual filter."""

from __future__ import annotations

import math

import numpy as np
from numba import types

from .runtime import A1, A2, F8, I1, I8, kernel

S2_FLOOR = 1e-12
B_MAG = 1.0

# Layout of the float state of ``garch_step``.
FA_S0 = 0
FA_C0 = 3
FA_G_CUSUM = 6
FA_S2 = 7
FA_DW = 8
FA_DA = 9
FA_DB = 10
IA_K = 0
IA_NC = 1


@kernel(types.void(A1, A1, A1, A1))
def nll_candidates(a_fit, alphas, betas, out):
    """Mean Gaussian negative log-likelihood of a unit-variance GARCH(1,1) per (alpha, beta) candidate."""
    n = a_fit.shape[0]
    c = alphas.shape[0]
    omegas = np.empty(c, dtype=np.float64)
    s2 = np.empty(c, dtype=np.float64)
    acc = np.empty(c, dtype=np.float64)
    for j in range(c):
        omegas[j] = (1.0 - alphas[j]) - betas[j]
        s2[j] = 1.0
        acc[j] = 0.0
    for t in range(n):
        v = a_fit[t]
        e2 = v * v
        for j in range(c):
            s2j = s2[j]
            if s2j != 0.0:
                quot = e2 / s2j
            elif e2 > 0.0:
                quot = math.inf
            elif e2 < 0.0:
                quot = -math.inf
            else:
                quot = math.nan
            lg = math.log(s2j)
            a = acc[j] + quot
            acc[j] = a + lg
            tmp = s2j * betas[j]
            sn = alphas[j] * e2
            sn = sn + omegas[j]
            s2[j] = sn + tmp
    for j in range(c):
        out[j] = acc[j] / n


@kernel(types.Tuple((F8, F8, F8, F8))(A1, F8, F8, F8, A2, A1))
def hist_score_pass(h, alpha, beta, u_cap, scores, center):
    """Score vectors (d/d omega, alpha, beta) over the history; returns the filter state at its end."""
    omega = (1.0 - alpha) - beta
    s2 = 1.0
    dw = 0.0
    da = 0.0
    db = 0.0
    n = h.shape[0]
    c0 = 0.0
    c1 = 0.0
    c2 = 0.0
    for t in range(n):
        ht = h[t]
        e2 = ht * ht
        inv = 1.0 / (s2 if s2 > S2_FLOOR else S2_FLOOR)
        u = e2 * inv
        if u > u_cap:
            u = u_cap
        v = u - 1.0
        z1 = v * da * inv
        z2 = v * db * inv
        scores[t, 0] = v
        scores[t, 1] = z1
        scores[t, 2] = z2
        c0 += v
        c1 += z1
        c2 += z2
        dw = 1.0 + beta * dw
        da = e2 + beta * da
        db = s2 + beta * db
        s2 = omega + alpha * e2 + beta * s2
    center[0] = c0 / n
    center[1] = c1 / n
    center[2] = c2 / n
    return (s2, dw, da, db)


@kernel(types.void(A2, A1, A2))
def score_covariance(scores, center, g):
    """Covariance of the centred score vectors (3 x 3)."""
    n = scores.shape[0]
    c0 = center[0]
    c1 = center[1]
    c2 = center[2]
    g00 = 0.0
    g01 = 0.0
    g02 = 0.0
    g11 = 0.0
    g12 = 0.0
    g22 = 0.0
    for i in range(n):
        x0 = scores[i, 0] - c0
        x1 = scores[i, 1] - c1
        x2 = scores[i, 2] - c2
        g00 += x0 * x0
        g01 += x0 * x1
        g02 += x0 * x2
        g11 += x1 * x1
        g12 += x1 * x2
        g22 += x2 * x2
    g[0, 0] = g00 / n
    g[0, 1] = g01 / n
    g[0, 2] = g02 / n
    g[1, 0] = g01 / n
    g[1, 1] = g11 / n
    g[1, 2] = g12 / n
    g[2, 0] = g02 / n
    g[2, 1] = g12 / n
    g[2, 2] = g22 / n


@kernel(F8(A1, F8, F8, F8))
def _quad_chol(chol, s0, s1, s2):
    """Squared norm of L^-1 s for the packed lower Cholesky factor ``chol``."""
    l11 = chol[0]
    l21 = chol[1]
    l22 = chol[2]
    l31 = chol[3]
    l32 = chol[4]
    l33 = chol[5]
    y0 = s0 / l11
    y1 = (s1 - l21 * y0) / l22
    y2 = (s2 - l31 * y0 - l32 * y1) / l33
    return y0 * y0 + y1 * y1 + y2 * y2


@kernel(types.Tuple((F8, F8, F8, F8))(F8, A1, I1, A1, I8, F8, A1, I8, F8, F8, F8, F8, F8, F8, F8))
def garch_step(xf, fa, ia, chol, has_chol, g00r, center, m_hist, kappa, u_cap, alpha, beta, omega, mu_h, sd_h):
    """One online point: cumulative score test, Horvath-type ratio, score CUSUM and level z."""
    s2 = fa[FA_S2]
    dw = fa[FA_DW]
    da = fa[FA_DA]
    db = fa[FA_DB]

    a = (xf - mu_h) / sd_h
    e2 = a * a
    if not math.isfinite(e2):
        return (0.0, 0.0, fa[FA_G_CUSUM], 0.0)
    inv = 1.0 / (s2 if s2 > S2_FLOOR else S2_FLOOR)
    u = e2 * inv
    if u > u_cap:
        u = u_cap
    v = u - 1.0
    sc0 = v - center[0]
    sc1 = v * da * inv - center[1]
    sc2 = v * db * inv - center[2]

    k = ia[IA_K] + 1
    ia[IA_K] = k
    s_0 = fa[FA_S0] + sc0
    s_1 = fa[FA_S0 + 1] + sc1
    s_2 = fa[FA_S0 + 2] + sc2
    fa[FA_S0] = s_0
    fa[FA_S0 + 1] = s_1
    fa[FA_S0 + 2] = s_2

    if has_chol == 0:
        q = 0.0
        hr = 0.0
        g_cusum = 0.0
        lvl = 0.0
    else:
        big_q = _quad_chol(chol, s_0, s_1, s_2)
        q = big_q / k
        ratio_k = k / (k + m_hist)
        hr = math.sqrt(big_q) / (math.sqrt(m_hist) * (1.0 + k / m_hist) * math.pow(ratio_k, kappa))
        if fa[FA_G_CUSUM] <= 0.0:
            ia[IA_NC] = 0
            fa[FA_C0] = 0.0
            fa[FA_C0 + 1] = 0.0
            fa[FA_C0 + 2] = 0.0
        nc = ia[IA_NC] + 1
        ia[IA_NC] = nc
        c_0 = fa[FA_C0] + sc0
        c_1 = fa[FA_C0 + 1] + sc1
        c_2 = fa[FA_C0 + 2] + sc2
        fa[FA_C0] = c_0
        fa[FA_C0 + 1] = c_1
        fa[FA_C0 + 2] = c_2
        qc = _quad_chol(chol, c_0, c_1, c_2)
        g_cusum = B_MAG * math.sqrt(qc) - 0.5 * nc * B_MAG * B_MAG
        fa[FA_G_CUSUM] = g_cusum
        lvl = s_0 / math.sqrt(k * g00r) if g00r > 0.0 else 0.0

    fa[FA_DW] = 1.0 + beta * dw
    fa[FA_DA] = e2 + beta * da
    fa[FA_DB] = s2 + beta * db
    fa[FA_S2] = omega + alpha * e2 + beta * s2
    return (q, hr, g_cusum, lvl)


@kernel(F8(A1, F8, F8, A1))
def resid_hist(h, alpha, beta, out):
    """GARCH-standardized residuals of the history; returns the next conditional variance."""
    omega = (1.0 - alpha) - beta
    s2 = 1.0
    for t in range(h.shape[0]):
        a = h[t]
        sig = math.sqrt(s2 if s2 > S2_FLOOR else S2_FLOOR)
        out[t] = a / sig
        s2 = omega + alpha * a * a + beta * s2
    return s2


@kernel(F8(F8, A1))
def resid_step(xf, rs):
    """GARCH-standardized residual of one online point; ``rs`` = [s2, mu, sd, alpha, beta, omega]."""
    s2 = rs[0]
    a = (xf - rs[1]) / rs[2]
    sig = math.sqrt(s2 if s2 > S2_FLOOR else S2_FLOOR)
    r = a / sig
    rs[0] = rs[5] + rs[3] * a * a + rs[4] * s2
    return r
