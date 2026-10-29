"""Compiled step of the whitened AR score statistics (cumulative or best suffix)."""

from __future__ import annotations

import math

from numba import types

from .runtime import A1, A2, F8, I1, I8, kernel


@kernel(types.Tuple((F8, I8))(F8, F8, F8, A1, F8, A1, A2, A1, A1, I1, I8, A2, I1, A1, A1))
def bn_step(xf, mu_h, sd_h, coefs, inv_s2, center, ell, phi, cs, ia, sfx, cs_hist, grid, s, w):
    """Whiten the AR(p) score of one point; return (statistic, 1 if the point is not finite)."""
    a = (xf - mu_h) / sd_h
    if not math.isfinite(a):
        return (0.0, 1)

    p = coefs.shape[0]
    eps = a
    for j in range(p):
        eps -= coefs[j] * phi[j]
    d = p + 1
    for j in range(p):
        s[j] = eps * phi[j] * inv_s2 - center[j]
    s[p] = eps * eps * inv_s2 - 1.0 - center[p]
    for i in range(d):
        acc = s[i]
        for k in range(i):
            acc -= ell[i, k] * w[k]
        w[i] = acc / ell[i, i]

    for i in range(d):
        cs[i] += w[i]
    t = ia[0] + 1
    ia[0] = t
    if sfx == 0:
        acc = 0.0
        for i in range(d):
            v = cs[i]
            acc += v * v
        stat = acc / t
    else:
        for i in range(d):
            cs_hist[t, i] = cs[i]
        best = -math.inf
        for gi in range(grid.shape[0]):
            length = grid[gi]
            if length > t:
                break
            prev = t - length
            acc = 0.0
            for i in range(d):
                dv = cs[i] - cs_hist[prev, i]
                acc += dv * dv
            val = acc / length
            if val > best:
                best = val
        stat = best

    for j in range(p - 1, 0, -1):
        phi[j] = phi[j - 1]
    phi[0] = a
    return (stat, 0)
