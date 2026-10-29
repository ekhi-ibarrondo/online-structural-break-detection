"""Compiled step of the variance detectors: FOCuS, Sanso kappa-2 and a Shiryaev-Roberts mixture.

Stream 0 is the online point; streams 1..B are the null sub-streams. Each detector's main
statistic is z-scored against its null statistics and the z values are averaged.
"""

from __future__ import annotations

import math

from .runtime import A1, A2, A3, F8, I1, I2, I3, I8, kernel

R_GUARD = 1e-12
SD_GUARD = 1e-8

KIND_FOCUS = 0
KIND_SANSO = 1
KIND_MIX = 2


@kernel(F8(A2, I8, I8))
def _py_sum(a, d, n):
    """Python 3.12 ``sum()`` of ``a[d, :n]``: Neumaier-compensated, same rounding as the builtin."""
    f = 0.0 + a[d, 0]
    c = 0.0
    for b in range(1, n):
        x = a[d, b]
        t = f + x
        if abs(f) >= abs(x):
            c += (f - t) + x
        else:
            c += (x - t) + f
        f = t
    if c != 0.0 and math.isfinite(c):
        f += c
    return f


@kernel(F8(I3, A3, A3, I2, A1, I8, F8))
def _focus_update(tau, s2a, bnd, hdr, s2tot, q, x):
    """FOCuS for a variance change against sigma^2 = 1 (increase and decrease branches)."""
    n = hdr[q, 0] + 1
    hdr[q, 0] = n
    s2 = s2tot[q] + x * x
    s2tot[q] = s2
    cap = tau.shape[2]

    best = 0.0
    inc_len = hdr[q, 1]
    for i in range(inc_len):
        seg = n - tau[q, 0, i]
        r = (s2 - s2a[q, 0, i]) / seg
        if r > 1.0:
            c = 0.5 * seg * (r - 1.0 - math.log(r))
            if c > best:
                best = c
    dec_len = hdr[q, 2]
    for i in range(dec_len):
        seg = n - tau[q, 1, i]
        r = (s2 - s2a[q, 1, i]) / seg
        if r < 1.0:
            if r < R_GUARD:
                r = R_GUARD
            c = 0.5 * seg * (r - 1.0 - math.log(r))
            if c > best:
                best = c

    k = inc_len
    phi = 1.0
    while k > 0:
        phi = (s2 - s2a[q, 0, k - 1]) / (n - tau[q, 0, k - 1])
        if phi <= bnd[q, 0, k - 1]:
            k -= 1
        else:
            break
    if k >= cap:
        for j in range(k - 1):
            tau[q, 0, j] = tau[q, 0, j + 1]
            s2a[q, 0, j] = s2a[q, 0, j + 1]
            bnd[q, 0, j] = bnd[q, 0, j + 1]
        bnd[q, 0, 0] = 1.0
        k -= 1
    tau[q, 0, k] = n
    s2a[q, 0, k] = s2
    bnd[q, 0, k] = phi if (k > 0 and phi > 1.0) else 1.0
    hdr[q, 1] = k + 1

    k = dec_len
    phi = 1.0
    while k > 0:
        phi = (s2 - s2a[q, 1, k - 1]) / (n - tau[q, 1, k - 1])
        if phi >= bnd[q, 1, k - 1]:
            k -= 1
        else:
            break
    if k >= cap:
        for j in range(k - 1):
            tau[q, 1, j] = tau[q, 1, j + 1]
            s2a[q, 1, j] = s2a[q, 1, j + 1]
            bnd[q, 1, j] = bnd[q, 1, j + 1]
        bnd[q, 1, 0] = 1.0
        k -= 1
    tau[q, 1, k] = n
    s2a[q, 1, k] = s2
    bnd[q, 1, k] = phi if (k > 0 and phi < 1.0) else 1.0
    hdr[q, 2] = k + 1
    return best


@kernel(F8(I1, A2, I8, F8, F8, F8))
def _sanso_update(sn_t, sn_f, s, omega4, inv_n, x):
    """Sanso kappa-2 CUSUM of squares, normalized by the long-run variance omega4."""
    u = sn_f[s, 0] + x * x - 1.0
    sn_f[s, 0] = u
    t = sn_t[s] + 1
    sn_t[s] = t
    if u < sn_f[s, 1]:
        sn_f[s, 1] = u
    if u > sn_f[s, 2]:
        sn_f[s, 2] = u
    spread = u - sn_f[s, 1]
    alt = sn_f[s, 2] - u
    if alt > spread:
        spread = alt
    return spread / math.sqrt(omega4 * t * (1.0 + t * inv_n))


@kernel(F8(A2, I8, A1, A1, A1, F8, F8))
def _mix_update(logr, s, c1, c2, logw, add, x):
    """Log Shiryaev-Roberts statistic mixed over a grid of variance ratios."""
    x2 = x * x
    m = c1.shape[0]
    vmax = -math.inf
    for j in range(m):
        u = logr[s, j]
        sp = u + math.log1p(math.exp(-u)) if u > 0.0 else math.log1p(math.exp(u))
        v = sp + c1[j] - x2 * c2[j] + add
        logr[s, j] = v
        vw = v + logw[j]
        if vw > vmax:
            vmax = vw
    acc = 0.0
    for j in range(m):
        acc += math.exp(logr[s, j] + logw[j] - vmax)
    return vmax + math.log(acc)


@kernel(
    F8(
        F8,
        F8,
        F8,
        F8,
        I1,
        A1,
        I1,
        A1,
        A1,
        I1,
        A2,
        I3,
        A3,
        A3,
        I2,
        A1,
        I3,
        A3,
        A3,
        I2,
        A1,
        I1,
        A2,
        F8,
        F8,
        A2,
        A1,
        A1,
        A1,
        F8,
    )
)
def variance_step(
    xf, mu_h, sd_h, cap,
    ctr, hist, starts, factors, buf, kinds, stats,
    fm_tau, fm_s2, fm_bnd, fm_hdr, fm_s2tot,
    fn_tau, fn_s2, fn_bnd, fn_hdr, fn_s2tot,
    sn_t, sn_f, omega4, inv_n_main,
    mx_logr, mx_c1, mx_c2, mx_logw, mx_add,
):  # fmt: skip
    """Advance every stream by one point; return the mean z of the selected detectors."""
    a = (xf - mu_h) / sd_h
    if a > cap:
        a = cap
    elif a < -cap:
        a = -cap

    i = ctr[0]
    n_hist = hist.shape[0]
    n_sub = starts.shape[0]
    n_det = kinds.shape[0]
    for b in range(n_sub):
        v = factors[b] * hist[(starts[b] + i) % n_hist]
        if v > cap:
            v = cap
        elif v < -cap:
            v = -cap
        buf[b] = v
    for d in range(n_det):
        kind = kinds[d]
        if kind == KIND_FOCUS:
            for b in range(n_sub):
                stats[d, b] = _focus_update(fn_tau, fn_s2, fn_bnd, fn_hdr, fn_s2tot, b, buf[b])
        elif kind == KIND_SANSO:
            for b in range(n_sub):
                stats[d, b] = _sanso_update(sn_t, sn_f, 1 + b, omega4, 0.0, buf[b])
        else:
            for b in range(n_sub):
                stats[d, b] = _mix_update(mx_logr, 1 + b, mx_c1, mx_c2, mx_logw, mx_add, buf[b])

    z_sum = 0.0
    for d in range(n_det):
        kind = kinds[d]
        if kind == KIND_FOCUS:
            stat = _focus_update(fm_tau, fm_s2, fm_bnd, fm_hdr, fm_s2tot, 0, a)
        elif kind == KIND_SANSO:
            stat = _sanso_update(sn_t, sn_f, 0, omega4, inv_n_main, a)
        else:
            stat = _mix_update(mx_logr, 0, mx_c1, mx_c2, mx_logw, mx_add, a)
        mean_b = _py_sum(stats, d, n_sub) / n_sub
        var_b = 0.0
        for b in range(n_sub):
            dv = stats[d, b] - mean_b
            var_b += dv * dv
        sd_b = math.sqrt(var_b / (n_sub - 1))
        if sd_b > SD_GUARD:
            z_sum += (stat - mean_b) / sd_b
    ctr[0] = i + 1
    return z_sum / n_det
