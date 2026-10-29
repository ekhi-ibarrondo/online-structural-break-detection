"""Compiled step of the mean and dependence detectors.

``kind`` selects one detector: two-sided FOCuS for a mean change, a Shiryaev-Roberts mixture
over mean shifts, whitened AR(p) local-score FOCuS, or a sign-product CUSUM on AR residuals.
Stream 0 is the online point; streams 1..B are the null lanes.
"""

from __future__ import annotations

import math

from .runtime import A1, A2, A3, F8, I1, I2, I3, I8, kernel

SD_GUARD = 1e-8

KIND_FOCUS_MEAN = 0
KIND_MIX_MEAN = 1
KIND_AR = 2
KIND_SIGN = 3


@kernel(F8(I3, A3, A3, I2, A1, I8, F8))
def _fmean_update(tau, sa, bnd, hdr, stot, q, x):
    """FOCuS for a mean change against mu = 0 (up and down branches)."""
    n = hdr[q, 0] + 1
    hdr[q, 0] = n
    s = stot[q] + x
    stot[q] = s
    cap = tau.shape[2]

    best = 0.0
    up_len = hdr[q, 1]
    for i in range(up_len):
        d = s - sa[q, 0, i]
        if d > 0.0:
            c = d * d / (2.0 * (n - tau[q, 0, i]))
            if c > best:
                best = c
    dn_len = hdr[q, 2]
    for i in range(dn_len):
        d = s - sa[q, 1, i]
        if d < 0.0:
            c = d * d / (2.0 * (n - tau[q, 1, i]))
            if c > best:
                best = c

    k = up_len
    m = 0.0
    while k > 0:
        m = 2.0 * (s - sa[q, 0, k - 1]) / (n - tau[q, 0, k - 1])
        if m <= bnd[q, 0, k - 1]:
            k -= 1
        else:
            break
    if k >= cap:
        for j in range(k - 1):
            tau[q, 0, j] = tau[q, 0, j + 1]
            sa[q, 0, j] = sa[q, 0, j + 1]
            bnd[q, 0, j] = bnd[q, 0, j + 1]
        bnd[q, 0, 0] = 0.0
        k -= 1
    tau[q, 0, k] = n
    sa[q, 0, k] = s
    bnd[q, 0, k] = m if (k > 0 and m > 0.0) else 0.0
    hdr[q, 1] = k + 1

    k = dn_len
    m = 0.0
    while k > 0:
        m = 2.0 * (s - sa[q, 1, k - 1]) / (n - tau[q, 1, k - 1])
        if m >= bnd[q, 1, k - 1]:
            k -= 1
        else:
            break
    if k >= cap:
        for j in range(k - 1):
            tau[q, 1, j] = tau[q, 1, j + 1]
            sa[q, 1, j] = sa[q, 1, j + 1]
            bnd[q, 1, j] = bnd[q, 1, j + 1]
        bnd[q, 1, 0] = 0.0
        k -= 1
    tau[q, 1, k] = n
    sa[q, 1, k] = s
    bnd[q, 1, k] = m if (k > 0 and m < 0.0) else 0.0
    hdr[q, 2] = k + 1
    return best


@kernel(F8(A2, I8, A1, A1, A1, F8, F8))
def _mixmean_update(logr, s, delta, c2, logw, add, x):
    """Log Shiryaev-Roberts statistic mixed over a grid of mean shifts."""
    m = delta.shape[0]
    vmax = -math.inf
    for j in range(m):
        u = logr[s, j]
        sp = u + math.log1p(math.exp(-u)) if u > 0.0 else math.log1p(math.exp(u))
        v = sp + delta[j] * x - c2[j] + add
        logr[s, j] = v
        vw = v + logw[j]
        if vw > vmax:
            vmax = vw
    acc = 0.0
    for j in range(m):
        acc += math.exp(logr[s, j] + logw[j] - vmax)
    return vmax + math.log(acc)


@kernel(F8(A2, I8, A2, A1, F8, I3, A3, A3, I2, A1, I8, F8))
def _ar_update(ph, s, lt, coefs, inv_sigma, tau, sa, bnd, hdr, stot, ubase, x):
    """Sum of per-component FOCuS statistics of the whitened AR(p) score."""
    p = coefs.shape[0]
    eps = x
    for j in range(p):
        eps -= coefs[j] * ph[s, j]
    u = eps * inv_sigma
    scale = inv_sigma * u
    total = 0.0
    for i in range(p):
        acc = 0.0
        for j in range(p - i):
            acc += lt[i, j] * ph[s, i + j]
        total += _fmean_update(tau, sa, bnd, hdr, stot, ubase + i, acc * scale)
    for j in range(p - 1, 0, -1):
        ph[s, j] = ph[s, j - 1]
    ph[s, 0] = x
    return total


@kernel(F8(A2, I8, A1, A2, A2, A2, I1, A1, A1, F8, F8))
def _sign_update(ph, s, coefs, ring, cp, cm, lags, phats, sdinv, k, x):
    """Two-sided CUSUM of 1(eps_t * eps_{t-lag} > 0) against its historical rate."""
    p = coefs.shape[0]
    eps = x
    for j in range(p):
        eps -= coefs[j] * ph[s, j]
    total = 0.0
    n_lags = lags.shape[0]
    for idx in range(n_lags):
        prev = ring[s, lags[idx] - 1]
        b = 1.0 if eps * prev > 0.0 else 0.0
        y = (b - phats[idx]) * sdinv[idx]
        vp = cp[s, idx] + y - k
        cp[s, idx] = vp if vp > 0.0 else 0.0
        vm = cm[s, idx] - y - k
        cm[s, idx] = vm if vm > 0.0 else 0.0
        hi = cp[s, idx] if cp[s, idx] > cm[s, idx] else cm[s, idx]
        total += hi
    r_len = ring.shape[1]
    for j in range(r_len - 1, 0, -1):
        ring[s, j] = ring[s, j - 1]
    ring[s, 0] = eps
    for j in range(p - 1, 0, -1):
        ph[s, j] = ph[s, j - 1]
    ph[s, 0] = x
    return total


@kernel(
    F8(
        F8,
        F8,
        F8,
        F8,
        I8,
        I1,
        A1,
        I1,
        A1,
        A1,
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
        A2,
        A1,
        A1,
        A1,
        F8,
        A2,
        A2,
        A1,
        F8,
        A2,
        A2,
        A2,
        I1,
        A1,
        A1,
        F8,
    )
)
def meandep_step(
    xf, mu_h, sd_h, cap, kind,
    ctr, null_src, starts, offs, stats,
    fm_tau, fm_sa, fm_bnd, fm_hdr, fm_stot,
    fn_tau, fn_sa, fn_bnd, fn_hdr, fn_stot,
    mx_logr, mx_delta, mx_c2, mx_logw, mx_add,
    ar_ph, ar_lt, ar_coefs, ar_inv_sigma,
    sg_ring, sg_cp, sg_cm, sg_lags, sg_phats, sg_sdinv, sg_k,
):  # fmt: skip
    """Advance the main stream and the null lanes by one point; return the main z."""
    a = (xf - mu_h) / sd_h
    if a > cap:
        a = cap
    elif a < -cap:
        a = -cap

    i = ctr[0]
    n_null = null_src.shape[0]
    n_sub = starts.shape[0]
    if kind == KIND_FOCUS_MEAN:
        for b in range(n_sub):
            v = null_src[(starts[b] + i) % n_null] + offs[b]
            if v > cap:
                v = cap
            elif v < -cap:
                v = -cap
            stats[b] = _fmean_update(fn_tau, fn_sa, fn_bnd, fn_hdr, fn_stot, b, v)
        stat = _fmean_update(fm_tau, fm_sa, fm_bnd, fm_hdr, fm_stot, 0, a)
    elif kind == KIND_MIX_MEAN:
        for b in range(n_sub):
            v = null_src[(starts[b] + i) % n_null] + offs[b]
            if v > cap:
                v = cap
            elif v < -cap:
                v = -cap
            stats[b] = _mixmean_update(mx_logr, 1 + b, mx_delta, mx_c2, mx_logw, mx_add, v)
        stat = _mixmean_update(mx_logr, 0, mx_delta, mx_c2, mx_logw, mx_add, a)
    elif kind == KIND_AR:
        n_units = ar_coefs.shape[0]
        for b in range(n_sub):
            v = null_src[(starts[b] + i) % n_null] + offs[b]
            if v > cap:
                v = cap
            elif v < -cap:
                v = -cap
            stats[b] = _ar_update(
                ar_ph, 1 + b, ar_lt, ar_coefs, ar_inv_sigma, fn_tau, fn_sa, fn_bnd, fn_hdr, fn_stot, b * n_units, v
            )
        stat = _ar_update(ar_ph, 0, ar_lt, ar_coefs, ar_inv_sigma, fm_tau, fm_sa, fm_bnd, fm_hdr, fm_stot, 0, a)
    else:
        for b in range(n_sub):
            v = null_src[(starts[b] + i) % n_null] + offs[b]
            if v > cap:
                v = cap
            elif v < -cap:
                v = -cap
            stats[b] = _sign_update(ar_ph, 1 + b, ar_coefs, sg_ring, sg_cp, sg_cm, sg_lags, sg_phats, sg_sdinv, sg_k, v)
        stat = _sign_update(ar_ph, 0, ar_coefs, sg_ring, sg_cp, sg_cm, sg_lags, sg_phats, sg_sdinv, sg_k, a)

    mean_b = 0.0
    for b in range(n_sub):
        mean_b += stats[b]
    mean_b /= n_sub
    var_b = 0.0
    for b in range(n_sub):
        dv = stats[b] - mean_b
        var_b += dv * dv
    sd_b = math.sqrt(var_b / (n_sub - 1))
    z = (stat - mean_b) / sd_b if sd_b > SD_GUARD else 0.0
    ctr[0] = i + 1
    return z
