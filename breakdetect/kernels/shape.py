"""Compiled steps of the distribution-shape detectors (quantile mixture, tail exceedance, Hermite).

Every step updates the main lane (lane 0, the online point) and the null lanes (rotated
copies of the history) and returns the main statistic standardized against the null lanes.
Sums reproduce numpy's pairwise summation so that results match the numpy reference exactly.
"""

from __future__ import annotations

import math

from numba import types

from .runtime import A1, A2, A3, F8, I1, I2, I3, I8, kernel

LOG2 = 0.693147180559945309417232121458176568
SD_GUARD = 1e-8
POT_CAP = 128

_RET = types.Tuple((F8, I8))


@kernel(F8(A1, I8, I8))
def _pairwise_sum(a, lo, n):
    """numpy's pairwise summation of ``a[lo:lo + n]`` (same blocking, same order)."""
    if n < 8:
        res = 0.0
        for i in range(n):
            res += a[lo + i]
        return res
    elif n <= 128:
        r0 = a[lo]
        r1 = a[lo + 1]
        r2 = a[lo + 2]
        r3 = a[lo + 3]
        r4 = a[lo + 4]
        r5 = a[lo + 5]
        r6 = a[lo + 6]
        r7 = a[lo + 7]
        i = 8
        while i < n - (n % 8):
            r0 += a[lo + i]
            r1 += a[lo + i + 1]
            r2 += a[lo + i + 2]
            r3 += a[lo + i + 3]
            r4 += a[lo + i + 4]
            r5 += a[lo + i + 5]
            r6 += a[lo + i + 6]
            r7 += a[lo + i + 7]
            i += 8
        res = ((r0 + r1) + (r2 + r3)) + ((r4 + r5) + (r6 + r7))
        while i < n:
            res += a[lo + i]
            i += 1
        return res
    else:
        n2 = n // 2
        n2 -= n2 % 8
        return _pairwise_sum(a, lo, n2) + _pairwise_sum(a, lo + n2, n - n2)


@kernel(F8(A1, I8, I8))
def _np_sum(a, lo, n):
    """``np.sum`` of a slice."""
    return 0.0 + _pairwise_sum(a, lo, n)


@kernel(F8(A1, I8, I8))
def _np_max(a, lo, n):
    """``np.max`` of a slice (NaN propagates)."""
    acc = a[lo]
    for i in range(1, n):
        b = a[lo + i]
        if not (acc > b or acc != acc):
            acc = b
    return acc


@kernel(F8(F8))
def _logaddexp0(y):
    """``np.logaddexp(0.0, y)``."""
    x = 0.0
    if x == y:
        return x + LOG2
    tmp = x - y
    if tmp > 0.0:
        return x + math.log1p(math.exp(-tmp))
    elif tmp <= 0.0:
        return y + math.log1p(math.exp(tmp))
    return tmp


@kernel(F8(F8, F8, F8))
def _np_clip(x, lo, hi):
    """``np.clip`` of a scalar."""
    if x < lo:
        x = lo
    if x > hi:
        x = hi
    return x


@kernel(F8(A1, I8, I8, A1))
def _logsumexp(a, lo, n, scratch):
    """Max-shifted log-sum-exp of a slice."""
    m = _np_max(a, lo, n)
    for i in range(n):
        scratch[i] = math.exp(a[lo + i] - m)
    return m + math.log(_np_sum(scratch, 0, n))


@kernel(I8(F8, F8, F8, I8, A2, I1, A1))
def _fill_lanes(xf, mu_h, sd_h, n_null, vals_null, ctr, vals):
    """Standardize the online point into lane 0 and copy the current null row; 1 if NaN."""
    a = (xf - mu_h) / sd_h
    if a != a:
        return 1
    row = ctr[0] % n_null
    vals[0] = a
    for b in range(1, vals.shape[0]):
        vals[b] = vals_null[row, b - 1]
    return 0


@kernel(F8(A1, A1, I1))
def _standardize(stats, work, ctr):
    """z of the main lane against the mean and sd of the null lanes."""
    n = stats.shape[0] - 1
    mean_b = _np_sum(stats, 1, n) / n
    for b in range(n):
        d = stats[1 + b] - mean_b
        work[b] = d * d
    var_b = _np_sum(work, 0, n) / (n - 1)
    sd_b = math.sqrt(var_b)
    z = (stats[0] - mean_b) / sd_b if sd_b > SD_GUARD else 0.0
    ctr[0] = ctr[0] + 1
    return z


@kernel(_RET(F8, F8, F8, I8, A2, I1, A1, A1, A1, A3, A1, A2, A2, F8, A1))
def qms_step(xf, mu_h, sd_h, n_null, vals_null, ctr, vals, stats, work, logr, q, l1, l0, log_g, scratch):
    """Bernoulli rate-change Shiryaev-Roberts mixture over quantile thresholds, summed over thresholds."""
    if _fill_lanes(xf, mu_h, sd_h, n_null, vals_null, ctr, vals):
        return (math.nan, 1)
    n_lanes = logr.shape[0]
    n_k = logr.shape[1]
    n_g = logr.shape[2]
    for lane in range(n_lanes):
        v = vals[lane]
        for k in range(n_k):
            flag = v <= q[k]
            for g in range(n_g):
                inc = l1[k, g] if flag else l0[k, g]
                r = _logaddexp0(logr[lane, k, g])
                r = r + inc
                logr[lane, k, g] = r
    for lane in range(n_lanes):
        for k in range(n_k):
            for g in range(n_g):
                scratch[g] = logr[lane, k, g]
            work[k] = _logsumexp(scratch, 0, n_g, scratch) - log_g
        stats[lane] = _np_sum(work, 0, n_k)
    return (_standardize(stats, work, ctr), 0)


@kernel(F8(I3, I3, I2, I2, I8, I8, I8, I8, F8))
def _pot_stream_step(taus, staus, cnt, s_cnt, lane, stream, x, n, th0):
    """Bernoulli FOCuS for a rate increase: prune candidates, return the best log-likelihood ratio."""
    s = s_cnt[lane, stream] + x
    s_cnt[lane, stream] = s
    c = cnt[lane, stream]
    while c > 0:
        seg = n - taus[lane, stream, c - 1]
        rate_last = (s - staus[lane, stream, c - 1]) / seg
        if c > 1:
            prev = (s - staus[lane, stream, c - 2]) / (n - taus[lane, stream, c - 2])
            bound = prev if prev > th0 else th0
        else:
            bound = th0
        if rate_last <= bound:
            c -= 1
        else:
            break
    taus[lane, stream, c] = n
    staus[lane, stream, c] = s
    c += 1
    if c > POT_CAP:
        for j in range(c - 1):
            taus[lane, stream, j] = taus[lane, stream, j + 1]
            staus[lane, stream, j] = staus[lane, stream, j + 1]
        c -= 1
    cnt[lane, stream] = c
    best = 0.0
    for j in range(c - 1):
        a = s - staus[lane, stream, j]
        if a <= 0:
            continue
        m_seg = n - taus[lane, stream, j]
        that = a / m_seg
        if that <= th0:
            continue
        b = m_seg - a
        val = a * math.log(that / th0)
        if b > 0:
            val += b * math.log((1.0 - that) / (1.0 - th0))
        if val > best:
            best = val
    return best


@kernel(_RET(F8, F8, F8, I8, A2, I1, A1, A1, A1, I3, I3, I2, I2, I1, A1, A1, I1))
def pot_step(xf, mu_h, sd_h, n_null, vals_null, ctr, vals, stats, work, taus, staus, cnt, s_cnt, nstep, q, th0, is_hi):
    """Sum of the tail-exceedance FOCuS statistics of the four quantile thresholds."""
    if _fill_lanes(xf, mu_h, sd_h, n_null, vals_null, ctr, vals):
        return (math.nan, 1)
    nstep[0] = nstep[0] + 1
    n = nstep[0]
    n_lanes = vals.shape[0]
    m = q.shape[0]
    for lane in range(n_lanes):
        v = vals[lane]
        total = 0.0
        for stream in range(m):
            x = (1 if v > q[stream] else 0) if is_hi[stream] else (1 if v <= q[stream] else 0)
            total += _pot_stream_step(taus, staus, cnt, s_cnt, lane, stream, x, n, th0[stream])
        stats[lane] = total
    return (_standardize(stats, work, ctr), 0)


@kernel(_RET(F8, F8, F8, I8, A2, I1, A1, A1, A1, A2, F8, F8, F8, F8, F8, F8, F8))
def hermite_step(
    xf, mu_h, sd_h, n_null, vals_null, ctr, vals, stats, work, c, cap, mu3, mu4, inv_sd3, inv_sd4, k, ycap
):
    """Two-sided Page CUSUMs on the standardized Hermite polynomials h3 and h4."""
    if _fill_lanes(xf, mu_h, sd_h, n_null, vals_null, ctr, vals):
        return (math.nan, 1)
    n_lanes = c.shape[0]
    neg_cap = -cap
    neg_ycap = -ycap
    for lane in range(n_lanes):
        a = _np_clip(vals[lane], neg_cap, cap)
        a2 = a * a
        h3 = _np_clip((a2 * a - 3.0 * a - mu3) * inv_sd3, neg_ycap, ycap)
        h4 = _np_clip((a2 * a2 - 6.0 * a2 + 3.0 - mu4) * inv_sd4, neg_ycap, ycap)
        v0 = c[lane, 0] + h3 - k
        c[lane, 0] = v0 if (v0 > 0.0 or v0 != v0) else 0.0
        v1 = c[lane, 1] - h3 - k
        c[lane, 1] = v1 if (v1 > 0.0 or v1 != v1) else 0.0
        v2 = c[lane, 2] + h4 - k
        c[lane, 2] = v2 if (v2 > 0.0 or v2 != v2) else 0.0
        v3 = c[lane, 3] - h4 - k
        c[lane, 3] = v3 if (v3 > 0.0 or v3 != v3) else 0.0
        for j in range(4):
            work[j] = c[lane, j]
        stats[lane] = _np_max(work, 0, 4)
    return (_standardize(stats, work, ctr), 0)
