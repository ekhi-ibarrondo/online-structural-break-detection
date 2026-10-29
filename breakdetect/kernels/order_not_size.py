"""Compiled kernels of the order_not_size member: portable transcendentals, AR and GARCH(1,1) fits.

``log``, ``exp`` and ``erfc`` are literal ports of fdlibm 5.3 written with basic IEEE
arithmetic and bit operations only, so the results do not depend on the platform libm.
Reductions are fixed-order compensated loops and the linear solve is a fixed-order
Gaussian elimination (no LAPACK). ``max``/``min`` are written as explicit comparisons to
pin their NaN and signed-zero behaviour.

fdlibm notice (e_log.c, s_erf.c: Copyright (C) 1993; e_exp.c: Copyright (C) 2004):

  ====================================================
  Copyright (C) 1993, 2004 by Sun Microsystems, Inc. All rights reserved.

  Developed at SunSoft, a Sun Microsystems, Inc. business.
  Permission to use, copy, modify, and distribute this
  software is freely granted, provided that this notice
  is preserved.
  ====================================================
"""

from __future__ import annotations

import math

import numpy as np

from .runtime import kernel

_A1 = "float64[::1]"

GARCH_GRID_A = (0.0, 0.03, 0.08, 0.15, 0.3)
GARCH_GRID_B = (0.0, 0.6, 0.8, 0.9, 0.95, 0.98)
GARCH_NM_EVALS = 40
_AMAX = 0.8
_BMAX = 0.998
_ABMAX = 0.999
_A = (
    -3.969683028665376e01,
    2.209460984245205e02,
    -2.759285104469687e02,
    1.383577518672690e02,
    -3.066479806614716e01,
    2.506628277459239e00,
)
_B = (-5.447609879822406e01, 1.615858368580409e02, -1.556989798598866e02, 6.680131188771972e01, -1.328068155288572e01)
_C = (
    -7.784894002430293e-03,
    -3.223964580411365e-01,
    -2.400758277161838e00,
    -2.549732539343734e00,
    4.374664141464968e00,
    2.938163982698783e00,
)
_D = (7.784695709041462e-03, 3.224671290700398e-01, 2.445134137142996e00, 3.754408661907416e00)
_P_LOW = 0.02425
SQRT2 = float.fromhex("0x1.6a09e667f3bcdp+0")
SQRT2PI = float.fromhex("0x1.40d931ff62705p+1")

_INF = float("inf")
_NAN = float("nan")

_LN2_HI = 6.93147180369123816490e-01
_LN2_LO = 1.90821492927058770002e-10
_LG1 = 6.666666666666735130e-01
_LG2 = 3.999999999940941908e-01
_LG3 = 2.857142874366239149e-01
_LG4 = 2.222219843214978396e-01
_LG5 = 1.818357216161805012e-01
_LG6 = 1.531383769920937332e-01
_LG7 = 1.479819860511658591e-01
_TWO54 = 1.80143985094819840000e16


@kernel("int64(float64)")
def _bits(x):
    """IEEE-754 bit pattern of a float64."""
    return int(np.float64(x).view(np.int64))


@kernel("float64(int64)")
def _from_bits(b):
    """float64 with the given bit pattern."""
    return float(np.int64(b).view(np.float64))


@kernel("int64(float64)")
def _hi(x):
    """Signed high 32-bit word (fdlibm __HI)."""
    return _bits(x) >> 32


@kernel("int64(float64)")
def _lo(x):
    """Unsigned low 32-bit word (fdlibm __LO)."""
    return _bits(x) & 0xFFFFFFFF


@kernel("float64(float64, int64)")
def _set_hi(x, v):
    """Replace the high word of x."""
    return _from_bits((v << 32) | (_bits(x) & 0xFFFFFFFF))


@kernel("float64(float64)")
def _set_lo0(x):
    """Clear the low word of x."""
    return _from_bits((_bits(x) >> 32) << 32)


@kernel("float64(float64)")
def log_p(x):
    """Natural logarithm, port of fdlibm e_log.c."""
    hx = _hi(x)
    lx = _lo(x)
    k = 0
    if hx < 0x00100000:
        if ((hx & 0x7FFFFFFF) | lx) == 0:
            return -_INF
        if hx < 0:
            return _NAN
        k -= 54
        x *= _TWO54
        hx = _hi(x)
    if hx >= 0x7FF00000:
        return x + x
    k += (hx >> 20) - 1023
    hx &= 0x000FFFFF
    i = (hx + 0x95F64) & 0x100000
    x = _set_hi(x, hx | (i ^ 0x3FF00000))
    k += i >> 20
    f = x - 1.0
    if (0x000FFFFF & (2 + hx)) < 3:
        if f == 0.0:
            if k == 0:
                return 0.0
            dk = float(k)
            return dk * _LN2_HI + dk * _LN2_LO
        R = f * f * (0.5 - 0.33333333333333333 * f)
        if k == 0:
            return f - R
        dk = float(k)
        return dk * _LN2_HI - ((R - dk * _LN2_LO) - f)
    s = f / (2.0 + f)
    dk = float(k)
    z = s * s
    i = hx - 0x6147A
    w = z * z
    j = 0x6B851 - hx
    t1 = w * (_LG2 + w * (_LG4 + w * _LG6))
    t2 = z * (_LG1 + w * (_LG3 + w * (_LG5 + w * _LG7)))
    i = i | j
    R = t2 + t1
    if i > 0:
        hfsq = 0.5 * f * f
        if k == 0:
            return f - (hfsq - s * (hfsq + R))
        return dk * _LN2_HI - ((hfsq - (s * (hfsq + R) + dk * _LN2_LO)) - f)
    if k == 0:
        return f - s * (f - R)
    return dk * _LN2_HI - ((s * (f - R) - dk * _LN2_LO) - f)


_O_THRESHOLD = 7.09782712893383973096e02
_U_THRESHOLD = -7.45133219101941108420e02
_INVLN2 = 1.44269504088896338700e00
_P1 = 1.66666666666666019037e-01
_P2 = -2.77777777770155933842e-03
_P3 = 6.61375632143793436117e-05
_P4 = -1.65339022054652515390e-06
_P5 = 4.13813679705723846039e-08
_HUGE = 1.0e300
_TWOM1000 = 9.33263618503218878990e-302
_HALF = (0.5, -0.5)
_LN2HI = (6.93147180369123816490e-01, -6.93147180369123816490e-01)
_LN2LO = (1.90821492927058770002e-10, -1.90821492927058770002e-10)


@kernel("float64(float64)")
def exp_p(x):
    """Exponential, port of fdlibm e_exp.c."""
    hx = _hi(x)
    xsb = (hx >> 31) & 1
    hx &= 0x7FFFFFFF
    if hx >= 0x40862E42:
        if hx >= 0x7FF00000:
            if ((hx & 0xFFFFF) | _lo(x)) != 0:
                return x + x
            if xsb == 0:
                return x
            return 0.0
        if x > _O_THRESHOLD:
            return _HUGE * _HUGE
        if x < _U_THRESHOLD:
            return _TWOM1000 * _TWOM1000
    k = 0
    hi = 0.0
    lo = 0.0
    if hx > 0x3FD62E42:
        if hx < 0x3FF0A2B2:
            hi = x - _LN2HI[xsb]
            lo = _LN2LO[xsb]
            k = 1 - xsb - xsb
        else:
            k = int(_INVLN2 * x + _HALF[xsb])
            t = float(k)
            hi = x - t * _LN2HI[0]
            lo = t * _LN2LO[0]
        x = hi - lo
    elif hx < 0x3E300000:
        if _HUGE + x > 1.0:
            return 1.0 + x
    else:
        k = 0
    t = x * x
    c = x - t * (_P1 + t * (_P2 + t * (_P3 + t * (_P4 + t * _P5))))
    if k == 0:
        return 1.0 - ((x * c) / (c - 2.0) - x)
    y = 1.0 - ((lo - (x * c) / (2.0 - c)) - hi)
    if k >= -1021:
        return _set_hi(y, _hi(y) + (k << 20))
    y = _set_hi(y, _hi(y) + ((k + 1000) << 20))
    return y * _TWOM1000


_TINY = 1e-300
_ERX = 8.45062911510467529297e-01
_PP0 = 1.28379167095512558561e-01
_PP1 = -3.25042107247001499370e-01
_PP2 = -2.84817495755985104766e-02
_PP3 = -5.77027029648944159157e-03
_PP4 = -2.37630166566501626084e-05
_QQ1 = 3.97917223959155352819e-01
_QQ2 = 6.50222499887672944485e-02
_QQ3 = 5.08130628187576562776e-03
_QQ4 = 1.32494738004321644526e-04
_QQ5 = -3.96022827877536812320e-06
_PA0 = -2.36211856075265944077e-03
_PA1 = 4.14856118683748331666e-01
_PA2 = -3.72207876035701323847e-01
_PA3 = 3.18346619901161753674e-01
_PA4 = -1.10894694282396677476e-01
_PA5 = 3.54783043256182359371e-02
_PA6 = -2.16637559486879084300e-03
_QA1 = 1.06420880400844228286e-01
_QA2 = 5.40397917702171048937e-01
_QA3 = 7.18286544141962662868e-02
_QA4 = 1.26171219808761642112e-01
_QA5 = 1.36370839120290507362e-02
_QA6 = 1.19844998467991074170e-02
_RA0 = -9.86494403484714822705e-03
_RA1 = -6.93858572707181764372e-01
_RA2 = -1.05586262253232909814e01
_RA3 = -6.23753324503260060396e01
_RA4 = -1.62396669462573470355e02
_RA5 = -1.84605092906711035994e02
_RA6 = -8.12874355063065934246e01
_RA7 = -9.81432934416914548592e00
_SA1 = 1.96512716674392571292e01
_SA2 = 1.37657754143519042600e02
_SA3 = 4.34565877475229228821e02
_SA4 = 6.45387271733267880336e02
_SA5 = 4.29008140027567833386e02
_SA6 = 1.08635005541779435134e02
_SA7 = 6.57024977031928170135e00
_SA8 = -6.04244152148580987438e-02
_RB0 = -9.86494292470009928597e-03
_RB1 = -7.99283237680523006574e-01
_RB2 = -1.77579549177547519889e01
_RB3 = -1.60636384855821916062e02
_RB4 = -6.37566443368389627722e02
_RB5 = -1.02509513161107724954e03
_RB6 = -4.83519191608651397019e02
_SB1 = 3.03380607434824582924e01
_SB2 = 3.25792512996573918826e02
_SB3 = 1.53672958608443695994e03
_SB4 = 3.19985821950859553908e03
_SB5 = 2.55305040643316442583e03
_SB6 = 4.74528541206955367215e02
_SB7 = -2.24409524465858183362e01


@kernel("float64(float64)")
def erfc_p(x):
    """Complementary error function, port of fdlibm s_erf.c."""
    hx = _hi(x)
    ix = hx & 0x7FFFFFFF
    if ix >= 0x7FF00000:
        return float(((hx >> 31) & 1) << 1) + 1.0 / x
    if ix < 0x3FEB0000:
        if ix < 0x3C700000:
            return 1.0 - x
        z = x * x
        r = _PP0 + z * (_PP1 + z * (_PP2 + z * (_PP3 + z * _PP4)))
        s = 1.0 + z * (_QQ1 + z * (_QQ2 + z * (_QQ3 + z * (_QQ4 + z * _QQ5))))
        y = r / s
        if hx < 0x3FD00000:
            return 1.0 - (x + x * y)
        r = x * y
        r += x - 0.5
        return 0.5 - r
    if ix < 0x3FF40000:
        s = abs(x) - 1.0
        P = _PA0 + s * (_PA1 + s * (_PA2 + s * (_PA3 + s * (_PA4 + s * (_PA5 + s * _PA6)))))
        Q = 1.0 + s * (_QA1 + s * (_QA2 + s * (_QA3 + s * (_QA4 + s * (_QA5 + s * _QA6)))))
        if hx >= 0:
            z = 1.0 - _ERX
            return z - P / Q
        z = _ERX + P / Q
        return 1.0 + z
    if ix < 0x403C0000:
        x = abs(x)
        s = 1.0 / (x * x)
        if ix < 0x4006DB6D:
            R = _RA0 + s * (_RA1 + s * (_RA2 + s * (_RA3 + s * (_RA4 + s * (_RA5 + s * (_RA6 + s * _RA7))))))
            S = 1.0 + s * (
                _SA1 + s * (_SA2 + s * (_SA3 + s * (_SA4 + s * (_SA5 + s * (_SA6 + s * (_SA7 + s * _SA8))))))
            )
        else:
            if hx < 0 and ix >= 0x40180000:
                return 2.0 - _TINY
            R = _RB0 + s * (_RB1 + s * (_RB2 + s * (_RB3 + s * (_RB4 + s * (_RB5 + s * _RB6)))))
            S = 1.0 + s * (_SB1 + s * (_SB2 + s * (_SB3 + s * (_SB4 + s * (_SB5 + s * (_SB6 + s * _SB7))))))
        z = _set_lo0(x)
        r = exp_p(-z * z - 0.5625) * exp_p((z - x) * (z + x) + R / S)
        if hx > 0:
            return r / x
        return 2.0 - r / x
    if hx > 0:
        return _TINY * _TINY
    return 2.0 - _TINY


@kernel("float64(float64)")
def ndtri_p(p):
    """Inverse normal CDF: Acklam's rational approximation plus one Halley step."""
    if p <= 0.0:
        return -_INF
    if p >= 1.0:
        return _INF
    if p < _P_LOW:
        q = math.sqrt(-2.0 * log_p(p))
        x = (((((_C[0] * q + _C[1]) * q + _C[2]) * q + _C[3]) * q + _C[4]) * q + _C[5]) / (
            (((_D[0] * q + _D[1]) * q + _D[2]) * q + _D[3]) * q + 1.0
        )
    elif p <= 1.0 - _P_LOW:
        q = p - 0.5
        r = q * q
        x = (
            (((((_A[0] * r + _A[1]) * r + _A[2]) * r + _A[3]) * r + _A[4]) * r + _A[5])
            * q
            / (((((_B[0] * r + _B[1]) * r + _B[2]) * r + _B[3]) * r + _B[4]) * r + 1.0)
        )
    else:
        q = math.sqrt(-2.0 * log_p(1.0 - p))
        x = -(((((_C[0] * q + _C[1]) * q + _C[2]) * q + _C[3]) * q + _C[4]) * q + _C[5]) / (
            (((_D[0] * q + _D[1]) * q + _D[2]) * q + _D[3]) * q + 1.0
        )
    e = 0.5 * erfc_p(-x / SQRT2) - p
    u = e * SQRT2PI * exp_p(0.5 * x * x)
    return x - u / (1.0 + 0.5 * x * u)


@kernel(f"{_A1}({_A1})")
def ndtri_array_p(p):
    """``ndtri_p`` applied elementwise."""
    n = p.shape[0]
    out = np.empty(n, dtype=np.float64)
    for i in range(n):
        out[i] = ndtri_p(p[i])
    return out


@kernel(f"float64({_A1}, int64, int64)")
def nsum_p(a, start, stop):
    """Neumaier-compensated sum of a[start:stop], the rounding of Python 3.12 ``sum``."""
    if stop <= start:
        return 0.0
    f = 0.0 + a[start]
    c = 0.0
    for i in range(start + 1, stop):
        x = a[i]
        t = f + x
        if abs(f) >= abs(x):
            c += (f - t) + x
        else:
            c += (x - t) + f
        f = t
    if c != 0.0 and math.isfinite(c):
        f += c
    return f


@kernel(f"float64({_A1}, int64, {_A1}, int64, int64)")
def ndot_p(a, ia, b, ib, n):
    """Compensated dot product of two strided slices."""
    if n <= 0:
        return 0.0
    f = 0.0 + a[ia] * b[ib]
    c = 0.0
    for i in range(1, n):
        x = a[ia + i] * b[ib + i]
        t = f + x
        if abs(f) >= abs(x):
            c += (f - t) + x
        else:
            c += (x - t) + f
        f = t
    if c != 0.0 and math.isfinite(c):
        f += c
    return f


@kernel(f"float64({_A1})")
def mean_p(a):
    """Compensated mean."""
    n = a.shape[0]
    return nsum_p(a, 0, n) / n


@kernel(f"{_A1}({_A1}, int64)")
def yule_walker_p(y, p):
    """Yule-Walker AR(p) coefficients by Gaussian elimination with partial pivoting."""
    N = y.shape[0]
    if N < 4 * p + 8:
        return np.zeros(p, dtype=np.float64)
    r = np.empty(p + 1, dtype=np.float64)
    for k in range(p + 1):
        r[k] = ndot_p(y, 0, y, k, N - k) / N
    R = np.empty((p, p), dtype=np.float64)
    for i in range(p):
        for j in range(p):
            R[i, j] = r[abs(i - j)]
    for i in range(p):
        R[i, i] = r[0] + 1e-8
    b = np.empty(p, dtype=np.float64)
    for i in range(p):
        b[i] = r[i + 1]
    for k in range(p):
        piv = k
        amax = abs(R[k, k])
        for i in range(k + 1, p):
            v = abs(R[i, k])
            if v > amax:
                amax = v
                piv = i
        if piv != k:
            for j in range(p):
                tmp = R[k, j]
                R[k, j] = R[piv, j]
                R[piv, j] = tmp
            tmp = b[k]
            b[k] = b[piv]
            b[piv] = tmp
        for i in range(k + 1, p):
            lm = R[i, k] / R[k, k]
            for j in range(k + 1, p):
                R[i, j] = R[i, j] - lm * R[k, j]
            b[i] = b[i] - lm * b[k]
    x = np.empty(p, dtype=np.float64)
    for i in range(p - 1, -1, -1):
        s = b[i]
        for j in range(i + 1, p):
            s = s - R[i, j] * x[j]
        x[i] = s / R[i, i]
    return x


@kernel(f"Tuple(({_A1}, {_A1}, {_A1}, float64))({_A1}, float64, int64)")
def ar_fit_p(g, mean_g, p):
    """Centred series, AR(p) fit, innovations and their variance."""
    n = g.shape[0]
    y = np.empty(n, dtype=np.float64)
    for i in range(n):
        y[i] = g[i] - mean_g
    phi = yule_walker_p(y, p)
    Ne = n - p
    eps = np.empty(Ne, dtype=np.float64)
    for t in range(Ne):
        eps[t] = y[p + t]
    for k in range(1, p + 1):
        ph = phi[k - 1]
        for t in range(Ne):
            eps[t] = eps[t] - ph * y[p - k + t]
    sq = np.empty(Ne, dtype=np.float64)
    for t in range(Ne):
        sq[t] = eps[t] * eps[t]
    sig2 = nsum_p(sq, 0, Ne) / Ne
    if 1e-12 > sig2:
        sig2 = 1e-12
    return y, phi, eps, sig2


@kernel(f"{_A1}({_A1}, float64, float64)")
def ar1_filter_p(c, b, init):
    """Sequential recursion y_t = b y_{t-1} + c_{t-1} from y_0 = init."""
    N1 = c.shape[0]
    y = np.empty(N1 + 1, dtype=np.float64)
    y[0] = init
    if b < 1e-12:
        for t in range(N1):
            y[t + 1] = c[t]
        return y
    prev = init
    for t in range(N1):
        prev = b * prev + c[t]
        y[t + 1] = prev
    return y


@kernel(f"{_A1}({_A1}, float64, float64, float64)")
def garch_filter_p(e2, a, b, s2):
    """GARCH(1,1) conditional variances with variance targeting."""
    Ne = e2.shape[0]
    omega = s2 * ((1.0 - a) - b)
    nc = Ne - 1 if Ne >= 1 else 0
    c = np.empty(nc, dtype=np.float64)
    for t in range(nc):
        c[t] = omega + a * e2[t]
    h = ar1_filter_p(c, b, s2)
    fl = 1e-8 * s2
    for t in range(h.shape[0]):
        if h[t] < fl:
            h[t] = fl
    return h


@kernel(f"float64({_A1}, float64, float64, float64)")
def garch_loss_p(e2, a, b, s2):
    """Gaussian quasi-likelihood loss of a GARCH(1,1) candidate (inf outside the box)."""
    if a < 0.0 or b < 0.0 or a > _AMAX or b > _BMAX or a + b > _ABMAX:
        return _INF
    Ne = e2.shape[0]
    if Ne == 0:
        return 0.0
    omega = s2 * ((1.0 - a) - b)
    fl = 1e-8 * s2
    prev = s2
    h = prev
    if h < fl:
        h = fl
    f = 0.0 + (log_p(h) + e2[0] / h)
    comp = 0.0
    for t in range(1, Ne):
        ct = omega + a * e2[t - 1]
        if b < 1e-12:
            prev = ct
        else:
            prev = b * prev + ct
        h = prev
        if h < fl:
            h = fl
        x = log_p(h) + e2[t] / h
        tt = f + x
        if abs(f) >= abs(x):
            comp += (f - tt) + x
        else:
            comp += (x - tt) + f
        f = tt
    if comp != 0.0 and math.isfinite(comp):
        f += comp
    return f


@kernel("boolean(float64, float64)")
def _in_box(a, b):
    """Whether (alpha, beta) satisfies the parameter constraints."""
    return not (a < 0.0 or b < 0.0 or a > _AMAX or b > _BMAX or a + b > _ABMAX)


@kernel(f"Tuple((float64, float64, int64))({_A1}, float64)")
def garch_fit_p(e2, s2):
    """GARCH(1,1) fit: 5 x 6 grid, then at most 40 Nelder-Mead evaluations."""
    best_v = _INF
    best_a = 0.0
    best_b = 0.0
    n_grid = 0
    for ia in range(len(GARCH_GRID_A)):
        a = GARCH_GRID_A[ia]
        for ib in range(len(GARCH_GRID_B)):
            b = GARCH_GRID_B[ib]
            if a + b > _ABMAX:
                continue
            v = garch_loss_p(e2, a, b, s2)
            if _in_box(a, b):
                n_grid += 1
            if v < best_v:
                best_v = v
                best_a = a
                best_b = b
    a0 = best_a
    b0 = best_b
    da = 0.05 if (a0 + 0.05 <= _AMAX and a0 + 0.05 + b0 <= _ABMAX) else -0.05
    db = 0.05 if (b0 + 0.05 <= _BMAX and a0 + b0 + 0.05 <= _ABMAX) else -0.05
    px = np.empty(3, dtype=np.float64)
    py = np.empty(3, dtype=np.float64)
    vals = np.empty(3, dtype=np.float64)
    ta = a0 + da
    if 0.0 > ta:
        ta = 0.0
    if _AMAX < ta:
        ta = _AMAX
    tb = b0 + db
    if 0.0 > tb:
        tb = 0.0
    if _BMAX < tb:
        tb = _BMAX
    px[0] = a0
    py[0] = b0
    px[1] = ta
    py[1] = b0
    px[2] = a0
    py[2] = tb
    vals[0] = best_v
    nm_evals = 1
    for i in range(1, 3):
        if nm_evals >= GARCH_NM_EVALS:
            vals[i] = _INF
            continue
        vals[i] = garch_loss_p(e2, px[i], py[i], s2)
        nm_evals += 1
    order = np.empty(3, dtype=np.int64)
    ox_ = np.empty(3, dtype=np.float64)
    oy_ = np.empty(3, dtype=np.float64)
    ov_ = np.empty(3, dtype=np.float64)
    while nm_evals < GARCH_NM_EVALS:
        for i in range(3):
            order[i] = i
        for i in range(1, 3):
            j = i
            while j > 0:
                u = order[j]
                w = order[j - 1]
                if vals[u] < vals[w] or (vals[u] == vals[w] and u < w):
                    order[j] = w
                    order[j - 1] = u
                    j -= 1
                else:
                    break
        for i in range(3):
            ox_[i] = px[order[i]]
            oy_[i] = py[order[i]]
            ov_[i] = vals[order[i]]
        for i in range(3):
            px[i] = ox_[i]
            py[i] = oy_[i]
            vals[i] = ov_[i]
        dmax = abs(px[1] - px[0])
        d = abs(px[2] - px[0])
        if d > dmax:
            dmax = d
        d = abs(py[1] - py[0])
        if d > dmax:
            dmax = d
        d = abs(py[2] - py[0])
        if d > dmax:
            dmax = d
        if dmax < 1e-6:
            break
        cx = 0.5 * (px[0] + px[1])
        cy = 0.5 * (py[0] + py[1])
        wx = px[2]
        wy = py[2]
        rx = cx + (cx - wx)
        ry = cy + (cy - wy)
        fr = garch_loss_p(e2, rx, ry, s2)
        nm_evals += 1
        if fr < vals[0]:
            ex = cx + 2.0 * (cx - wx)
            ey = cy + 2.0 * (cy - wy)
            if nm_evals < GARCH_NM_EVALS:
                fe = garch_loss_p(e2, ex, ey, s2)
                nm_evals += 1
            else:
                fe = _INF
            if fe < fr:
                px[2] = ex
                py[2] = ey
                vals[2] = fe
            else:
                px[2] = rx
                py[2] = ry
                vals[2] = fr
        elif fr < vals[1]:
            px[2] = rx
            py[2] = ry
            vals[2] = fr
        else:
            if fr < vals[2]:
                oxx = cx + 0.5 * (rx - cx)
                oyy = cy + 0.5 * (ry - cy)
            else:
                oxx = cx + 0.5 * (wx - cx)
                oyy = cy + 0.5 * (wy - cy)
            if nm_evals < GARCH_NM_EVALS:
                fo = garch_loss_p(e2, oxx, oyy, s2)
                nm_evals += 1
            else:
                fo = _INF
            mn = fr
            if vals[2] < mn:
                mn = vals[2]
            if fo < mn:
                px[2] = oxx
                py[2] = oyy
                vals[2] = fo
            else:
                for i in range(1, 3):
                    px[i] = px[0] + 0.5 * (px[i] - px[0])
                    py[i] = py[0] + 0.5 * (py[i] - py[0])
                    if nm_evals < GARCH_NM_EVALS:
                        vals[i] = garch_loss_p(e2, px[i], py[i], s2)
                        nm_evals += 1
                    else:
                        vals[i] = _INF
    i_best = 0
    for i in range(1, 3):
        if vals[i] < vals[i_best]:
            i_best = i
    if vals[i_best] <= best_v:
        a_hat = px[i_best]
        b_hat = py[i_best]
    else:
        a_hat = a0
        b_hat = b0
    return a_hat, b_hat, n_grid + nm_evals - 1


@kernel(f"Tuple((float64, float64, int64, {_A1}, {_A1}, {_A1}, {_A1}, {_A1}))({_A1}, float64)")
def garch_block_p(e2, s2):
    """GARCH fit plus the filtered variances, score derivatives and log-variances."""
    a, b, n = garch_fit_p(e2, s2)
    h = garch_filter_p(e2, a, b, s2)
    Ne = e2.shape[0]
    if Ne >= 2:
        ones = np.ones(Ne - 1, dtype=np.float64)
        e2m = np.empty(Ne - 1, dtype=np.float64)
        hm = np.empty(Ne - 1, dtype=np.float64)
        for t in range(Ne - 1):
            e2m[t] = e2[t]
            hm[t] = h[t]
        d1 = ar1_filter_p(ones, b, 0.0)
        d2 = ar1_filter_p(e2m, b, 0.0)
        d3 = ar1_filter_p(hm, b, 0.0)
    else:
        d1 = np.zeros(1, dtype=np.float64)
        d2 = np.zeros(1, dtype=np.float64)
        d3 = np.zeros(1, dtype=np.float64)
    v = np.empty(h.shape[0], dtype=np.float64)
    for t in range(h.shape[0]):
        v[t] = log_p(h[t])
    return a, b, n, h, d1, d2, d3, v
