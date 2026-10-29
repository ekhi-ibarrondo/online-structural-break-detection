"""Compiled step of the causal GRU members (broken_or_not, which_one_broke, blind_to_size).

Per point the network sees 9 inputs: the AR(5) innovation and its rank PIT, its absolute
value, a step clock, two history constants and the standardized level with its rank PIT.
Two GRU layers feed a small head that splices a first logit with the inputs before a
second, layer-normalized MLP. The rank PIT uses the inverse normal CDF ``ndtri``, a port of
the Cephes implementation (Stephen L. Moshier), as used by SciPy and PyTorch.
"""

from __future__ import annotations

import math

import numpy as np
from numba import types

from .network import gelu, gru_step, layernorm_step, linear_step
from .runtime import A1, A2, F8, I1, I8, kernel

P = 5
F_IN = 9
EMB = 16
HH = 32
DH = 64

_P0 = np.array(
    [
        -5.99633501014107895267e1,
        9.80010754185999661536e1,
        -5.66762857469070293439e1,
        1.39312609387279679503e1,
        -1.23916583867381258016e0,
    ]
)
_Q0 = np.array(
    [
        1.95448858338141759834e0,
        4.67627912898881538453e0,
        8.63602421390890590575e1,
        -2.25462687854119370527e2,
        2.00260212380060660359e2,
        -8.20372256168333339912e1,
        1.59056225126211695515e1,
        -1.18331621121330003142e0,
    ]
)
_P1 = np.array(
    [
        4.05544892305962419923e0,
        3.15251094599893866154e1,
        5.71628192246421288162e1,
        4.40805073893200834700e1,
        1.46849561928858024014e1,
        2.18663306850790267539e0,
        -1.40256079171354495875e-1,
        -3.50424626827848203418e-2,
        -8.57456785154685413611e-4,
    ]
)
_Q1 = np.array(
    [
        1.57799883256466749731e1,
        4.53907635128879210584e1,
        4.13172038254672030440e1,
        1.50425385692907503408e1,
        2.50464946208309415979e0,
        -1.42182922854787788574e-1,
        -3.80806407691578277194e-2,
        -9.33259480895457427372e-4,
    ]
)
_P2 = np.array(
    [
        3.23774891776946035970e0,
        6.91522889068984211695e0,
        3.93881025292474443415e0,
        1.33303460815807542389e0,
        2.01485389549179081538e-1,
        1.23716634817820021358e-2,
        3.01581553508235416007e-4,
        2.65806974686737550832e-6,
        6.23974539184983293730e-9,
    ]
)
_Q2 = np.array(
    [
        6.02427039364742014255e0,
        3.67983563856160859403e0,
        1.37702099489081330271e0,
        2.16236993594496635890e-1,
        1.34204006088543189037e-2,
        3.28014464682127739104e-4,
        2.89247864745380683936e-6,
        6.79019408009981274425e-9,
    ]
)


@kernel(F8(F8, A1))
def _polevl(x, c):
    """Horner evaluation of the polynomial with coefficients c."""
    ans = c[0]
    for i in range(1, c.shape[0]):
        ans = ans * x + c[i]
    return ans


@kernel(F8(F8, A1))
def _p1evl(x, c):
    """Horner evaluation with an implicit leading coefficient 1."""
    ans = x + c[0]
    for i in range(1, c.shape[0]):
        ans = ans * x + c[i]
    return ans


@kernel(F8(F8, A1, A1, A1, A1, A1, A1))
def _ndtri(y0, p0, q0, p1, q1, p2, q2):
    """Inverse of the standard normal CDF (Cephes ``ndtri``)."""
    if y0 <= 0.0:
        return -math.inf
    if y0 >= 1.0:
        return math.inf
    code = 1
    y = y0
    if y > (1.0 - 0.13533528323661269189):
        y = 1.0 - y
        code = 0
    if y > 0.13533528323661269189:
        y = y - 0.5
        y2 = y * y
        x = y + y * (y2 * _polevl(y2, p0) / _p1evl(y2, q0))
        return x * 2.50662827463100050242e0
    x = math.sqrt(-2.0 * math.log(y))
    x0 = x - math.log(x) / x
    z = 1.0 / x
    if x < 8.0:
        x1 = z * _polevl(z, p1) / _p1evl(z, q1)
    else:
        x1 = z * _polevl(z, p2) / _p1evl(z, q2)
    x = x0 - x1
    if code != 0:
        x = -x
    return x


@kernel(types.void(A1, F8, A1, A1, A1, A1, A1, A1))
def _pit_table(out, cnt, p0, q0, p1, q1, p2, q2):
    """out[j] = ndtri((j + 1) / (2 (cnt + 1))): normal score of mid-rank j / 2."""
    for j in range(out.shape[0]):
        out[j] = _ndtri((j + 1.0) / (2.0 * (cnt + 1.0)), p0, q0, p1, q1, p2, q2)


def ndtri_table(cnt: int) -> np.ndarray:
    """Normal scores of the 2 cnt + 1 possible mid-ranks against a sample of size cnt."""
    out = np.empty(2 * cnt + 1)
    _pit_table(out, float(cnt), _P0, _Q0, _P1, _Q1, _P2, _Q2)
    return out


@kernel(I8(A1, F8))
def _count_lt(a, v):
    """Number of sorted values strictly below v."""
    lo, hi = 0, a.shape[0]
    while lo < hi:
        mid = (lo + hi) // 2
        if a[mid] < v:
            lo = mid + 1
        else:
            hi = mid
    return lo


@kernel(I8(A1, F8))
def _count_le(a, v):
    """Number of sorted values at or below v."""
    lo, hi = 0, a.shape[0]
    while lo < hi:
        mid = (lo + hi) // 2
        if a[mid] <= v:
            lo = mid + 1
        else:
            hi = mid
    return lo


@kernel(types.void(A1, A1, I1, A1, A1, A1, I8))
def _gru2(f, W, off, h1, h2, tmp, H):
    """Two stacked GRU layers."""
    gru_step(f, 0, W, off[0], off[1], off[2], off[3], h1, 0, tmp, 0, F_IN, H)
    gru_step(h1, 0, W, off[4], off[5], off[6], off[7], h2, 0, tmp, 0, H, H)


@kernel(types.void(A2, A1, I1, A1, A1, A1, I8))
def warm(F, W, off, h1, h2, tmp, H):
    """Run the GRU over the history inputs (rows of F) to initialize the hidden states."""
    f = np.empty(F_IN)
    for s in range(F.shape[0]):
        for j in range(F_IN):
            f[j] = F[s, j]
        _gru2(f, W, off, h1, h2, tmp, H)


@kernel(F8(A1, A1, I1, A1, A1, A1, A1, I8))
def _head(f, W, off, h2, Z, Y, work, H):
    """Output logit from the top hidden state h2 and the step inputs f."""
    for j in range(H):
        Z[j] = h2[j]
    linear_step(h2, 0, W, off[8], off[9], Z, H, H, EMB)
    for j in range(EMB):
        Y[j] = Z[H + j]
    for j in range(F_IN):
        Y[EMB + j] = f[j]
    linear_step(Y, 0, W, off[10], off[11], work, 0, EMB + F_IN, HH)
    for j in range(HH):
        work[j] = gelu(work[j])
    linear_step(work, 0, W, off[12], off[13], work, HH, HH, 1)
    Z[H + EMB] = work[HH]
    for j in range(F_IN):
        Z[H + EMB + 1 + j] = f[j]
    n = H + EMB + 1 + F_IN
    layernorm_step(Z, 0, W, off[14], off[15], work, 0, n, 1e-5)
    linear_step(work, 0, W, off[16], off[17], work, n, n, DH)
    for j in range(DH):
        work[n + j] = gelu(work[n + j])
    linear_step(work, n, W, off[18], off[19], work, n + DH, DH, 1)
    return work[n + DH]


@kernel(F8(F8, A1, A1, A1, A1, A1, A1, A1, A1, I1, A1, A1, A1, A1, A1, A1))
def step(x, sc, phi, lags, srt, tabu, srtx, tabx, W, off, h1, h2, tmp, Z, Y, work):
    """One online point; ``sc`` = [mu, sd, se, log n / 8, log se, t], ``lags`` oldest first."""
    H = h1.shape[0]
    z = (x - sc[0]) / sc[1]
    pred = 0.0
    for q in range(P):
        pred += phi[q] * lags[P - 1 - q]
    u = (z - pred) / sc[2]
    f = np.empty(F_IN)
    f[0] = min(max(u, -8.0), 8.0)
    f[1] = tabu[_count_lt(srt, u) + _count_le(srt, u)]
    f[2] = min(abs(u), 8.0)
    f[3] = 1.0
    f[4] = math.log1p(sc[5]) / 7.0
    f[5] = sc[3]
    f[6] = sc[4]
    f[7] = min(max(z, -8.0), 8.0)
    f[8] = tabx[_count_lt(srtx, z) + _count_le(srtx, z)]
    _gru2(f, W, off, h1, h2, tmp, H)
    lo = _head(f, W, off, h2, Z, Y, work, H)
    for q in range(P - 1):
        lags[q] = lags[q + 1]
    lags[P - 1] = z
    sc[5] += 1.0
    return lo


HALF_LIFE = 50.0
LAM = 2.0 ** (-1.0 / HALF_LIFE)
FLOOR = 1e-24


@kernel(F8(F8, A1))
def scale_blind_step(x, state):
    """Divide x minus the context mean by the strictly past EWMA volatility; ``state`` = [mean, var, floor]."""
    d = x - state[0]
    out = d / math.sqrt(max(state[1], state[2]))
    state[1] = LAM * state[1] + (1.0 - LAM) * d * d
    return out


@kernel(types.void(A1, A1, A1))
def scale_blind_history(x, state, out):
    """``scale_blind_step`` over the history, carrying the state forward."""
    for t in range(x.shape[0]):
        out[t] = scale_blind_step(x[t], state)
