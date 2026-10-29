"""Serial float64 building blocks of the recurrent members (fixed summation order, caller-owned buffers).

Weights are one flat vector; each layer reads its block through integer offsets.
"""

from __future__ import annotations

import math

from numba import types

from .runtime import F8, I8, kernel

_A = types.Array(F8, 1, "C")


@kernel(F8(F8))
def sigmoid(x):
    """Logistic function."""
    return 1.0 / (1.0 + math.exp(-x))


@kernel(F8(F8))
def gelu(x):
    """Exact (erf) GELU."""
    return 0.5 * x * (1.0 + math.erf(x / math.sqrt(2.0)))


@kernel(types.void(_A, I8, _A, I8, I8, _A, I8, I8, I8))
def linear_step(src, si, weights, wi, bi, dst, di, ni, no):
    """dst[di:di+no] = W src[si:si+ni] + b."""
    for o in range(no):
        value = 0.0
        for j in range(ni):
            value += weights[wi + o * ni + j] * src[si + j]
        dst[di + o] = value + weights[bi + o]


@kernel(types.void(_A, I8, _A, I8, I8, I8, I8, _A, I8, _A, I8, I8, I8))
def gru_step(src, si, weights, wi, wh, bi, bh, state, hi, dst, di, ni, nh):
    """One GRU cell (PyTorch weight order r, z, n); the new state is written to dst and state."""
    for o in range(nh):
        ir = iz = inn = 0.0
        hr = hz = hn = 0.0
        for j in range(ni):
            x = src[si + j]
            ir += weights[wi + o * ni + j] * x
            iz += weights[wi + (nh + o) * ni + j] * x
            inn += weights[wi + (2 * nh + o) * ni + j] * x
        for j in range(nh):
            h = state[hi + j]
            hr += weights[wh + o * nh + j] * h
            hz += weights[wh + (nh + o) * nh + j] * h
            hn += weights[wh + (2 * nh + o) * nh + j] * h
        r = sigmoid((ir + weights[bi + o]) + (hr + weights[bh + o]))
        z = sigmoid((iz + weights[bi + nh + o]) + (hz + weights[bh + nh + o]))
        n = math.tanh((inn + weights[bi + 2 * nh + o]) + r * (hn + weights[bh + 2 * nh + o]))
        dst[di + o] = (1.0 - z) * n + z * state[hi + o]
    for o in range(nh):
        state[hi + o] = dst[di + o]


@kernel(types.void(_A, I8, _A, I8, I8, _A, I8, I8, F8))
def layernorm_step(src, si, weights, wi, bi, dst, di, n, eps):
    """Layer normalization of n values with affine weights."""
    mean = 0.0
    for j in range(n):
        mean += src[si + j]
    mean /= n
    variance = 0.0
    for j in range(n):
        delta = src[si + j] - mean
        variance += delta * delta
    inverse = 1.0 / math.sqrt(variance / n + eps)
    for j in range(n):
        dst[di + j] = (src[si + j] - mean) * inverse * weights[wi + j] + weights[bi + j]
