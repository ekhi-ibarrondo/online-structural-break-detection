"""Compiled forward pass of the reads_two_rebuilds network: input transform, two GRU layers and an MLP head."""

from __future__ import annotations

import math

from .runtime import kernel


@kernel("void(float64[::1], float64[::1], float64[::1], float64[::1])")
def transform(xraw, med, sc, out):
    """asinh((x - median) / scale), clipped to [-8, 8]; non-finite values become 0."""
    for j in range(xraw.shape[0]):
        v = math.asinh((xraw[j] - med[j]) / sc[j])
        if v > 8.0:
            v = 8.0
        elif v < -8.0:
            v = -8.0
        if not math.isfinite(v):
            v = 0.0
        out[j] = v


@kernel(
    "void(float64[::1], float64[::1], float64[:, ::1], float64[::1], float64[:, ::1], float64[::1], "
    "float64[::1], float64[::1])"
)
def gru_layer(x, h, WT_ih, b_ih, WT_hh, b_hh, gi, gh):
    """One GRU cell update of the hidden state ``h`` in place (fixed summation order)."""
    G = gi.shape[0]
    Hh = h.shape[0]
    for i in range(G):
        gi[i] = 0.0
        gh[i] = 0.0
    for j in range(x.shape[0]):
        xj = x[j]
        for i in range(G):
            gi[i] += WT_ih[j, i] * xj
    for j in range(Hh):
        hj = h[j]
        for i in range(G):
            gh[i] += WT_hh[j, i] * hj
    for i in range(G):
        gi[i] += b_ih[i]
        gh[i] += b_hh[i]
    for i in range(Hh):
        r = 1.0 / (1.0 + math.exp(-(gi[i] + gh[i])))
        zz = 1.0 / (1.0 + math.exp(-(gi[Hh + i] + gh[Hh + i])))
        nn_ = math.tanh(gi[2 * Hh + i] + r * gh[2 * Hh + i])
        h[i] = (h[i] - nn_) * zz + nn_


@kernel(
    "float64(float64[::1], float64[::1], float64[::1], float64[:, ::1], float64[::1], float64[:, ::1], "
    "float64[::1], float64[:, ::1], float64[::1], float64[:, ::1], float64[::1], float64[:, ::1], "
    "float64[::1], float64[::1], float64, float64[::1], float64[::1], float64[::1])"
)
def net_step(x, h1, h2, WT_ih0, b_ih0, WT_hh0, b_hh0, WT_ih1, b_ih1, WT_hh1, b_hh1, W1T, b1, w2, b2, gi, gh, a):
    """Two stacked GRU updates, then a GELU MLP on [h2, x]; returns the logit."""
    gru_layer(x, h1, WT_ih0, b_ih0, WT_hh0, b_hh0, gi, gh)
    gru_layer(h1, h2, WT_ih1, b_ih1, WT_hh1, b_hh1, gi, gh)
    Hh = h2.shape[0]
    na = a.shape[0]
    for k in range(na):
        a[k] = 0.0
    for j in range(Hh):
        v = h2[j]
        for k in range(na):
            a[k] += W1T[j, k] * v
    for j in range(x.shape[0]):
        v = x[j]
        for k in range(na):
            a[k] += W1T[Hh + j, k] * v
    acc = 0.0
    for k in range(na):
        ak = a[k] + b1[k]
        g = ak * 0.5 * (1.0 + math.erf(ak * 0.7071067811865476))
        acc += w2[k] * g
    return acc + b2
