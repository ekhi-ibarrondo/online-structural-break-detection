"""Causal GRU member shared by broken_or_not, which_one_broke and blind_to_size.

Per series the history fixes the standardization, an AR(5) filter, the rank tables and the
GRU hidden state (warmed up on the last 1000 history points). The raw score of a point is
the network logit minus ``b`` times an offset: the mean logit the same network produces on
up to three pseudo-online windows cut from the end of the history. A guard checks every
context first; if one is degenerate the member returns a fixed per-step fallback value.
"""

from __future__ import annotations

import math

import numpy as np

from ..kernels import sequence_net as k

H_KEEP = 1000
N_WIN = 3
WIN = 250
T_CAL = 25
N_C_MIN = 600
N_MIN = 856
SD_FLOOR = 1e-8
C_MAX = 1e12
N_KEYS = 20


class Model:
    """Flat float64 weight vector ``W``, the offset of each tensor ``OFF`` and the hidden size."""

    def __init__(self, npz) -> None:
        z = np.load(npz, allow_pickle=False)
        self.W = np.ascontiguousarray(z["W"], dtype=np.float64)
        self.OFF = np.ascontiguousarray(z["OFF"], dtype=np.int64)
        self.H = int(z["H"][0])
        if self.OFF.shape != (N_KEYS,) or not np.isfinite(self.W).all():
            raise ValueError("malformed network weights")


def scale_blind_history(xh: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """Volatility-scaled history and the transform state that continues into the online stream."""
    mu = float(xh.mean())
    v0 = float(((xh - mu) ** 2).mean())
    state = np.array([mu, v0, k.FLOOR * v0], dtype=np.float64)
    out = np.empty(len(xh), dtype=np.float64)
    k.scale_blind_history(xh, state, out)
    return out, state


class _Stream:
    """Streaming state of one network on one context (a prefix of the history)."""

    def __init__(self, model: Model, xh: np.ndarray, scale_blind: bool) -> None:
        self.sb_state = None
        if scale_blind:
            xh, self.sb_state = scale_blind_history(xh)
        nc = len(xh)
        mu = xh.mean()
        sd = math.sqrt(((xh - mu) ** 2).mean())
        C = (xh - mu) / sd
        N = nc - k.P
        X = np.empty((N, k.P))
        for q in range(k.P):
            X[:, q] = C[k.P - 1 - q : k.P - 1 - q + N]
        y = C[k.P :]
        phi = np.linalg.solve(X.T @ X, X.T @ y)
        eh = y - X @ phi
        se = math.sqrt(((eh - eh.mean()) ** 2).mean()) + 1e-12
        uh = eh / se
        self.srt = np.sort(uh)
        self.srtx = np.sort(C)
        self.tabu = k.ndtri_table(N)
        self.tabx = k.ndtri_table(nc)
        hl = min(N, H_KEEP)
        s0 = N - hl
        hu = uh[s0:]
        hz = C[s0 + k.P :]
        lognc, logse = math.log(nc) / 8.0, math.log(se)
        F = np.zeros((hl, k.F_IN))
        F[:, 0] = np.clip(hu, -8, 8)
        F[:, 1] = self.tabu[np.searchsorted(self.srt, hu, "left") + np.searchsorted(self.srt, hu, "right")]
        F[:, 2] = np.minimum(np.abs(hu), 8)
        F[:, 5] = lognc
        F[:, 6] = logse
        F[:, 7] = np.clip(hz, -8, 8)
        F[:, 8] = self.tabx[np.searchsorted(self.srtx, hz, "left") + np.searchsorted(self.srtx, hz, "right")]
        H = model.H
        self.m = model
        self.h1, self.h2 = np.zeros(H), np.zeros(H)
        self.tmp = np.empty(H)
        n = H + k.EMB + 1 + k.F_IN
        self.Z, self.Y, self.work = np.empty(n), np.empty(k.EMB + k.F_IN), np.empty(n + k.DH + 1)
        k.warm(np.ascontiguousarray(F), model.W, model.OFF, self.h1, self.h2, self.tmp, H)
        self.phi = np.ascontiguousarray(phi)
        self.lags = np.ascontiguousarray(C[nc - k.P :])
        self.sc = np.array([mu, sd, se, lognc, logse, 0.0])

    def step(self, x) -> float:
        """Network logit of the next point (after the volatility scaling if enabled)."""
        x = float(x)
        if self.sb_state is not None:
            x = k.scale_blind_step(x, self.sb_state)
        return k.step(
            x,
            self.sc,
            self.phi,
            self.lags,
            self.srt,
            self.tabu,
            self.srtx,
            self.tabx,
            self.m.W,
            self.m.OFF,
            self.h1,
            self.h2,
            self.tmp,
            self.Z,
            self.Y,
            self.work,
        )


def offset(model: Model, xh: np.ndarray, scale_blind: bool) -> float:
    """Mean logit over up to N_WIN pseudo-online windows of WIN history points (steps T_CAL..WIN-1)."""
    n = len(xh)
    K = min(N_WIN, (n - N_C_MIN) // WIN)
    acc = 0.0
    for j in range(1, K + 1):
        nc = n - WIN * j
        s = _Stream(model, xh[:nc], scale_blind)
        v = np.empty(WIN)
        for t in range(WIN):
            v[t] = s.step(xh[nc + t])
        acc += float(v[T_CAL:].mean())
    return acc / K


def _context_ok(xh: np.ndarray, scale_blind: bool) -> str | None:
    """None if the network's AR(5) fit on this context is well posed, else the reason."""
    if scale_blind:
        variance = float(((xh - xh.mean()) ** 2).mean())
        if not (math.isfinite(variance) and variance > 0.0):
            return "original_variance"
        xh, _ = scale_blind_history(xh)
    mu = xh.mean()
    sd = math.sqrt(((xh - mu) ** 2).mean())
    if not (math.isfinite(sd) and sd > SD_FLOOR):
        return "sd"
    C = (xh - mu) / sd
    N = len(xh) - k.P
    X = np.empty((N, k.P))
    for q in range(k.P):
        X[:, q] = C[k.P - 1 - q : k.P - 1 - q + N]
    xtx = X.T @ X
    if not np.all(np.isfinite(xtx)):
        return "xtx"
    cond = float(np.linalg.cond(xtx))
    if not (math.isfinite(cond) and cond <= C_MAX):
        return "cond"
    try:
        phi = np.linalg.solve(xtx, X.T @ C[k.P :])
    except np.linalg.LinAlgError:
        return "linalg"
    if not np.all(np.isfinite(phi)):
        return "phi"
    return None


def guard(xh: np.ndarray, scale_blind: bool) -> str | None:
    """None if the network can run on this history and on all its offset contexts."""
    if xh.ndim != 1:
        return "shape"
    n = len(xh)
    if n < N_MIN:
        return "n"
    if not np.all(np.isfinite(xh)):
        return "nonfinite"
    contexts = [n] + [n - WIN * j for j in range(1, min(N_WIN, (n - N_C_MIN) // WIN) + 1)]
    for nc in contexts:
        why = _context_ok(xh[:nc], scale_blind)
        if why is not None:
            return f"{why}@{nc}"
    return None


class _State:
    """Per-series state: the running stream and offset, the failure reason and the step count."""

    __slots__ = ("inner", "t", "why")

    def __init__(self) -> None:
        self.inner, self.why, self.t = None, None, 0


class SequenceNet:
    """One network with offset coefficient ``b``.

    A degenerate history or a failing point either drops the member (``strict``) or
    switches it to a fixed per-step fallback value for the rest of the series.
    """

    feature_inputs: tuple[str, ...] = ()

    def __init__(self, weights_npz, b: float, fallback_npy, scale_blind: bool, strict: bool) -> None:
        self.model = Model(weights_npz)
        self.b = float(b)
        self.scale_blind = scale_blind
        self.strict = strict
        fb = np.load(fallback_npy, allow_pickle=False)
        self.fb = np.ascontiguousarray(fb, dtype=np.float64).ravel()
        if not (self.fb.size >= 1 and np.all(np.isfinite(self.fb))):
            raise ValueError("fallback table must be finite")

    def start(self, history) -> _State:
        st = _State()
        try:
            xh = np.ascontiguousarray(np.asarray(history, dtype=np.float64))
            why = guard(xh, self.scale_blind)
            if why is None:
                stream = _Stream(self.model, xh, self.scale_blind)
                st.inner = (stream, self.b * offset(self.model, xh, self.scale_blind))
            else:
                st.why = why
        except MemoryError:
            raise
        except Exception as e:
            st.inner, st.why = None, "init:" + type(e).__name__
        if self.strict and st.why:
            raise ValueError(f"network member degraded: {st.why}")
        return st

    def step(self, st: _State, x: float, inputs=()) -> tuple[list[float], None]:
        """Raw score of the point (network logit minus the history offset, or the fallback)."""
        t = st.t
        st.t = t + 1
        if st.inner is not None:
            try:
                xf = float(x)
                if math.isfinite(xf):
                    stream, shift = st.inner
                    out = stream.step(xf) - shift
                    if math.isfinite(out):
                        return [out], None
                    st.why = "nonfinite_out"
                else:
                    st.why = "nonfinite_x"
            except MemoryError:
                raise
            except Exception as e:
                st.why = "step:" + type(e).__name__
            st.inner = None
        if self.strict:
            raise ValueError(f"network member degraded: {st.why}")
        return [float(self.fb[min(t, self.fb.size - 1)])], None
