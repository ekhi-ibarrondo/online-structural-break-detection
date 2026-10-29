"""Per-series spectral features of slow_and_fast on AR-whitened, winsorized innovations.

Per point the innovation buffer gives 15 energy channels: 9 bins of a sliding Hann-window
STFT and 6 undecimated Haar wavelet scales. They are aggregated with fast and slow
exponential forgetting, over the whole online prefix and over a scan of suffix windows,
and summarized by cepstral distances to the historical spectrum. Every statistic is
standardized against its own-series null, obtained by replaying the same online statistic
from many start positions inside the history. Seven rank, CUSUM and bridge statistics and
five covariates complete the 96 features.
"""

from __future__ import annotations

import math

import numpy as np

P_MAX = 10
L_STFT = 16
N_BINS = 9
HAAR_H = (1, 2, 4, 8, 16, 32)
N_HAAR = len(HAAR_H)
BUF = 64
N_CH = N_BINS + N_HAAR
LAM_FAST = 0.90
LAM_SLOW = 0.98
SCAN_W = (1, 2, 4, 8, 16, 32, 64, 128, 256, 512)
AGE_GRID = (1, 2, 3, 4, 6, 8, 12, 16, 24, 32, 48, 64, 96, 128, 192, 256, 384, 512, 768, 1024)
N_STARTS = 96
MIN_STARTS = 32
M_DENSE = 1024
CLIP_Q = 0.995
TAIL_Q = 0.90
BRIDGE_DELTA = 0.05
NHIST_EDGES = np.array([2000.0, 3000.0, 4000.0])
TAILRATIO_EDGES = np.array([2.4, 2.75, 3.1])
INNOVVAR_EDGES = np.array([0.25, 0.5, 0.8])
N_CEPS_COEF = 4
N_CEPS = N_CEPS_COEF + 2 + 2
N_SPEC = 4 * N_CH + 3 * N_CEPS
N_SEC = 7
N_COV = 5
N_FEAT = N_SPEC + N_SEC + N_COV
SIG_FLOOR = 1e-9


def _build_matrices():
    """Linear map from the 64-point buffer to STFT and Haar coefficients, and the channel sum of squares."""
    M = np.zeros((N_BINS + 7 + N_HAAR, BUF))
    j = np.arange(L_STFT)
    w = 0.5 * (1.0 - np.cos(2.0 * np.pi * (j + 0.5) / L_STFT))
    norm = 1.0 / math.sqrt(float(np.sum(w * w)))
    lag = (L_STFT - 1 - j).astype(float)
    for k in range(N_BINS):
        om = k * np.pi / (N_BINS - 1)
        M[k, BUF - L_STFT :] = w * np.cos(om * lag) * norm
        if 1 <= k <= 7:
            M[N_BINS + k - 1, BUF - L_STFT :] = -w * np.sin(om * lag) * norm
    for i, h in enumerate(HAAR_H):
        row = N_BINS + 7 + i
        c = math.sqrt(h / 2.0) / h
        M[row, BUF - h :] = c
        M[row, BUF - 2 * h : BUF - h] = -c
    MSQ = np.zeros((N_CH, N_BINS + 7 + N_HAAR))
    for k in range(N_BINS):
        MSQ[k, k] = 1.0
        if 1 <= k <= 7:
            MSQ[k, N_BINS + k - 1] = 1.0
    for i in range(N_HAAR):
        MSQ[N_BINS + i, N_BINS + 7 + i] = 1.0
    return M, MSQ


M_TRANS, M_SQ = _build_matrices()


def _build_dct():
    """DCT-I rows giving the first cepstral coefficients of the 9 log-energies."""
    D = np.zeros((N_CEPS_COEF, N_BINS))
    for q in range(N_CEPS_COEF):
        for k in range(N_BINS):
            a = 0.5 if k in (0, N_BINS - 1) else 1.0
            D[q, k] = a * math.cos(math.pi * k * q / (N_BINS - 1)) / (N_BINS - 1)
    return D


DCT = _build_dct()
DCT_T = np.ascontiguousarray(DCT.T)
LR_BAND = np.array([2 * L_STFT] * N_BINS + [min(4 * h, 64) for h in HAAR_H], dtype=float)
LR_MAXLAG = int(LR_BAND.max())

_SQRT = np.sqrt(np.arange(0, M_DENSE + 2, dtype=float))
_AR_CH = np.arange(N_CH)
_AR4 = np.arange(4)


def _windows_for(m):
    """Suffix windows shorter than the age ``m``, plus ``m`` itself."""
    ws = [w for w in SCAN_W if w < m]
    ws.append(m)
    return np.array(ws, dtype=np.int64)


_WIN = [None] + [_windows_for(m) for m in range(1, M_DENSE + 1)]
_WIN_SQRT = [None] + [np.sqrt(_WIN[m].astype(float))[:, None] for m in range(1, M_DENSE + 1)]
_WIN_INV = [None] + [(1.0 / _WIN[m].astype(float))[:, None] for m in range(1, M_DENSE + 1)]
_LOG_AGE = np.log(np.array(AGE_GRID, dtype=float))
_INV_SQRT_DESC = [None] + [1.0 / _SQRT[m:0:-1] for m in range(1, M_DENSE + 1)]
_BOUNDARY = [None, None]
_KOVERM = [None, None]
for _m in range(2, M_DENSE + 1):
    _u = np.arange(1, _m, dtype=float) / _m
    _KOVERM.append(_u)
    _BOUNDARY.append(1.0 / np.sqrt(np.maximum(_u * (1.0 - _u), BRIDGE_DELTA * (1.0 - BRIDGE_DELTA))))
_EWW_F = (1.0 - LAM_FAST ** np.arange(0, M_DENSE + 2, dtype=float)) / (1.0 - LAM_FAST)
_EWW_S = (1.0 - LAM_SLOW ** np.arange(0, M_DENSE + 2, dtype=float)) / (1.0 - LAM_SLOW)
_INV_EWW_F = np.concatenate([[0.0], 1.0 / _EWW_F[1:]])
_INV_EWW_S = np.concatenate([[0.0], 1.0 / _EWW_S[1:]])
_LAMPOW_F = LAM_FAST ** np.arange(0, M_DENSE + 2, dtype=float)
_LAMPOW_S = LAM_SLOW ** np.arange(0, M_DENSE + 2, dtype=float)

_I_LEF = slice(0, N_CH)
_I_LES = slice(N_CH, 2 * N_CH)
_I_LEC = slice(2 * N_CH, 3 * N_CH)
_I_SCAN = slice(3 * N_CH, 4 * N_CH)
_I_CEPS = slice(4 * N_CH, 4 * N_CH + 3 * N_CEPS)
_SIG_SQRT_SCALE = np.zeros(N_SPEC, dtype=bool)
_MU_SQRT_SCALE = np.zeros(N_SPEC, dtype=bool)
_SIG_SQRT_SCALE[_I_LEC] = True
_c0 = 4 * N_CH + 2 * N_CEPS
_SIG_SQRT_SCALE[_c0 : _c0 + N_CEPS] = True
for _q in (0, 1, 6, 7):
    _MU_SQRT_SCALE[_c0 + _q] = True


def fit_ar_yw(x, p_max=P_MAX):
    """Yule-Walker AR fit with BIC order selection over 0..p_max; returns (a, p, sigma^2)."""
    n = x.shape[0]
    xc = x - x.mean()
    nfft = 1
    while nfft < 2 * n:
        nfft <<= 1
    f = np.fft.rfft(xc, nfft)
    acov = np.fft.irfft(f * np.conj(f), nfft)[: p_max + 1] / n
    if not (acov[0] > 0):
        return np.zeros(0), 0, 1.0
    sig2 = float(acov[0])
    best_bic = n * math.log(sig2)
    best_p = 0
    best_a = np.zeros(0)
    best_sig2 = sig2
    a = np.zeros(0)
    logn = math.log(n)
    for p in range(1, p_max + 1):
        if p == 1:
            k = acov[1] / sig2
        else:
            k = (acov[p] - float(np.dot(a, acov[p - 1 : 0 : -1]))) / sig2
        k = float(k)
        if abs(k) >= 0.9999:
            break
        a_new = np.empty(p)
        a_new[: p - 1] = a - k * a[::-1]
        a_new[p - 1] = k
        a = a_new
        sig2 = sig2 * (1.0 - k * k)
        if not (sig2 > 0):
            break
        bic = n * math.log(sig2) + p * logn
        if bic < best_bic - 1e-12:
            best_bic, best_p, best_a, best_sig2 = bic, p, a.copy(), sig2
    return best_a, best_p, best_sig2


def _ew_full(Y, lam, block=128):
    """Exponentially weighted running sums of the rows of Y, computed blockwise."""
    n, c = Y.shape
    A = np.empty_like(Y)
    carry = np.zeros(c)
    pw = lam ** np.arange(1, block + 1, dtype=float)
    ipw = lam ** (-np.arange(1, block + 1, dtype=float))
    for b0 in range(0, n, block):
        Yb = Y[b0 : b0 + block]
        nb = Yb.shape[0]
        S = np.cumsum(Yb * ipw[:nb, None], axis=0) * pw[:nb, None]
        S += carry[None, :] * pw[:nb, None]
        A[b0 : b0 + nb] = S
        carry = S[-1]
    return A


def _long_run_sd(Y, band):
    """Bartlett long-run sd of every column, with a per-column bandwidth."""
    n, c = Y.shape
    Yc = Y - Y.mean(axis=0)
    nfft = 1
    while nfft < 2 * n:
        nfft <<= 1
    F = np.fft.rfft(Yc, nfft, axis=0)
    ac = np.fft.irfft(F * np.conj(F), nfft, axis=0)[: LR_MAXLAG + 1] / n
    lags = np.arange(LR_MAXLAG + 1, dtype=float)[:, None]
    wts = np.clip(1.0 - lags / (band[None, :] + 1.0), 0.0, None)
    wts[1:] *= 2.0
    v = np.sum(wts * ac, axis=0)
    v = np.maximum(v, 1e-12 * np.maximum(ac[0], 1e-300))
    return np.sqrt(v)


_ONES_S = np.full(N_BINS, 1.0)
_ONES_H = np.full(N_HAAR, 1.0)
_MEAN_S = np.full(N_BINS, 1.0 / N_BINS)
_MEAN_H = np.full(N_HAAR, 1.0 / N_HAAR)


def _cepstral(LE, lhist, out):
    """Cepstral distances (full and shape) and the first 4 cepstral deviations, STFT and Haar."""
    DL = LE - lhist
    DS = DL[:, :N_BINS]
    DH = DL[:, N_BINS:]
    ss = (DS * DS) @ _ONES_S
    ms = DS @ _MEAN_S
    out[:, 0] = np.sqrt(ss)
    out[:, 1] = np.sqrt(np.maximum(ss - N_BINS * (ms * ms), 0.0))
    out[:, 2 : 2 + N_CEPS_COEF] = DS @ DCT_T
    sh = (DH * DH) @ _ONES_H
    mh = DH @ _MEAN_H
    out[:, 6] = np.sqrt(sh)
    out[:, 7] = np.sqrt(np.maximum(sh - N_HAAR * (mh * mh), 0.0))
    return out


class SeriesState:
    """Per-series constants and online accumulators."""

    __slots__ = (
        "mu_x",
        "inv_sd_x",
        "a_pad",
        "lags",
        "inv_sd_e",
        "clip",
        "buf",
        "m",
        "A_f",
        "A_s",
        "P",
        "S7",
        "mu_y",
        "inv_sig_y",
        "lhist",
        "floor",
        "mu_tab",
        "inv_sig_tab",
        "sorted_abs",
        "sorted_eps",
        "n_hist_eps",
        "med",
        "mean_epc",
        "sd_epc",
        "mean_sq",
        "sd_sq_lr",
        "mean_rank",
        "sd_rank",
        "sd_wil",
        "tail_thr",
        "p_tail",
        "sd_tail",
        "mean_sp",
        "sd_sp",
        "prev_sign",
        "cov",
        "raw",
        "feat",
        "ceps_work",
        "LE",
        "E3",
        "cap",
    )


class FeatureEngine:
    """``start(history)`` builds the per-series state; ``step_features(state, x)`` returns 96 features."""

    def start(self, x_hist):
        """Fit the AR filter, the clipping level, the replayed null tables and the secondary constants."""
        x = np.asarray(x_hist, dtype=np.float64)
        n = x.shape[0]
        st = SeriesState()
        mu = float(x.mean())
        sd = float(x.std())
        if not (sd > 0):
            sd = 1.0
        st.mu_x = mu
        st.inv_sd_x = 1.0 / sd
        xz = (x - mu) / sd
        a, p, _ = fit_ar_yw(xz, min(P_MAX, max(0, n // 100)))
        a_pad = np.zeros(P_MAX)
        a_pad[:p] = a
        st.a_pad = a_pad
        e = xz[p:].copy()
        for j in range(1, p + 1):
            e -= a[j - 1] * xz[p - j : n - j]
        sd_e = float(e.std())
        if not (sd_e > 0):
            sd_e = 1.0
        st.inv_sd_e = 1.0 / sd_e
        eps = e / sd_e
        abs_eps = np.abs(eps)
        clip = float(np.quantile(abs_eps, CLIP_Q))
        if not (clip > 0):
            clip = 1.0
        st.clip = clip
        epc = np.clip(eps, -clip, clip)
        lags = np.zeros(P_MAX)
        k = min(P_MAX, n)
        for j in range(k):
            lags[j] = xz[n - 1 - j]
        st.lags = lags
        buf = np.zeros(2 * BUF)
        tail = epc[-BUF:]
        buf[BUF - tail.shape[0] : BUF] = tail
        buf[2 * BUF - tail.shape[0] :] = tail
        st.buf = buf
        st.m = 0
        nE = epc.shape[0]
        if nE < BUF + MIN_STARTS + 2:
            reps = int(math.ceil((BUF + MIN_STARTS + 2) / max(nE, 1))) + 1
            epc_h = np.tile(epc, reps)
        else:
            epc_h = epc
        W = np.lib.stride_tricks.sliding_window_view(epc_h, BUF)
        V = W @ M_TRANS.T
        Y = (V * V) @ M_SQ.T
        nY = Y.shape[0]
        mu_y = Y.mean(axis=0)
        sig_y = _long_run_sd(Y, LR_BAND)
        st.mu_y = mu_y
        st.inv_sig_y = 1.0 / sig_y
        floor = 1e-6 * np.maximum(mu_y, 1e-12)
        st.floor = floor
        st.lhist = np.log(mu_y + floor)
        st.mu_tab, st.inv_sig_tab = self._replay_null(Y, nY, mu_y, st.inv_sig_y, st.lhist, floor)
        sorted_abs = np.sort(abs_eps)
        st.sorted_abs = sorted_abs
        sorted_eps = np.sort(eps)
        st.sorted_eps = sorted_eps
        st.n_hist_eps = float(nE)
        st.med = float(np.median(eps))
        st.mean_epc = float(epc.mean())
        st.sd_epc = max(float(epc.std()), SIG_FLOOR)
        sq = epc * epc
        st.mean_sq = max(float(sq.mean()), SIG_FLOOR)
        sq_raw = (eps * eps)[:, None]
        st.sd_sq_lr = max(float(_long_run_sd(sq_raw, np.array([2.0 * L_STFT]))[0]), SIG_FLOOR)
        rank = (
            np.searchsorted(sorted_abs, abs_eps, side="left") + np.searchsorted(sorted_abs, abs_eps, side="right")
        ) / (2.0 * nE)
        st.mean_rank = float(rank.mean())
        st.sd_rank = max(float(rank.std()), SIG_FLOOR)
        rank_s = (np.searchsorted(sorted_eps, eps, side="left") + np.searchsorted(sorted_eps, eps, side="right")) / (
            2.0 * nE
        ) - 0.5
        st.sd_wil = max(float(rank_s.std()), SIG_FLOOR)
        thr = float(np.quantile(abs_eps, TAIL_Q))
        st.tail_thr = thr
        p_tail = float(np.mean(abs_eps > thr))
        st.p_tail = p_tail
        st.sd_tail = max(math.sqrt(p_tail * (1.0 - p_tail)), SIG_FLOOR)
        s = np.sign(eps - st.med)
        sp = s[1:] * s[:-1]
        st.mean_sp = float(sp.mean())
        st.sd_sp = max(float(sp.std()), SIG_FLOOR)
        st.prev_sign = float(s[-1])
        q75 = float(np.quantile(abs_eps, 0.75))
        tailratio = float(np.quantile(abs_eps, 0.95)) / max(q75, 1e-12)
        st.cov = np.array(
            [
                0.0,
                float(np.searchsorted(NHIST_EDGES, float(n), side="right")),
                float(np.searchsorted(TAILRATIO_EDGES, tailratio, side="right")),
                float(p),
                float(np.searchsorted(INNOVVAR_EDGES, sd_e * sd_e, side="right")),
            ]
        )
        st.cap = 256
        st.A_f = np.zeros(N_CH)
        st.A_s = np.zeros(N_CH)
        st.P = np.zeros((st.cap + 1, N_CH))
        st.S7 = np.zeros((st.cap + 1, N_SEC))
        st.raw = np.zeros(N_SPEC)
        st.feat = np.zeros(N_FEAT)
        st.LE = st.raw[: 3 * N_CH].reshape(3, N_CH)
        st.ceps_work = st.raw[_I_CEPS].reshape(3, N_CEPS)
        st.E3 = np.zeros((3, N_CH))
        return st

    @staticmethod
    def replay_stats(P, A_f, A_s, s, m, mu_y, inv_sig_y, lhist, floor):
        """Spectral statistics of age ``m`` for online streams starting after each history row in ``s``."""
        ns = s.shape[0]
        stat = np.empty((ns, N_SPEC))
        Ef = (A_f[s + m] - _LAMPOW_F[m] * A_f[s]) * _INV_EWW_F[m]
        Es = (A_s[s + m] - _LAMPOW_S[m] * A_s[s]) * _INV_EWW_S[m]
        Ec = (P[s + m + 1] - P[s + 1]) / m
        LE = np.empty((ns, 3, N_CH))
        np.log(np.maximum(Ef, 0.0) + floor, out=LE[:, 0])
        np.log(np.maximum(Es, 0.0) + floor, out=LE[:, 1])
        np.log(np.maximum(Ec, 0.0) + floor, out=LE[:, 2])
        stat[:, _I_LEF] = LE[:, 0]
        stat[:, _I_LES] = LE[:, 1]
        stat[:, _I_LEC] = LE[:, 2]
        ws = _WIN[m]
        end = s + m + 1
        sufs = (P[end][None, :, :] - P[end[None, :] - ws[:, None]]) * _WIN_INV[m][:, :, None]
        z = (sufs - mu_y) * (_WIN_SQRT[m][:, :, None] * inv_sig_y)
        j = np.argmax(np.abs(z), axis=0)
        stat[:, _I_SCAN] = np.take_along_axis(z, j[None, :, :], axis=0)[0]
        cep = np.empty((ns * 3, N_CEPS))
        _cepstral(LE.reshape(ns * 3, N_CH), lhist, cep)
        stat[:, _I_CEPS] = cep.reshape(ns, 3 * N_CEPS)
        return stat

    @staticmethod
    def historical_arrays(Y):
        """Prefix sums and fast/slow exponential sums of the historical channel energies."""
        P = np.zeros((Y.shape[0] + 1, N_CH))
        np.cumsum(Y, axis=0, out=P[1:])
        return P, _ew_full(Y, LAM_FAST), _ew_full(Y, LAM_SLOW)

    @classmethod
    def _replay_null(cls, Y, nY, mu_y, inv_sig_y, lhist, floor):
        """Null mean and 1/sd of every spectral statistic per online age, interpolated in log age."""
        P, A_f, A_s = cls.historical_arrays(Y)
        n_grid = len(AGE_GRID)
        mu_g = np.zeros((n_grid, N_SPEC))
        sd_g = np.zeros((n_grid, N_SPEC))
        nv = 0
        for gi, m in enumerate(AGE_GRID):
            npos = nY - m
            if npos < MIN_STARTS:
                break
            nv = gi + 1
            s = np.unique(np.round(np.linspace(0, npos - 1, min(N_STARTS, npos))).astype(np.int64))
            stat = cls.replay_stats(P, A_f, A_s, s, m, mu_y, inv_sig_y, lhist, floor)
            mu_g[gi] = stat.mean(axis=0)
            sd_g[gi] = stat.std(axis=0)
        if nv == 0:
            raise RuntimeError("historical too short for replay null")
        ages = np.arange(1, M_DENSE + 1, dtype=float)
        la = np.log(ages)
        gl = _LOG_AGE[:nv]
        if nv >= 2:
            idx = np.clip(np.searchsorted(gl, la, side="right") - 1, 0, nv - 2)
            frac = np.clip((la - gl[idx]) / (gl[idx + 1] - gl[idx]), 0.0, 1.0)[:, None]
            mu_i = mu_g[idx] * (1.0 - frac) + mu_g[idx + 1] * frac
            sd_i = sd_g[idx] * (1.0 - frac) + sd_g[idx + 1] * frac
        else:
            mu_i = np.repeat(mu_g[:1], M_DENSE, axis=0)
            sd_i = np.repeat(sd_g[:1], M_DENSE, axis=0)
        m_last = AGE_GRID[nv - 1]
        beyond = ages > m_last
        nb = int(beyond.sum())
        if nb:
            scale = np.sqrt(m_last / ages[beyond])[:, None]
            mu_b = np.repeat(mu_g[nv - 1 : nv], nb, axis=0)
            sd_b = np.repeat(sd_g[nv - 1 : nv], nb, axis=0)
            mu_b[:, _MU_SQRT_SCALE] *= scale
            sd_b[:, _SIG_SQRT_SCALE] *= scale
            mu_i[beyond] = mu_b
            sd_i[beyond] = sd_b
        mu_tab = np.empty((M_DENSE + 1, N_SPEC))
        sd_tab = np.empty((M_DENSE + 1, N_SPEC))
        mu_tab[1:] = mu_i
        sd_tab[1:] = sd_i
        mu_tab[0] = mu_i[0]
        sd_tab[0] = sd_i[0]
        sd_tab = np.maximum(sd_tab, SIG_FLOOR + 1e-6 * np.abs(mu_tab))
        return mu_tab, 1.0 / sd_tab

    def step_features(self, st, x):
        """Consume one online point; return the 96 features (a buffer reused at the next call)."""
        xz = (float(x) - st.mu_x) * st.inv_sd_x
        lags = st.lags
        e = xz - float(np.dot(st.a_pad, lags))
        lags[1:] = lags[:-1]
        lags[0] = xz
        eps = e * st.inv_sd_e
        c = st.clip
        epc = c if eps > c else (-c if eps < -c else eps)
        m = st.m + 1
        st.m = m
        if m > st.cap:
            self._grow(st)
        pos = (m - 1) & (BUF - 1)
        buf = st.buf
        buf[pos] = epc
        buf[pos + BUF] = epc
        v = M_TRANS @ buf[pos + 1 : pos + BUF + 1]
        y = M_SQ @ (v * v)
        A_f = st.A_f
        A_f *= LAM_FAST
        A_f += y
        A_s = st.A_s
        A_s *= LAM_SLOW
        A_s += y
        P = st.P
        Pm = P[m]
        np.add(P[m - 1], y, out=Pm)
        if m <= M_DENSE:
            mm = m
            ws = _WIN[m]
            winv = _WIN_INV[m]
            wsq = _WIN_SQRT[m]
        else:
            mm = M_DENSE
            ws = _windows_for(m)
            winv = (1.0 / ws.astype(float))[:, None]
            wsq = np.sqrt(ws.astype(float))[:, None]
        LE = st.LE
        E3 = st.E3
        np.multiply(A_f, _INV_EWW_F[mm], out=E3[0])
        np.multiply(A_s, _INV_EWW_S[mm], out=E3[1])
        np.multiply(Pm, 1.0 / m, out=E3[2])
        E3 += st.floor
        np.log(E3, out=LE)
        raw = st.raw
        suf = (Pm - P[m - ws]) * winv
        zw = (suf - st.mu_y) * (wsq * st.inv_sig_y)
        j = np.argmax(np.abs(zw), axis=0)
        raw[_I_SCAN] = zw[j, _AR_CH]
        _cepstral(LE, st.lhist, st.ceps_work)
        feat = st.feat
        np.multiply(raw - st.mu_tab[mm], st.inv_sig_tab[mm], out=feat[:N_SPEC])
        self._secondary(st, m, mm, eps, epc, feat)
        cov = st.cov
        cov[0] = float(m - 1)
        feat[N_SPEC + N_SEC :] = cov
        return feat

    @staticmethod
    def _grow(st):
        """Double the capacity of the prefix-sum buffers."""
        cap = st.cap * 2
        P = np.zeros((cap + 1, N_CH))
        P[: st.cap + 1] = st.P
        S7 = np.zeros((cap + 1, N_SEC))
        S7[: st.cap + 1] = st.S7
        st.P, st.S7, st.cap = P, S7, cap

    @staticmethod
    def _secondary(st, m, mm, eps, epc, feat):
        """The 7 secondary statistics: mean and variance GLR, CUSUM of squares, rank bridges and CUSUMs."""
        abs_e = abs(eps)
        sa = st.sorted_abs
        nE = st.n_hist_eps
        lo = int(np.searchsorted(sa, abs_e, side="left"))
        hi = int(np.searchsorted(sa, abs_e, side="right"))
        rank_abs = (lo + hi) / (2.0 * nE)
        se = st.sorted_eps
        lo2 = int(np.searchsorted(se, eps, side="left"))
        hi2 = int(np.searchsorted(se, eps, side="right"))
        rank_sgn = (lo2 + hi2) / (2.0 * nE) - 0.5
        sgn = 1.0 if eps > st.med else (-1.0 if eps < st.med else 0.0)
        sp = sgn * st.prev_sign
        st.prev_sign = sgn
        tail = 1.0 if abs_e > st.tail_thr else 0.0
        S7 = st.S7
        row = S7[m]
        prev = S7[m - 1]
        row[0] = prev[0] + (epc - st.mean_epc)
        row[1] = prev[1] + epc * epc
        row[2] = prev[2] + eps * eps
        row[3] = prev[3] + (rank_abs - st.mean_rank)
        row[4] = prev[4] + rank_sgn
        row[5] = prev[5] + (tail - st.p_tail)
        row[6] = prev[6] + (sp - st.mean_sp)
        if m <= M_DENSE:
            inv_sqrt_desc = _INV_SQRT_DESC[m]
        else:
            inv_sqrt_desc = 1.0 / np.sqrt(np.arange(m, 0, -1, dtype=float))
        diff = row - S7[:m]
        glr = diff * inv_sqrt_desc[:, None]
        sub = glr[:, (0, 4, 5, 6)]
        jj = np.argmax(np.abs(sub), axis=0)
        vals = sub[jj, _AR4]
        base = N_SPEC
        feat[base + 0] = vals[0] / st.sd_epc
        feat[base + 4] = vals[1] / st.sd_wil
        feat[base + 5] = vals[2] / st.sd_tail
        feat[base + 6] = vals[3] / st.sd_sp
        inv_len = inv_sqrt_desc * inv_sqrt_desc
        rho = np.maximum(diff[:, 1] * inv_len / st.mean_sq, 1e-12)
        lr = 0.5 * (rho - 1.0 - np.log(rho)) / inv_len
        kk = int(np.argmax(lr))
        feat[base + 1] = math.copysign(math.sqrt(2.0 * max(float(lr[kk]), 0.0)), float(rho[kk]) - 1.0)
        if m >= 2:
            if m <= M_DENSE:
                kom = _KOVERM[m]
                bw = _BOUNDARY[m]
            else:
                kom = np.arange(1, m, dtype=float) / m
                bw = 1.0 / np.sqrt(np.maximum(kom * (1.0 - kom), BRIDGE_DELTA * (1.0 - BRIDGE_DELTA)))
            br2 = S7[1:m, 2] - kom * row[2]
            k2 = int(np.argmax(np.abs(br2)))
            feat[base + 2] = (
                float(br2[k2]) / (_SQRT[m] * st.sd_sq_lr)
                if m <= M_DENSE + 1
                else float(br2[k2]) / (math.sqrt(m) * st.sd_sq_lr)
            )
            br3 = (S7[1:m, 3] - kom * row[3]) * bw
            k3 = int(np.argmax(np.abs(br3)))
            feat[base + 3] = float(br3[k3]) / (math.sqrt(m) * st.sd_rank)
        else:
            feat[base + 2] = (eps * eps - st.mean_sq) / st.sd_sq_lr
            feat[base + 3] = (rank_abs - st.mean_rank) / st.sd_rank
