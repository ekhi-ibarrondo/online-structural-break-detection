"""Per-series online feature engine of slow_and_fast (pure numpy + stdlib).

Design:
  * z-score by historical mean/sd; AR(p) by Yule-Walker (BIC, p<=10) fitted on the
    historical only and frozen; online innovations filtered point by point.
  * Innovations standardized by the historical innovation sd and winsorized at the
    historical 0.995 quantile of |innovation| (robust spectral estimation).
  * PRIMARY spectral-structure family on the innovations (84 columns):
      (i)  Hann-windowed sliding STFT (L=16) band energies at 9 bins (k*pi/8),
           aggregated with exponential forgetting (lam=0.90, 0.98), cumulative from the
           online start, and a multiscale suffix-window scan (geometric grid) -- each
           compared with the historical spectrum of the same innovations;
      (ii) undecimated Haar wavelet energy at 6 dyadic scales (half-widths 1..32), same
           aggregators, compared with the historical counterpart;
      (iii) cepstral (log-spectrum) distances (full and gain-free shape) between the
           running online spectral estimate and the historical one, plus the first 4
           cepstral coefficient deviations; also the Haar log-energy distances.
    Every quantity is standardized against the OWN-SERIES historical null: the exact
    same online statistic is replayed from many start positions inside the historical
    innovations (age-indexed null mean/sd; interpolated over a fixed age grid).
  * SECONDARY reference-bank columns (7): FOCuS-type mean GLR, FOCuS-type variance GLR,
    Sanso CUSUM-of-squares (long-run fourth-moment normalizer), Brownian-bridge on the midrank of
    |innovation|, Wilcoxon rank CUSUM, peaks-over-threshold exceedance CUSUM, sign-product CUSUM.
  * Covariates (5): step t, n_hist bucket (4), tail-ratio bucket (4), AR order, innovation-variance bucket (4).
Nothing here depends on other series; every step is deterministic.
"""
from __future__ import annotations

import math

import numpy as np

# ----------------------------------------------------------------------------- constants
P_MAX = 10                       # AR order upper bound (BIC selection over 0..P_MAX)
L_STFT = 16                      # Hann window length of the sliding STFT
N_BINS = 9                       # bins k = 0..8 at omega_k = k*pi/8
HAAR_H = (1, 2, 4, 8, 16, 32)    # Haar half-widths (window 2h)
N_HAAR = len(HAAR_H)
BUF = 64                         # innovation buffer length (= 2 * max half-width)
N_CH = N_BINS + N_HAAR           # 15 per-point energy channels
LAM_FAST = 0.90
LAM_SLOW = 0.98
SCAN_W = (1, 2, 4, 8, 16, 32, 64, 128, 256, 512)   # suffix windows (+ the full online prefix)
AGE_GRID = (1, 2, 3, 4, 6, 8, 12, 16, 24, 32, 48, 64, 96, 128, 192, 256, 384, 512, 768, 1024)
N_STARTS = 96                    # replay start positions per age (subsampled, evenly spaced)
MIN_STARTS = 32                  # minimum number of replay positions for an age to be valid
M_DENSE = 1024                   # dense null tables cover online ages 1..M_DENSE
CLIP_Q = 0.995                   # winsorization quantile of |innovation| (historical)
TAIL_Q = 0.90                    # POT exceedance quantile of |innovation| (historical)
BRIDGE_DELTA = 0.05              # trimming of the bridge boundary weight sqrt(u(1-u))
# coarse per-series covariate buckets (fixed edges; avoid series-identifying continuous constants)
NHIST_EDGES = np.array([2000.0, 3000.0, 4000.0])
TAILRATIO_EDGES = np.array([2.4, 2.75, 3.1])      # q95/q75 of |innovation|; Gaussian ~2.9
INNOVVAR_EDGES = np.array([0.25, 0.5, 0.8])       # innovation variance relative to unit series variance
N_CEPS_COEF = 4                  # cepstral coefficients c_0..c_3 used as features
N_CEPS = N_CEPS_COEF + 2 + 2     # per aggregator: d_full, d_shape, c_0..c_3, haar d_full, d_shape
N_SPEC = 4 * N_CH + 3 * N_CEPS   # 60 + 24 = 84 spectral columns
N_SEC = 7
N_COV = 5
N_FEAT = N_SPEC + N_SEC + N_COV  # 96
SIG_FLOOR = 1e-9


def feature_names():
    names = []
    ch = [f"stft{k}" for k in range(N_BINS)] + [f"haar{h}" for h in HAAR_H]
    for agg in ("ewf", "ews", "cum"):
        names += [f"logE_{agg}_{c}" for c in ch]
    names += [f"scan_{c}" for c in ch]
    for agg in ("ewf", "ews", "cum"):
        names += [f"ceps_{agg}_{q}" for q in ("dfull", "dshape", "c0", "c1", "c2", "c3", "hdfull", "hdshape")]
    names += ["sec_meanGLR", "sec_varGLR", "sec_sanso", "sec_rankbridge", "sec_wilcoxon", "sec_tailPOT",
              "sec_signprod"]
    names += ["cov_t", "cov_nhist_bucket", "cov_tailratio_bucket", "cov_arorder", "cov_innovvar_bucket"]
    assert len(names) == N_FEAT
    return names


# ---- per-point energy transform matrices (constant) --------------------------------------
def _build_matrices():
    """M: (22, BUF) maps the chronological 64-buffer (oldest first) to
    [cos parts of 9 bins (9), sin parts of bins 1..7 (7), Haar coefficients (6)].
    MSQ: (15, 22) sums squares into the 15 energy channels."""
    M = np.zeros((N_BINS + 7 + N_HAAR, BUF))
    j = np.arange(L_STFT)                       # j = 0 -> oldest inside the STFT window
    w = 0.5 * (1.0 - np.cos(2.0 * np.pi * (j + 0.5) / L_STFT))   # Hann (periodic-symmetric)
    norm = 1.0 / math.sqrt(float(np.sum(w * w)))
    lag = (L_STFT - 1 - j).astype(float)        # lag from the newest sample
    for k in range(N_BINS):
        om = k * np.pi / (N_BINS - 1)
        M[k, BUF - L_STFT:] = w * np.cos(om * lag) * norm
        if 1 <= k <= 7:
            M[N_BINS + k - 1, BUF - L_STFT:] = -w * np.sin(om * lag) * norm
    for i, h in enumerate(HAAR_H):
        row = N_BINS + 7 + i
        c = math.sqrt(h / 2.0) / h                # unit variance for white unit-variance input
        M[row, BUF - h:] = c
        M[row, BUF - 2 * h:BUF - h] = -c
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

# ---- global per-age tables ----------------------------------------------------------------
_SQRT = np.sqrt(np.arange(0, M_DENSE + 2, dtype=float))
_AR_CH = np.arange(N_CH)
_AR4 = np.arange(4)


def _windows_for(m):
    ws = [w for w in SCAN_W if w < m]
    ws.append(m)
    return np.array(ws, dtype=np.int64)


_WIN = [None] + [_windows_for(m) for m in range(1, M_DENSE + 1)]
_WIN_SQRT = [None] + [np.sqrt(_WIN[m].astype(float))[:, None] for m in range(1, M_DENSE + 1)]
_WIN_INV = [None] + [(1.0 / _WIN[m].astype(float))[:, None] for m in range(1, M_DENSE + 1)]
_LOG_AGE = np.log(np.array(AGE_GRID, dtype=float))
_INV_SQRT_DESC = [None] + [1.0 / _SQRT[m:0:-1] for m in range(1, M_DENSE + 1)]  # 1/sqrt(m-k), k=0..m-1
_BOUNDARY = [None, None]
_KOVERM = [None, None]
for _m in range(2, M_DENSE + 1):
    _u = np.arange(1, _m, dtype=float) / _m
    _KOVERM.append(_u)
    _BOUNDARY.append(1.0 / np.sqrt(np.maximum(_u * (1.0 - _u), BRIDGE_DELTA * (1.0 - BRIDGE_DELTA))))
_EWW_F = (1.0 - LAM_FAST ** np.arange(0, M_DENSE + 2, dtype=float)) / (1.0 - LAM_FAST)   # sum_{j<m} lam^j
_EWW_S = (1.0 - LAM_SLOW ** np.arange(0, M_DENSE + 2, dtype=float)) / (1.0 - LAM_SLOW)
_INV_EWW_F = np.concatenate([[0.0], 1.0 / _EWW_F[1:]])
_INV_EWW_S = np.concatenate([[0.0], 1.0 / _EWW_S[1:]])
_LAMPOW_F = LAM_FAST ** np.arange(0, M_DENSE + 2, dtype=float)
_LAMPOW_S = LAM_SLOW ** np.arange(0, M_DENSE + 2, dtype=float)

# statistic layout inside the 84 spectral raw vector
_I_LEF = slice(0, N_CH)
_I_LES = slice(N_CH, 2 * N_CH)
_I_LEC = slice(2 * N_CH, 3 * N_CH)
_I_SCAN = slice(3 * N_CH, 4 * N_CH)
_I_CEPS = slice(4 * N_CH, 4 * N_CH + 3 * N_CEPS)   # order: fast(8), slow(8), cum(8)
_SIG_SQRT_SCALE = np.zeros(N_SPEC, dtype=bool)    # sigma ~ 1/sqrt(m) for cumulative-type stats
_MU_SQRT_SCALE = np.zeros(N_SPEC, dtype=bool)     # mu ~ 1/sqrt(m) for cumulative-type distances
_SIG_SQRT_SCALE[_I_LEC] = True
_c0 = 4 * N_CH + 2 * N_CEPS
_SIG_SQRT_SCALE[_c0:_c0 + N_CEPS] = True
for _q in (0, 1, 6, 7):
    _MU_SQRT_SCALE[_c0 + _q] = True


# ----------------------------------------------------------------------------- helpers
def fit_ar_yw(x, p_max=P_MAX):
    """Yule-Walker AR fit with BIC order selection over 0..p_max. Returns (a, p, sig2)."""
    n = x.shape[0]
    xc = x - x.mean()
    nfft = 1
    while nfft < 2 * n:
        nfft <<= 1
    f = np.fft.rfft(xc, nfft)
    acov = np.fft.irfft(f * np.conj(f), nfft)[:p_max + 1] / n
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
            k = (acov[p] - float(np.dot(a, acov[p - 1:0:-1]))) / sig2
        k = float(k)
        if abs(k) >= 0.9999:
            break
        a_new = np.empty(p)
        a_new[:p - 1] = a - k * a[::-1]
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
    """Full-history exponentially weighted sums A[i] = sum_{j<=i} lam^(i-j) Y[j] (blockwise stable)."""
    n, c = Y.shape
    A = np.empty_like(Y)
    carry = np.zeros(c)
    pw = lam ** np.arange(1, block + 1, dtype=float)
    ipw = lam ** (-np.arange(1, block + 1, dtype=float))
    for b0 in range(0, n, block):
        Yb = Y[b0:b0 + block]
        nb = Yb.shape[0]
        S = np.cumsum(Yb * ipw[:nb, None], axis=0) * pw[:nb, None]
        S += carry[None, :] * pw[:nb, None]
        A[b0:b0 + nb] = S
        carry = S[-1]
    return A


def _long_run_sd(Y, band):
    """Bartlett long-run sd per column with per-column bandwidth."""
    n, c = Y.shape
    Yc = Y - Y.mean(axis=0)
    nfft = 1
    while nfft < 2 * n:
        nfft <<= 1
    F = np.fft.rfft(Yc, nfft, axis=0)
    ac = np.fft.irfft(F * np.conj(F), nfft, axis=0)[:LR_MAXLAG + 1] / n
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
    """LE: (K, 15) log energies; writes (K, 8) [d_full, d_shape, c0..c3, haar d_full, haar d_shape]."""
    DL = LE - lhist
    DS = DL[:, :N_BINS]
    DH = DL[:, N_BINS:]
    ss = (DS * DS) @ _ONES_S
    ms = DS @ _MEAN_S
    out[:, 0] = np.sqrt(ss)
    out[:, 1] = np.sqrt(np.maximum(ss - N_BINS * (ms * ms), 0.0))
    out[:, 2:2 + N_CEPS_COEF] = DS @ DCT_T
    sh = (DH * DH) @ _ONES_H
    mh = DH @ _MEAN_H
    out[:, 6] = np.sqrt(sh)
    out[:, 7] = np.sqrt(np.maximum(sh - N_HAAR * (mh * mh), 0.0))
    return out


class SeriesState:
    __slots__ = (
        "mu_x", "inv_sd_x", "a_pad", "lags", "inv_sd_e", "clip", "buf",
        "m", "A_f", "A_s", "P", "S7", "mu_y", "inv_sig_y", "lhist", "floor",
        "mu_tab", "inv_sig_tab", "sorted_abs", "sorted_eps", "n_hist_eps", "med",
        "mean_epc", "sd_epc", "mean_sq", "sd_sq_lr", "mean_rank", "sd_rank", "sd_wil",
        "tail_thr", "p_tail", "sd_tail", "mean_sp", "sd_sp", "prev_sign", "cov",
        "raw", "feat", "ceps_work", "LE", "E3", "cap",
    )


class FeatureEngine:
    """start(x_hist) -> state ; step_features(state, x) -> feature vector (N_FEAT,)"""

    # ------------------------------------------------------------------ start
    def start(self, x_hist):
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
        # historical innovations for t = p..n-1
        e = xz[p:].copy()
        for j in range(1, p + 1):
            e -= a[j - 1] * xz[p - j:n - j]
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
        # AR lags: most recent first
        lags = np.zeros(P_MAX)
        k = min(P_MAX, n)
        for j in range(k):
            lags[j] = xz[n - 1 - j]
        st.lags = lags
        # buffer with the last 64 winsorized innovations (chronological)
        buf = np.zeros(2 * BUF)
        tail = epc[-BUF:]
        buf[BUF - tail.shape[0]:BUF] = tail
        buf[2 * BUF - tail.shape[0]:] = tail
        st.buf = buf
        st.m = 0
        # ---- historical per-point channel energies
        nE = epc.shape[0]
        if nE < BUF + MIN_STARTS + 2:   # never with n_hist >= 1000; keeps the code total
            reps = int(math.ceil((BUF + MIN_STARTS + 2) / max(nE, 1))) + 1
            epc_h = np.tile(epc, reps)
        else:
            epc_h = epc
        W = np.lib.stride_tricks.sliding_window_view(epc_h, BUF)      # (nY, 64), oldest first
        V = W @ M_TRANS.T
        Y = (V * V) @ M_SQ.T                                            # (nY, 15)
        nY = Y.shape[0]
        mu_y = Y.mean(axis=0)
        sig_y = _long_run_sd(Y, LR_BAND)
        st.mu_y = mu_y
        st.inv_sig_y = 1.0 / sig_y
        floor = 1e-6 * np.maximum(mu_y, 1e-12)
        st.floor = floor
        st.lhist = np.log(mu_y + floor)
        st.mu_tab, st.inv_sig_tab = self._replay_null(Y, nY, mu_y, st.inv_sig_y, st.lhist, floor)
        # ---- secondary-channel historical constants
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
        rank = (np.searchsorted(sorted_abs, abs_eps, side="left") +
                np.searchsorted(sorted_abs, abs_eps, side="right")) / (2.0 * nE)
        st.mean_rank = float(rank.mean())
        st.sd_rank = max(float(rank.std()), SIG_FLOOR)
        rank_s = (np.searchsorted(sorted_eps, eps, side="left") +
                  np.searchsorted(sorted_eps, eps, side="right")) / (2.0 * nE) - 0.5
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
        st.cov = np.array([0.0,
                           float(np.searchsorted(NHIST_EDGES, float(n), side="right")),
                           float(np.searchsorted(TAILRATIO_EDGES, tailratio, side="right")),
                           float(p),
                           float(np.searchsorted(INNOVVAR_EDGES, sd_e * sd_e, side="right"))])
        st.cap = 256
        st.A_f = np.zeros(N_CH)
        st.A_s = np.zeros(N_CH)
        st.P = np.zeros((st.cap + 1, N_CH))
        st.S7 = np.zeros((st.cap + 1, N_SEC))
        st.raw = np.zeros(N_SPEC)
        st.feat = np.zeros(N_FEAT)
        st.LE = st.raw[:3 * N_CH].reshape(3, N_CH)            # view: [logE_fast; logE_slow; logE_cum]
        st.ceps_work = st.raw[_I_CEPS].reshape(3, N_CEPS)      # view
        st.E3 = np.zeros((3, N_CH))
        return st

    # ------------------------------------------------------------------ replay null
    @staticmethod
    def replay_stats(P, A_f, A_s, s, m, mu_y, inv_sig_y, lhist, floor):
        """Raw spectral statistics (ns, N_SPEC) of an online stream of age m that starts
        right after historical row s (rows s+1..s+m are the 'online' points)."""
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
        # scan over suffix windows, vectorized over (window, start)
        ws = _WIN[m]
        end = s + m + 1
        sufs = (P[end][None, :, :] - P[end[None, :] - ws[:, None]]) * _WIN_INV[m][:, :, None]
        z = (sufs - mu_y) * (_WIN_SQRT[m][:, :, None] * inv_sig_y)
        j = np.argmax(np.abs(z), axis=0)
        stat[:, _I_SCAN] = np.take_along_axis(z, j[None, :, :], axis=0)[0]
        # cepstral
        cep = np.empty((ns * 3, N_CEPS))
        _cepstral(LE.reshape(ns * 3, N_CH), lhist, cep)
        stat[:, _I_CEPS] = cep.reshape(ns, 3 * N_CEPS)
        return stat

    @staticmethod
    def historical_arrays(Y):
        P = np.zeros((Y.shape[0] + 1, N_CH))
        np.cumsum(Y, axis=0, out=P[1:])
        return P, _ew_full(Y, LAM_FAST), _ew_full(Y, LAM_SLOW)

    @classmethod
    def _replay_null(cls, Y, nY, mu_y, inv_sig_y, lhist, floor):
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
            mu_b = np.repeat(mu_g[nv - 1:nv], nb, axis=0)
            sd_b = np.repeat(sd_g[nv - 1:nv], nb, axis=0)
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

    # ------------------------------------------------------------------ step
    def step_features(self, st, x):
        # 1. innovation
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
        # 2. buffer + per-point energies
        pos = (m - 1) & (BUF - 1)
        buf = st.buf
        buf[pos] = epc
        buf[pos + BUF] = epc
        v = M_TRANS @ buf[pos + 1:pos + BUF + 1]
        y = M_SQ @ (v * v)
        # 3. aggregators
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
        np.log(E3, out=LE)                     # writes raw[0:45]
        raw = st.raw
        # 4. scan over suffix windows
        suf = (Pm - P[m - ws]) * winv
        zw = (suf - st.mu_y) * (wsq * st.inv_sig_y)
        j = np.argmax(np.abs(zw), axis=0)
        raw[_I_SCAN] = zw[j, _AR_CH]
        # 5. cepstral block (writes raw[60:84] through the view)
        _cepstral(LE, st.lhist, st.ceps_work)
        # 6. standardize against the own-series replay null
        feat = st.feat
        np.multiply(raw - st.mu_tab[mm], st.inv_sig_tab[mm], out=feat[:N_SPEC])
        # 7. secondary channels
        self._secondary(st, m, mm, eps, epc, feat)
        # 8. covariates
        cov = st.cov
        cov[0] = float(m - 1)
        feat[N_SPEC + N_SEC:] = cov
        return feat

    @staticmethod
    def _grow(st):
        cap = st.cap * 2
        P = np.zeros((cap + 1, N_CH))
        P[:st.cap + 1] = st.P
        S7 = np.zeros((cap + 1, N_SEC))
        S7[:st.cap + 1] = st.S7
        st.P, st.S7, st.cap = P, S7, cap

    @staticmethod
    def _secondary(st, m, mm, eps, epc, feat):
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
        diff = row - S7[:m]                                # k = 0..m-1 : S_m - S_k   (m, 7)
        glr = diff * inv_sqrt_desc[:, None]
        sub = glr[:, (0, 4, 5, 6)]
        jj = np.argmax(np.abs(sub), axis=0)
        vals = sub[jj, _AR4]
        base = N_SPEC
        feat[base + 0] = vals[0] / st.sd_epc
        feat[base + 4] = vals[1] / st.sd_wil
        feat[base + 5] = vals[2] / st.sd_tail
        feat[base + 6] = vals[3] / st.sd_sp
        # variance GLR on winsorized squares (col 1): Gaussian LR over suffixes, signed
        inv_len = inv_sqrt_desc * inv_sqrt_desc            # 1/(m-k)
        rho = np.maximum(diff[:, 1] * inv_len / st.mean_sq, 1e-12)
        lr = 0.5 * (rho - 1.0 - np.log(rho)) / inv_len
        kk = int(np.argmax(lr))
        feat[base + 1] = math.copysign(math.sqrt(2.0 * max(float(lr[kk]), 0.0)), float(rho[kk]) - 1.0)
        # Sanso CUSUM-of-squares (col 2) and midrank bridge (col 3)
        if m >= 2:
            if m <= M_DENSE:
                kom = _KOVERM[m]
                bw = _BOUNDARY[m]
            else:
                kom = np.arange(1, m, dtype=float) / m
                bw = 1.0 / np.sqrt(np.maximum(kom * (1.0 - kom), BRIDGE_DELTA * (1.0 - BRIDGE_DELTA)))
            br2 = S7[1:m, 2] - kom * row[2]
            k2 = int(np.argmax(np.abs(br2)))
            feat[base + 2] = float(br2[k2]) / (_SQRT[m] * st.sd_sq_lr) if m <= M_DENSE + 1 else \
                float(br2[k2]) / (math.sqrt(m) * st.sd_sq_lr)
            br3 = (S7[1:m, 3] - kom * row[3]) * bw
            k3 = int(np.argmax(np.abs(br3)))
            feat[base + 3] = float(br3[k3]) / (math.sqrt(m) * st.sd_rank)
        else:
            feat[base + 2] = (eps * eps - st.mean_sq) / st.sd_sq_lr
            feat[base + 3] = (rank_abs - st.mean_rank) / st.sd_rank


def extract_features(x_hist, x_online, engine=None):
    """Offline convenience: all per-step feature vectors of a series (n_online, N_FEAT)."""
    eng = engine or FeatureEngine()
    st = eng.start(x_hist)
    xo = np.asarray(x_online, dtype=np.float64)
    out = np.empty((xo.shape[0], N_FEAT))
    for i in range(xo.shape[0]):
        out[i] = eng.step_features(st, xo[i])
    return out
