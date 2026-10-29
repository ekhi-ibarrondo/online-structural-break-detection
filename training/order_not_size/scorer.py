"""Scorer of order_not_size: 18 detector columns built on ranks / PITs of the series' own history.

Per series, single thread, causal (each step sees only the history and the
online prefix) and independent of the online length.  ``Scorer(model_dir)``
loads an exported model (model.npz + meta.json); ``start(x_historical)``
returns the per-series state; ``step(state, x_t)`` returns the score in [0, 1]
(high = the break already occurred).  ``Scorer(None)`` is the columns-only
extractor used for training (``step_columns(state, x_t)`` -> the 18 raw
detector columns).

The arithmetic of ``start`` that feeds the GARCH fit and the ``ndtri`` /
``log(h)`` calls runs in portable numba kernels (``kernels.py``), so the
columns are identical across CPUs.
"""
from __future__ import annotations

import bisect
import json
import math
import os

import numpy as np

from . import kernels as _K  # numba pins + eagerly compiled kernels

# ----------------------------------------------------------------------------- constants
WGRID = np.array(
    [1, 2, 3, 4, 5, 6, 7, 8, 10, 12, 16, 20, 24, 32, 40, 48, 64, 80, 96, 128, 160, 192,
     256, 320, 384, 512, 640, 768, 1024], dtype=np.int64)
WGRID_LIST = WGRID.tolist()
SR_THETAS = (float.fromhex("0x1.0000000000000p-2"), float.fromhex("0x1.0000000000000p-1"),
             float.fromhex("0x1.6a09e667f3bcdp-1"), float.fromhex("0x1.6a09e667f3bcdp+0"),
             float.fromhex("0x1.0000000000000p+1"), float.fromhex("0x1.0000000000000p+2"))
T_EDGES = [1, 2, 3, 4, 5, 7, 9, 12, 16, 22, 30, 42, 58, 80, 110, 152, 210, 290, 400, 550, 760]
N_TBUCKETS = len(T_EDGES)
KNOT_LEVELS = np.array(
    [0.001, 0.002, 0.005, 0.01, 0.02, 0.03, 0.05, 0.075, 0.1, 0.15, 0.2, 0.25, 0.3, 0.4,
     0.5, 0.6, 0.7, 0.75, 0.8, 0.85, 0.9, 0.925, 0.95, 0.97, 0.98, 0.99, 0.995, 0.998,
     0.999], dtype=np.float64)
N_KNOTS = len(KNOT_LEVELS)
KAPPA = float.fromhex("0x1.2934f0979a371p+1")
U_CLIP = 0.01
CUSUM_K = 0.25
INNOV_CLIP2 = 16.0
WHITE_LAGS = 4
POT_P = 0.1
CELL_PS = (0.1, 0.5, 0.9)
RIDGE_ALPHA = 1.0
MIN_CELL = 1500
N_COLS = 18
COLUMN_NAMES = [
    "focus_var", "sanso_cusum_sq", "sr_mixture", "focus_mean", "hermite_T2", "pot_T2",
    "rank_cells_chi2", "white_identity", "white_calibrated", "sign_cusum",
    "garch_score_cum", "garch_score_cusum", "garch_vol_level", "focus_garch_resid",
    "ar_score_iid", "ar_score_bartlett", "bridge_sq", "bridge_rank",
]
GROUP_A = list(range(14))
GROUP_B = [14, 15, 16, 17]
PAIRS = [(i, j) for a, i in enumerate(GROUP_A) for j in GROUP_A[a + 1:]] + \
        [(i, j) for a, i in enumerate(GROUP_B) for j in GROUP_B[a + 1:]]
PAIRS_I = np.array([p[0] for p in PAIRS], dtype=np.int64)
PAIRS_J = np.array([p[1] for p in PAIRS], dtype=np.int64)
N_FEATURES = N_COLS + len(PAIRS)  # 115
FEATURE_NAMES = list(COLUMN_NAMES) + [f"{COLUMN_NAMES[i]}*{COLUMN_NAMES[j]}" for i, j in PAIRS]
LOG6 = float.fromhex("0x1.cab0bfa2a2002p+0")
SQRT2 = math.sqrt(2.0)
SQRT2PI = math.sqrt(2.0 * math.pi)
NEG_INF = -math.inf

# ----------------------------------------------------------------------------- inverse normal CDF
_A = (-3.969683028665376e+01, 2.209460984245205e+02, -2.759285104469687e+02,
      1.383577518672690e+02, -3.066479806614716e+01, 2.506628277459239e+00)
_B = (-5.447609879822406e+01, 1.615858368580409e+02, -1.556989798598866e+02,
      6.680131188771972e+01, -1.328068155288572e+01)
_C = (-7.784894002430293e-03, -3.223964580411365e-01, -2.400758277161838e+00,
      -2.549732539343734e+00, 4.374664141464968e+00, 2.938163982698783e+00)
_D = (7.784695709041462e-03, 3.224671290700398e-01, 2.445134137142996e+00,
      3.754408661907416e+00)
_P_LOW = 0.02425


def ndtri(p: float) -> float:
    """Phi^{-1}(p): Acklam's rational approximation + one Halley refinement (fdlibm erfc)."""
    return _K.ndtri_p(float(p))


def ndtri_array(p: np.ndarray) -> np.ndarray:
    return _K.ndtri_array_p(np.ascontiguousarray(p, dtype=np.float64))


# ----------------------------------------------------------------------------- historical estimators
def bartlett_bandwidth(n_hist: int) -> int:
    return int(math.floor(4.0 * (n_hist / 100.0) ** (2.0 / 9.0)))


def bartlett_cov(Y: np.ndarray, L: int) -> np.ndarray:
    """Bartlett long-run covariance of the rows of Y (N, d); returns (d, d)."""
    if Y.ndim == 1:
        Y = Y[:, None]
    N = Y.shape[0]
    if N < 2:
        return np.eye(Y.shape[1])
    Yc = Y - Y.mean(axis=0)
    Om = Yc.T @ Yc / N
    for k in range(1, L + 1):
        if k >= N:
            break
        G = Yc[k:].T @ Yc[:-k] / N
        Om = Om + (1.0 - k / (L + 1.0)) * (G + G.T)
    return Om


def sample_cov(Y: np.ndarray) -> np.ndarray:
    if Y.ndim == 1:
        Y = Y[:, None]
    N = Y.shape[0]
    if N < 2:
        return np.eye(Y.shape[1])
    Yc = Y - Y.mean(axis=0)
    return Yc.T @ Yc / N


def reg_inv(S: np.ndarray) -> np.ndarray:
    S = np.atleast_2d(np.asarray(S, dtype=np.float64))
    d = S.shape[0]
    Sr = S + (1e-6 * float(np.trace(S)) + 1e-12) * np.eye(d)
    return np.linalg.inv(Sr)


def lrv_scalar(y: np.ndarray, L: int, floor: float = 1e-3) -> float:
    return max(float(bartlett_cov(y, L)[0, 0]), floor)


# ----------------------------------------------------------------------------- FOCuS envelopes
def _push_lower(K: list, V: list, k: int, v: float) -> None:
    while len(K) >= 2:
        k1, v1, k0, v0 = K[-1], V[-1], K[-2], V[-2]
        if (k1 - k0) * (v - v0) - (v1 - v0) * (k - k0) <= 0.0:
            K.pop()
            V.pop()
        else:
            break
    K.append(k)
    V.append(v)


def _push_upper(K: list, V: list, k: int, v: float) -> None:
    while len(K) >= 2:
        k1, v1, k0, v0 = K[-1], V[-1], K[-2], V[-2]
        if (k1 - k0) * (v - v0) - (v1 - v0) * (k - k0) >= 0.0:
            K.pop()
            V.pop()
        else:
            break
    K.append(k)
    V.append(v)


class _Focus:
    """Two one-sided envelopes (lower hull: increase / positive; upper hull: decrease / negative)."""

    __slots__ = ("LK", "LV", "UK", "UV", "C", "m", "n_hist", "variance")

    def __init__(self, n_hist: int, variance: bool):
        self.LK, self.LV, self.UK, self.UV = [0], [0.0], [0], [0.0]
        self.C = 0.0
        self.m = 0
        self.n_hist = n_hist
        self.variance = variance

    def update(self, x: float) -> float:
        """Add x, return the unscaled FOCuS statistic (max over both envelopes)."""
        self.m += 1
        m = self.m
        Cm = self.C + x
        self.C = Cm
        inv_n = 1.0 / self.n_hist
        best = 0.0
        if self.variance:
            for k, ck in zip(self.LK, self.LV):
                nk = m - k
                r = (Cm - ck) / nk
                if r > 1.0:
                    v = nk * (r - 1.0 - math.log(r)) / (1.0 + nk * inv_n)
                    if v > best:
                        best = v
            for k, ck in zip(self.UK, self.UV):
                nk = m - k
                r = (Cm - ck) / nk
                if r < 1.0:
                    if r < 1e-8:
                        r = 1e-8
                    v = nk * (r - 1.0 - math.log(r)) / (1.0 + nk * inv_n)
                    if v > best:
                        best = v
        else:
            for k, ck in zip(self.LK, self.LV):
                S = Cm - ck
                if S > 0.0:
                    nk = m - k
                    v = S * S / (nk * (1.0 + nk * inv_n))
                    if v > best:
                        best = v
            for k, ck in zip(self.UK, self.UV):
                S = Cm - ck
                if S < 0.0:
                    nk = m - k
                    v = S * S / (nk * (1.0 + nk * inv_n))
                    if v > best:
                        best = v
        _push_lower(self.LK, self.LV, m, Cm)
        _push_upper(self.UK, self.UV, m, Cm)
        return best


# ----------------------------------------------------------------------------- conditional PIT
def pit_transform(v: np.ndarray, K: np.ndarray, s_lo: np.ndarray, s_hi: np.ndarray) -> np.ndarray:
    """Monotone CDF map of values v against knot rows K (broadcastable to (len(v), 29)).

    Linear CDF interpolation between knots, plateau mid-probabilities for ties,
    Pareto-type tails beyond the outer knots. Vectorized; identical code for
    training (K = one row) and inference (K = one row per column).
    """
    v = np.asarray(v, dtype=np.float64)
    vv = v[:, None]
    lo = np.sum(K < vv, axis=1)
    hi = np.sum(K <= vv, axis=1)
    lev = KNOT_LEVELS
    lo_c = np.clip(lo, 1, N_KNOTS - 1)
    q_lo = np.take_along_axis(K, (lo_c - 1)[:, None], axis=1)[:, 0]
    q_hi = np.take_along_axis(K, lo_c[:, None], axis=1)[:, 0]
    p_lo = lev[lo_c - 1]
    p_hi = lev[lo_c]
    den = q_hi - q_lo
    den = np.where(den > 0.0, den, 1.0)
    u_mid = p_lo + (p_hi - p_lo) * (v - q_lo) / den
    hi_c = np.clip(hi, 1, N_KNOTS)
    u_tie = 0.5 * (lev[np.clip(lo, 0, N_KNOTS - 1)] + lev[hi_c - 1])
    q_first = K[:, 0] if K.shape[0] > 1 else np.broadcast_to(K[:, 0], v.shape)
    q_last = K[:, -1] if K.shape[0] > 1 else np.broadcast_to(K[:, -1], v.shape)
    # tails (the base is clipped to >= 1 only where the tail branch is not selected)
    u_low = lev[0] * np.power(np.maximum(1.0 + (q_first - v) / s_lo, 1.0), -KAPPA)
    u_high = 1.0 - (1.0 - lev[-1]) * np.power(np.maximum(1.0 + (v - q_last) / s_hi, 1.0), -KAPPA)
    u = np.where(hi > lo, u_tie, np.where(lo == 0, u_low, np.where(lo == N_KNOTS, u_high, u_mid)))
    return u


def tail_scales(K: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """s_lo = q005 - q001, s_hi = q999 - q995 (floored) for knot rows K (..., 29)."""
    s_lo = np.maximum(K[..., 2] - K[..., 0], 1e-12)
    s_hi = np.maximum(K[..., 28] - K[..., 26], 1e-12)
    return s_lo, s_hi


def features_from_u(u: np.ndarray) -> np.ndarray:
    """u: (N, 18) -> features (N, 115): logits + within-group products."""
    uc = np.clip(u, U_CLIP, 1.0 - U_CLIP)
    lg = np.log(uc / (1.0 - uc))
    return np.concatenate([lg, lg[:, PAIRS_I] * lg[:, PAIRS_J]], axis=1)


def score_map(s: np.ndarray | float):
    return 1.0 / (1.0 + np.exp(-4.0 * (s - 0.5)))


def t_bucket(m: int) -> int:
    return bisect.bisect_right(T_EDGES, m) - 1


# ----------------------------------------------------------------------------- state
class _State:
    __slots__ = (
        "n", "xs", "mean_g", "mu0", "s1", "omega4", "lrv_g", "sr_mu", "sr_sd", "sr_l",
        "M_clip", "tau_pot", "p_pot", "lrv_pot", "ebar_pot", "ve_pot", "Nhx",
        "phi", "p", "sig2_eps", "ylag", "elag", "S_white", "mu_s", "Sinv_s",
        "sp_mean", "sp_sd", "g_prev", "cusum_p", "cusum_m",
        "g_omega", "g_a", "g_b", "g_s2", "g_floor", "h_prev", "e2_prev", "d_prev",
        "mu_G", "SinvG", "S_G", "v_mean", "lrv_v", "focus_var", "focus_mean", "focus_e",
        "mu0e", "s_e", "mu_A", "eps_sorted", "abs_eps_sorted", "Ne",
        "ta_mean", "ta_sd", "tb_mean", "tb_sd", "G2", "P", "cap", "m", "B", "Gm",
        "muH", "pC", "D", "iA", "tb_cur", "K_cur", "slo_cur", "shi_cur", "hb", "cols",
    )


class Scorer:
    """`Scorer(artifacts_dir)` loads one exported fold model; `Scorer(None)` = columns only."""

    def __init__(self, artifacts_dir: str | os.PathLike | None = None):
        self.model = None
        if artifacts_dir is not None:
            d = os.fspath(artifacts_dir)
            path = os.path.join(d, "model.npz")
            if not os.path.exists(path):
                raise FileNotFoundError(
                    f"{path} not found: pass one exported fold-model directory "
                    "(artifacts/fold_models/s{seed}f{fold})")
            z = np.load(path)
            self.knots = np.ascontiguousarray(z["knots"], dtype=np.float64)  # (3, 21, 18, 29)
            self.nh_edges = z["nh_edges"].astype(np.float64)
            self.coef = np.ascontiguousarray(z["coef"], dtype=np.float64)
            self.intercept = float(z["intercept"])
            if not np.array_equal(z["pairs_i"], PAIRS_I) or not np.array_equal(z["pairs_j"], PAIRS_J):
                raise ValueError("feature pair order mismatch between model and code")
            self.slo_all, self.shi_all = tail_scales(self.knots)
            meta_path = os.path.join(d, "meta.json")
            self.meta = json.load(open(meta_path)) if os.path.exists(meta_path) else {}
            self.model = path

    # ------------------------------------------------------------------ start
    def start(self, x_historical: np.ndarray) -> _State:
        x = np.asarray(x_historical, dtype=np.float64).ravel()
        n = int(x.shape[0])
        st = _State()
        st.n = n
        L = bartlett_bandwidth(n)
        xs = np.sort(x)
        st.xs = xs.tolist()
        # historical self-PIT (midrank) and normal scores
        c_lt = np.searchsorted(xs, x, side="left")
        c_eq = np.searchsorted(xs, x, side="right") - c_lt
        u_h = (c_lt + 0.5 * c_eq) / n
        g = ndtri_array(u_h)
        st.mean_g = float(_K.mean_p(g))
        g2 = g * g
        mu0 = float(g2.mean())
        st.mu0 = mu0
        x2 = g2 / mu0
        st.s1 = max(float(x2.var()), 0.05)
        st.omega4 = lrv_scalar(x2, L)
        st.lrv_g = lrv_scalar(g, L)
        # SR mixture null run over the historical
        st.sr_l = [NEG_INF] * 6
        sr_mu, sr_sd = 0.0, 1.0
        if n > 21:
            ls = [NEG_INF] * 6
            raws = []
            thetas = SR_THETAS
            for t in range(n):
                xt = x2[t]
                tot = NEG_INF
                for i in range(6):
                    th = thetas[i]
                    lp = ls[i]
                    if lp > 0.0:
                        la = lp + math.log1p(math.exp(-lp))
                    else:
                        la = math.log1p(math.exp(lp))
                    ln = la - 0.5 * math.log(th) - 0.5 * xt * (1.0 / th - 1.0)
                    ls[i] = ln
                    if ln > tot:
                        tot = ln + math.log1p(math.exp(tot - ln)) if tot > NEG_INF else ln
                    else:
                        tot = tot + math.log1p(math.exp(ln - tot))
                if t + 1 > 20:
                    raws.append(tot - LOG6 - math.log(t + 1))
            raws = np.array(raws)
            sr_mu = float(raws.mean())
            sr_sd = max(float(raws.std()), 1e-3)
        st.sr_mu, st.sr_sd = sr_mu, sr_sd
        # Hermite
        M = float(np.max(np.abs(g)))
        st.M_clip = M
        gc = np.clip(g, -M, M)
        H = np.stack([gc ** 3 - 3.0 * gc, gc ** 4 - 6.0 * gc * gc + 3.0], axis=1)
        st.muH = H.mean(axis=0)
        SinvH = reg_inv(bartlett_cov(H, L))
        # POT
        ag = np.abs(g)
        tau = float(np.quantile(ag, 0.9))
        st.tau_pot = tau
        I = (ag > tau).astype(np.float64)
        E = np.maximum(ag - tau, 0.0)
        st.p_pot = float(I.mean())
        st.lrv_pot = lrv_scalar(I, L)
        exc = E[I > 0.0]
        st.Nhx = max(int(exc.shape[0]), 1)
        st.ebar_pot = float(exc.mean()) if exc.shape[0] > 0 else 0.0
        st.ve_pot = max(float(exc.var()), 1e-3) if exc.shape[0] > 1 else 1e-3
        # rank cells
        Ic = np.stack([(u_h < 0.1), (u_h < 0.5), (u_h > 0.9)], axis=1).astype(np.float64)
        st.pC = Ic.mean(axis=0)
        SinvC = reg_inv(bartlett_cov(Ic, L))
        # AR fit (shared)
        p = int(min(max(round(math.log10(n)), 1), 4))
        st.p = p
        y, phi, eps, sig2 = _K.ar_fit_p(g, st.mean_g, p)
        st.phi = phi.tolist()
        Ne = eps.shape[0]
        st.Ne = Ne
        sig2 = float(sig2)
        st.sig2_eps = sig2
        st.ylag = [float(y[n - 1 - k]) for k in range(p)]  # y_{t-1}, ..., y_{t-p}
        # whiteness scores
        el = [float(eps[Ne - 1 - k]) if Ne - 1 - k >= 0 else 0.0 for k in range(WHITE_LAGS)]
        st.elag = el
        st.S_white = [0.0] * WHITE_LAGS
        if Ne > WHITE_LAGS + 8:
            Sw = np.stack([eps[WHITE_LAGS:] * eps[WHITE_LAGS - k: Ne - k] / sig2
                           for k in range(1, WHITE_LAGS + 1)], axis=1)
            st.mu_s = Sw.mean(axis=0).tolist()
            st.Sinv_s = reg_inv(sample_cov(Sw)).tolist()
        else:
            st.mu_s = [0.0] * WHITE_LAGS
            st.Sinv_s = np.eye(WHITE_LAGS).tolist()
        # sign products
        sg = np.sign(g)
        sp = sg[1:] * sg[:-1]
        st.sp_mean = float(sp.mean())
        st.sp_sd = max(float(sp.std()), 0.05)
        st.g_prev = float(g[-1])
        st.cusum_p = 0.0
        st.cusum_m = 0.0
        # GARCH(1,1) on the AR innovations
        e2 = eps * eps
        s2 = sig2
        a_hat, b_hat, _, h, d1, d2, d3, v = _K.garch_block_p(np.ascontiguousarray(e2), s2)
        a_hat, b_hat = float(a_hat), float(b_hat)
        st.g_a, st.g_b, st.g_s2 = a_hat, b_hat, s2
        st.g_omega = s2 * (1.0 - a_hat - b_hat)
        st.g_floor = 1e-8 * s2
        Dm = np.stack([d1, d2, d3], axis=1)
        sc = (0.5 * (e2 / h - 1.0) / h)[:, None] * Dm
        sc_h = sc[1:] if Ne >= 2 else sc
        st.mu_G = sc_h.mean(axis=0)
        SinvG = reg_inv(sample_cov(sc_h))
        st.SinvG = SinvG.tolist()
        st.S_G = [0.0, 0.0, 0.0]
        st.h_prev = float(h[-1])
        st.e2_prev = float(e2[-1])
        st.d_prev = [float(Dm[-1, 0]), float(Dm[-1, 1]), float(Dm[-1, 2])]
        st.v_mean = float(v.mean())
        vc = v - st.v_mean
        var_v = float(vc.var())
        if Ne >= 3 and var_v > 0.0:
            rho = float(np.mean(vc[1:] * vc[:-1]) / var_v)
        else:
            rho = 0.0
        rho = min(max(rho, -0.995), 0.995)
        st.lrv_v = max(var_v * (1.0 + rho) / (1.0 - rho), 1e-3)
        eh2 = e2 / h
        mu0e = float(eh2.mean())
        st.mu0e = mu0e
        st.s_e = max(float((eh2 / mu0e).var()), 0.05)
        # AR score vectors
        A = np.empty((Ne, p + 1))
        for k in range(1, p + 1):
            A[:, k - 1] = eps * y[p - k: n - k]
        A[:, p] = e2 - sig2
        st.mu_A = A.mean(axis=0)
        SinvAi = reg_inv(sample_cov(A))
        SinvAb = reg_inv(bartlett_cov(A, L))
        # bridges
        es = np.sort(eps)
        st.eps_sorted = es.tolist()
        aes = np.sort(np.abs(eps))
        st.abs_eps_sorted = aes.tolist()
        clt = np.searchsorted(es, eps, side="left")
        ceq = np.searchsorted(es, eps, side="right") - clt
        eta = ndtri_array((clt + 0.5 * ceq) / Ne)
        Ta = np.minimum(eta * eta, INNOV_CLIP2)
        st.ta_mean, st.ta_sd = float(Ta.mean()), max(float(Ta.std()), 1e-3)
        aeps = np.abs(eps)
        clt2 = np.searchsorted(aes, aeps, side="left")
        ceq2 = np.searchsorted(aes, aeps, side="right") - clt2
        Tb = (clt2 + 0.5 * ceq2) / Ne
        st.tb_mean, st.tb_sd = float(Tb.mean()), max(float(Tb.std()), 1e-3)
        # FOCuS detectors
        st.focus_var = _Focus(n, True)
        st.focus_mean = _Focus(n, False)
        st.focus_e = _Focus(n, True)
        # window prefix matrix: [He(2) | POT I,E (2) | cells (3) | GARCH scores (3) | vol (1) | A (p+1) | A (p+1)]
        D = 11 + 2 * (p + 1)
        st.D = D
        st.iA = 11
        Bm = np.zeros((D, D))
        Bm[0:2, 0:2] = SinvH
        Bm[4:7, 4:7] = SinvC
        Bm[7:10, 7:10] = SinvG
        Bm[11: 12 + p, 11: 12 + p] = SinvAi
        Bm[12 + p: 13 + 2 * p, 12 + p: 13 + 2 * p] = SinvAb
        st.B = Bm
        Gm = np.zeros((D, 5))
        Gm[0:2, 0] = 1.0
        Gm[4:7, 1] = 1.0
        Gm[7:10, 2] = 1.0
        Gm[11: 12 + p, 3] = 1.0
        Gm[12 + p: 13 + 2 * p, 4] = 1.0
        st.Gm = Gm
        st.cap = 256
        st.P = np.zeros((st.cap + 1, D))
        st.G2 = np.zeros((st.cap + 1, 2))
        st.m = 0
        st.cols = np.zeros(N_COLS)
        st.tb_cur = -1
        if self.model is not None:
            st.hb = int(np.sum(self.nh_edges <= n))
        return st

    # ------------------------------------------------------------------ columns
    def step_columns(self, st: _State, x_t: float) -> np.ndarray:
        """Advance the state with one online point; return the 18 raw columns (new array)."""
        x = float(x_t)
        n = st.n
        m = st.m + 1
        st.m = m
        if m > st.cap:
            newcap = 2 * st.cap
            P2 = np.zeros((newcap + 1, st.D))
            P2[: st.cap + 1] = st.P
            st.P = P2
            G3 = np.zeros((newcap + 1, 2))
            G3[: st.cap + 1] = st.G2
            st.G2 = G3
            st.cap = newcap
        # rank-native input
        xs = st.xs
        c_lt = bisect.bisect_left(xs, x)
        c_eq = bisect.bisect_right(xs, x) - c_lt
        u = (c_lt + 0.5 * c_eq + 0.5) / (n + 1.0)
        g = ndtri(u)
        g2 = g * g
        x2 = g2 / st.mu0
        cm = 1.0 + m / n
        out = np.empty(N_COLS)
        # C1 FOCuS variance
        out[0] = st.focus_var.update(x2) / st.s1
        # C2 Sanso / Horvath
        Cvar = st.focus_var.C  # sum of x2 over the prefix
        out[1] = abs(Cvar - m) / math.sqrt(st.omega4 * m * cm)
        # C3 SR mixture
        ls = st.sr_l
        tot = NEG_INF
        for i in range(6):
            th = SR_THETAS[i]
            lp = ls[i]
            if lp > 0.0:
                la = lp + math.log1p(math.exp(-lp))
            else:
                la = math.log1p(math.exp(lp))
            ln = la - 0.5 * math.log(th) - 0.5 * x2 * (1.0 / th - 1.0)
            ls[i] = ln
            if tot == NEG_INF:
                tot = ln
            elif ln > tot:
                tot = ln + math.log1p(math.exp(tot - ln))
            else:
                tot = tot + math.log1p(math.exp(ln - tot))
        out[2] = (tot - LOG6 - math.log(m) - st.sr_mu) / st.sr_sd
        # C4 FOCuS mean
        y = g - st.mean_g
        out[3] = st.focus_mean.update(y) / st.lrv_g
        # Hermite transform
        M = st.M_clip
        gcl = g if -M <= g <= M else (M if g > M else -M)
        he3 = gcl * gcl * gcl - 3.0 * gcl
        he4 = gcl * gcl * gcl * gcl - 6.0 * gcl * gcl + 3.0
        # POT
        ag = abs(g)
        I_pot = 1.0 if ag > st.tau_pot else 0.0
        E_pot = ag - st.tau_pot if I_pot > 0.0 else 0.0
        # cells
        c1 = 1.0 if u < 0.1 else 0.0
        c2 = 1.0 if u < 0.5 else 0.0
        c3 = 1.0 if u > 0.9 else 0.0
        # AR innovation
        phi = st.phi
        ylag = st.ylag
        p = st.p
        eps = y
        for k in range(p):
            eps -= phi[k] * ylag[k]
        # whiteness (C8, C9)
        elag = st.elag
        Sw = st.S_white
        inv_s2 = 1.0 / st.sig2_eps
        for k in range(WHITE_LAGS):
            Sw[k] += eps * elag[k] * inv_s2
        mcm = m * cm
        out[7] = (Sw[0] * Sw[0] + Sw[1] * Sw[1] + Sw[2] * Sw[2] + Sw[3] * Sw[3]) / mcm
        mu_s = st.mu_s
        Sinv = st.Sinv_s
        dv = [Sw[0] - m * mu_s[0], Sw[1] - m * mu_s[1], Sw[2] - m * mu_s[2], Sw[3] - m * mu_s[3]]
        q = 0.0
        for i in range(WHITE_LAGS):
            row = Sinv[i]
            q += dv[i] * (row[0] * dv[0] + row[1] * dv[1] + row[2] * dv[2] + row[3] * dv[3])
        out[8] = q / mcm
        # C10 sign CUSUM
        sg = 1.0 if g > 0.0 else (-1.0 if g < 0.0 else 0.0)
        gp = st.g_prev
        sgp = 1.0 if gp > 0.0 else (-1.0 if gp < 0.0 else 0.0)
        ysp = (sg * sgp - st.sp_mean) / st.sp_sd
        cp = st.cusum_p + ysp - CUSUM_K
        cmn = st.cusum_m - ysp - CUSUM_K
        st.cusum_p = cp if cp > 0.0 else 0.0
        st.cusum_m = cmn if cmn > 0.0 else 0.0
        out[9] = st.cusum_p if st.cusum_p > st.cusum_m else st.cusum_m
        st.g_prev = g
        # GARCH step
        h = st.g_omega + st.g_a * st.e2_prev + st.g_b * st.h_prev
        if h < st.g_floor:
            h = st.g_floor
        b = st.g_b
        dp = st.d_prev
        d0 = 1.0 + b * dp[0]
        d1 = st.e2_prev + b * dp[1]
        d2 = st.h_prev + b * dp[2]
        e2 = eps * eps
        eh2 = e2 / h
        f = 0.5 * (eh2 - 1.0) / h
        muG = st.mu_G
        s0 = f * d0 - muG[0]
        s1 = f * d1 - muG[1]
        s2 = f * d2 - muG[2]
        SG = st.S_G
        SG[0] += s0
        SG[1] += s1
        SG[2] += s2
        SiG = st.SinvG
        q = 0.0
        for i in range(3):
            row = SiG[i]
            q += SG[i] * (row[0] * SG[0] + row[1] * SG[1] + row[2] * SG[2])
        out[10] = q / mcm
        v_vol = _K.log_p(h) - st.v_mean
        out[13] = st.focus_e.update(eh2 / st.mu0e) / st.s_e
        st.h_prev = h
        st.e2_prev = e2
        st.d_prev = [d0, d1, d2]
        # AR score vector
        muA = st.mu_A
        avals = [eps * ylag[k] - muA[k] for k in range(p)]
        avals.append(e2 - st.sig2_eps - muA[p])
        # bridges transforms
        es = st.eps_sorted
        Ne = st.Ne
        cl = bisect.bisect_left(es, eps)
        ce = bisect.bisect_right(es, eps) - cl
        eta = ndtri((cl + 0.5 * ce + 0.5) / (Ne + 1.0))
        Ta = eta * eta
        if Ta > INNOV_CLIP2:
            Ta = INNOV_CLIP2
        aes = st.abs_eps_sorted
        ae = abs(eps)
        cl2 = bisect.bisect_left(aes, ae)
        ce2 = bisect.bisect_right(aes, ae) - cl2
        Tb = (cl2 + 0.5 * ce2) / Ne
        # prefix matrices
        muH = st.muH
        pC = st.pC
        row = [he3 - muH[0], he4 - muH[1], I_pot, E_pot, c1 - pC[0], c2 - pC[1], c3 - pC[2],
               s0, s1, s2, v_vol] + avals + avals
        P = st.P
        P[m] = P[m - 1] + np.array(row)
        G2 = st.G2
        G2[m, 0] = G2[m - 1, 0] + Ta
        G2[m, 1] = G2[m - 1, 1] + Tb
        # active windows
        nw = bisect.bisect_right(WGRID_LIST, m)
        if nw > 0 and WGRID_LIST[nw - 1] == m:
            ws = WGRID[:nw]
        else:
            ws = np.concatenate([WGRID[:nw], np.array([m], dtype=np.int64)])
        wsf = ws.astype(np.float64)
        denom = wsf * (1.0 + wsf / n)
        W = P[m] - P[m - ws]
        Q = ((W @ st.B) * W) @ st.Gm / denom[:, None]
        T = Q.max(axis=0)
        out[4] = T[0]
        out[6] = T[1]
        out[11] = T[2]
        out[14] = T[3]
        out[15] = T[4]
        # POT
        Nw = W[:, 2]
        Sw_ = W[:, 3]
        zc = (Nw - wsf * st.p_pot) ** 2 / (wsf * st.lrv_pot * (1.0 + wsf / n))
        Nw_safe = np.where(Nw > 0.0, Nw, 1.0)
        ze = np.where(Nw > 0.0, (Sw_ - Nw * st.ebar_pot) ** 2 / (Nw_safe * st.ve_pot * (1.0 + Nw / st.Nhx)), 0.0)
        out[5] = float(np.max(zc + ze))
        # volatility level
        zv = W[:, 10] / np.sqrt(wsf * st.lrv_v * (1.0 + wsf / n))
        out[12] = float(zv[int(np.argmax(np.abs(zv)))])
        # bridges
        if m == 1:
            out[16] = abs(Ta - st.ta_mean) / st.ta_sd
            out[17] = abs(Tb - st.tb_mean) / st.tb_sd
        else:
            kk = np.arange(1, m, dtype=np.float64)
            frac = kk / m
            num = np.abs(G2[1:m] - frac[:, None] * G2[m])
            den = np.sqrt(frac * (1.0 - frac)) * math.sqrt(m)
            bb = (num / den[:, None]).max(axis=0)
            out[16] = bb[0] / st.ta_sd
            out[17] = bb[1] / st.tb_sd
        # update lags
        if p > 0:
            st.ylag = [y] + ylag[: p - 1]
        st.elag = [eps] + elag[: WHITE_LAGS - 1]
        # NaN guard (never expected)
        if not np.all(np.isfinite(out)):
            out[~np.isfinite(out)] = 0.0
        st.cols = out
        return out

    # ------------------------------------------------------------------ score
    def step(self, st: _State, x_t: float) -> float:
        if self.model is None:
            raise RuntimeError("Scorer(None) is columns-only; load an exported fold model to score")
        cols = self.step_columns(st, x_t)
        m = st.m
        tb = t_bucket(m)
        if tb != st.tb_cur:
            st.tb_cur = tb
            st.K_cur = self.knots[st.hb, tb]
            st.slo_cur = self.slo_all[st.hb, tb]
            st.shi_cur = self.shi_all[st.hb, tb]
        u = pit_transform(cols, st.K_cur, st.slo_cur, st.shi_cur)
        uc = np.clip(u, U_CLIP, 1.0 - U_CLIP)
        lg = np.log(uc / (1.0 - uc))
        f = np.concatenate([lg, lg[PAIRS_I] * lg[PAIRS_J]])
        s = self.intercept + float(f @ self.coef)
        return 1.0 / (1.0 + math.exp(-4.0 * (s - 0.5)))

    def score_series(self, x_historical: np.ndarray, x_online: np.ndarray) -> np.ndarray:
        st = self.start(x_historical)
        return np.array([self.step(st, float(v)) for v in x_online])

    def columns_series(self, x_historical: np.ndarray, x_online: np.ndarray) -> np.ndarray:
        st = self.start(x_historical)
        return np.stack([self.step_columns(st, float(v)).copy() for v in x_online], axis=0)
