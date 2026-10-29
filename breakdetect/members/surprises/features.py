"""Per-series features of surprises: AR-whitened innovations summarized over trailing windows.

The history fixes the standardization, an AR(p) filter (p <= 8 by AIC) and the null
moments of 17 per-point quantities (level, scale, tails, ranks, dependence). Each online
point is turned into an innovation and the quantities are averaged over trailing windows
(8 to 256 points and the whole prefix), standardized against the history, plus CUSUM and
Page statistics. The result is a float32 vector of 182 features.
"""

from __future__ import annotations

import math

import numpy as np

P_MAX = 8
CLIP_C = 8.0
WINDOWS = (8, 16, 32, 64, 128, 256)
TAIL_Q = 0.95
PAGE_DRIFT = 0.5
EPS = 1e-6
SD_FLOOR = 1e-3
LRV_CAP = (1e-3, 1e3)

Q_NAMES = (
    "ec",
    "ec2",
    "abs_ec",
    "ec4",
    "ec3",
    "u",
    "v",
    "tail",
    "udisp",
    "r1",
    "r2",
    "r14",
    "sgn",
    "z",
    "zc2",
    "g1",
    "g2",
)
N_Q = len(Q_NAMES)
N_ELAG = 4
LOG_Q = (1, 2, 14)
CUSUM_Q = (0, 1, 6, 5)
N_WIN = len(WINDOWS) + 1
N_PER_WIN = N_Q + len(LOG_Q) + 2
N_CUSUM = 4 * len(CUSUM_Q)
CONTEXT_NAMES = (
    "log_nhist",
    "ar_order",
    "log_seps",
    "sum_phi",
    "hist_acf1",
    "log_lrv",
    "hist_kurt",
    "hist_skew",
    "hist_mean_ec2",
    "hist_mean_abs",
)
N_CONTEXT = len(CONTEXT_NAMES)
N_STEP = 2
N_FEATURES = N_WIN * N_PER_WIN + N_CUSUM + N_CONTEXT + N_STEP


def _levinson(r: np.ndarray, p_max: int):
    """Levinson-Durbin recursion: AR coefficients and innovation variances for orders 0..p_max."""
    sig2 = [float(r[0])]
    phis = [np.zeros(0)]
    phi = np.zeros(0)
    for k in range(1, p_max + 1):
        if sig2[-1] <= 0.0:
            phis.append(phi.copy())
            sig2.append(sig2[-1])
            continue
        acc = float(r[k])
        for j in range(k - 1):
            acc -= float(phi[j]) * float(r[k - 1 - j])
        kappa = acc / sig2[-1]
        kappa = max(-0.999, min(0.999, kappa))
        new = np.empty(k)
        if k > 1:
            new[: k - 1] = phi - kappa * phi[::-1]
        new[k - 1] = kappa
        phi = new
        phis.append(phi.copy())
        sig2.append(sig2[-1] * (1.0 - kappa * kappa))
    return phis, sig2


def _midrank(sorted_ref: np.ndarray, values) -> np.ndarray:
    """Mid-rank of ``values`` in a sorted reference, in [0, 1]."""
    n = sorted_ref.shape[0]
    lo = np.searchsorted(sorted_ref, values, side="left")
    hi = np.searchsorted(sorted_ref, values, side="right")
    return (lo + hi) / (2.0 * n)


def _quantities(sgn_e, ec, z, zc, u, v, ec_lags, z_lags, sgn_prev):
    """The 17 per-point quantities (scalars or arrays, same elementwise operations)."""
    ec2 = ec * ec
    ec4 = ec2 * ec2
    ec3 = ec2 * ec
    ud = (u - 0.5) * (u - 0.5)
    r1 = ec * ec_lags[0]
    r2 = ec * ec_lags[1]
    r14 = ec * (((ec_lags[0] + ec_lags[1]) + ec_lags[2]) + ec_lags[3])
    sgn = sgn_e * sgn_prev
    zc2 = zc * zc
    g1 = ec * z_lags[0]
    g2 = ec * z_lags[1]
    tail = (v > TAIL_Q) * 1.0
    abs_ec = ec * (ec >= 0.0) - ec * (ec < 0.0)
    return [ec, ec2, abs_ec, ec4, ec3, u, v, tail, ud, r1, r2, r14, sgn, z, zc2, g1, g2]


def fit_hist(x_hist) -> dict:
    """Standardization, AR filter and null reference of one history."""
    x = np.asarray(x_hist, dtype=np.float64)
    n = x.shape[0]
    mu = float(np.sum(x) / n)
    xc = x - mu
    var = float(np.sum(xc * xc) / n)
    sd = math.sqrt(var) if var > 1e-24 else 1.0
    z = xc / sd
    p_cap = min(P_MAX, max(0, n // 100))
    r = np.empty(P_MAX + 1)
    for k in range(P_MAX + 1):
        r[k] = np.sum(z[k:] * z[: n - k]) / n
    phis, sig2 = _levinson(r, p_cap)
    best_p, best_aic = 0, math.inf
    for k in range(p_cap + 1):
        s2 = max(sig2[k], 1e-12)
        aic = n * math.log(s2) + 2.0 * k
        if aic < best_aic - 1e-12:
            best_aic, best_p = aic, k
    phi = np.zeros(P_MAX)
    phi[:best_p] = phis[best_p]
    T = n - P_MAX
    pred = np.zeros(T)
    for k in range(1, P_MAX + 1):
        pred += phi[k - 1] * z[P_MAX - k : P_MAX - k + T]
    raw_e = z[P_MAX:] - pred
    s_eps = math.sqrt(float(np.sum(raw_e * raw_e) / T))
    if not (s_eps > 1e-6):
        s_eps = 1e-6
    e = raw_e / s_eps
    ec = np.clip(e, -CLIP_C, CLIP_C)
    sorted_e = np.sort(e)
    sorted_abs = np.sort(np.abs(e))
    u = _midrank(sorted_e, e)
    v = _midrank(sorted_abs, np.abs(e))
    sgn_e = np.sign(e)
    L = N_ELAG
    ec_lags = [ec[L - k : T - k] for k in range(1, N_ELAG + 1)]
    zz = z[P_MAX:]
    zfull = z
    z_lags = [zfull[P_MAX + L - k : P_MAX + T - k] for k in (1, 2)]
    zc_h = np.clip(zz, -CLIP_C, CLIP_C)
    q = _quantities(
        sgn_e[L:],
        ec[L:],
        zz[L:],
        zc_h[L:],
        u[L:],
        v[L:],
        ec_lags,
        z_lags,
        sgn_e[L - 1 : T - 1],
    )
    Qm = np.stack(q)
    mu0 = Qm.mean(axis=1)
    sd0 = Qm.std(axis=1)
    sd0 = np.maximum(sd0, SD_FLOOR)
    m2 = float(mu0[1])
    m4 = float(mu0[3])
    m3 = float(mu0[4])
    kurt0 = m4 / (m2 * m2 + 1e-12)
    skew0 = m3 / (m2 * math.sqrt(m2) + 1e-12)
    sum_phi = float(np.sum(phi))
    lrv = (s_eps * s_eps) / max((1.0 - sum_phi) ** 2, 1e-12)
    lrv = min(max(lrv, LRV_CAP[0]), LRV_CAP[1])
    mu0[13] = 0.0
    sd0[13] = math.sqrt(lrv)
    acf1 = float(r[1] / r[0]) if r[0] > 0 else 0.0
    context = np.array(
        [
            math.log(n),
            float(best_p),
            math.log(s_eps),
            sum_phi,
            acf1,
            math.log(lrv),
            math.log(kurt0 + EPS),
            skew0,
            m2,
            float(mu0[2]),
        ]
    )
    return {
        "mu": mu,
        "sd": sd,
        "phi": phi,
        "s_eps": s_eps,
        "sorted_e": sorted_e,
        "sorted_abs": sorted_abs,
        "mu0": mu0,
        "sd0": sd0,
        "log_kurt0": math.log(kurt0 + EPS),
        "skew0": skew0,
        "context": context,
        "z_tail": z[n - P_MAX :][::-1].copy(),
        "ec_tail": ec[T - N_ELAG :][::-1].copy(),
        "sgn_tail": float(sgn_e[T - 1]),
    }


def _window_features(S, n, const):
    """Standardized window means, log ratios, kurtosis and skewness from window sums ``S``."""
    mu0 = const["mu0"]
    sd0 = const["sd0"]
    shp = (N_Q,) + (1,) * (S.ndim - 1)
    mean = S / n
    sq = np.sqrt(n)
    zf = (mean - mu0.reshape(shp)) * sq / sd0.reshape(shp)
    logs = []
    for qi in LOG_Q:
        logs.append(np.log((mean[qi] + EPS) / (mu0[qi] + EPS)))
    m2 = mean[1]
    m4 = mean[3]
    m3 = mean[4]
    kurt = m4 / (m2 * m2 + 1e-12)
    logkurt = np.log(kurt + EPS) - const["log_kurt0"]
    skew = m3 / (m2 * np.sqrt(m2) + 1e-12) - const["skew0"]
    blocks = [zf] + [lg[None] for lg in logs] + [logkurt[None], skew[None]]
    F = np.concatenate(blocks, axis=0)
    F = np.swapaxes(F, 0, 1)
    return F.reshape((N_WIN * N_PER_WIN,) + F.shape[2:])


_WIN_ARR = np.array(WINDOWS, dtype=np.int64)


class StreamState:
    """Online state of one series: lag buffers, prefix sums and CUSUM accumulators."""

    __slots__ = (
        "const",
        "m",
        "cap",
        "C",
        "zlag",
        "eclag",
        "sgn_prev",
        "Sy",
        "runmin",
        "runmax",
        "pu",
        "pd",
        "cus_mu",
        "cus_sd",
    )

    def __init__(self, const: dict):
        self.const = const
        self.m = 0
        self.cap = 1024
        self.C = np.zeros((N_Q, self.cap))
        self.zlag = [float(v) for v in const["z_tail"]]
        self.eclag = [float(v) for v in const["ec_tail"]]
        self.sgn_prev = float(const["sgn_tail"])
        nc = len(CUSUM_Q)
        self.Sy = np.zeros(nc)
        self.runmin = np.zeros(nc)
        self.runmax = np.zeros(nc)
        self.pu = np.zeros(nc)
        self.pd = np.zeros(nc)
        self.cus_mu = const["mu0"][list(CUSUM_Q)].copy()
        self.cus_sd = const["sd0"][list(CUSUM_Q)].copy()


def stream_step(st: StreamState, x_t) -> np.ndarray:
    """Consume one online point and return its float32 feature vector."""
    const = st.const
    z = (float(x_t) - const["mu"]) / const["sd"]
    phi = const["phi"]
    zlag = st.zlag
    pred = 0.0
    for k in range(P_MAX):
        pred += phi[k] * zlag[k]
    e = (z - pred) / const["s_eps"]
    ec = min(max(e, -CLIP_C), CLIP_C)
    zc = min(max(z, -CLIP_C), CLIP_C)
    sgn_e = 1.0 if e > 0.0 else (-1.0 if e < 0.0 else 0.0)
    u = float(_midrank(const["sorted_e"], e))
    v = float(_midrank(const["sorted_abs"], -e if e < 0.0 else e))
    q = _quantities(sgn_e, ec, z, zc, u, v, st.eclag, zlag, st.sgn_prev)
    qv = np.array(q, dtype=np.float64)
    st.zlag = [z] + zlag[:-1]
    st.eclag = [ec] + st.eclag[:-1]
    st.sgn_prev = sgn_e
    m = st.m + 1
    if m >= st.cap:
        st.C = np.concatenate([st.C, np.zeros((N_Q, st.cap))], axis=1)
        st.cap *= 2
    C = st.C
    C[:, m] = C[:, m - 1] + qv
    st.m = m
    idx = np.empty(N_WIN, dtype=np.int64)
    np.maximum(m - _WIN_ARR, 0, out=idx[: N_WIN - 1])
    idx[N_WIN - 1] = 0
    n = (m - idx).astype(np.float64)
    S = C[:, m][:, None] - C[:, idx]
    Fw = _window_features(S, n, const)
    y = (qv[list(CUSUM_Q)] - st.cus_mu) / st.cus_sd
    Sy = st.Sy + y
    st.Sy = Sy
    st.runmin = np.minimum(st.runmin, np.minimum(Sy, 0.0))
    st.runmax = np.maximum(st.runmax, np.maximum(Sy, 0.0))
    sqm = math.sqrt(float(m))
    st.pu = np.maximum(0.0, (st.pu + y) - PAGE_DRIFT)
    st.pd = np.maximum(0.0, (st.pd - y) - PAGE_DRIFT)
    cus = np.stack([(Sy - st.runmin) / sqm, (st.runmax - Sy) / sqm, st.pu, st.pd], axis=1).reshape(-1)
    step = np.array([float(m - 1), 0.0])
    step[1:] = np.log1p(step[:1])
    F = np.concatenate([Fw, cus, const["context"], step])
    return F.astype(np.float32)
