"""Per-series feature engine of surprises (numpy only; shared by training and inference).

Historical setup (``fit_hist``):
  * standardize the series by its historical mean / sd;
  * fit an AR(p) model by Yule-Walker (Levinson-Durbin) on the standardized
    historical, order p in 0..P_MAX chosen by AIC;
  * filter the historical with the frozen AR to obtain innovations, scale them
    by their historical sd (so innovations are ~unit variance under the null);
  * build the null reference: sorted innovations (signed and absolute) for
    midrank PIT transforms, and mean / sd of every per-point quantity.

Online stream (``batch_quantities`` / ``StreamState``):
  each online point is standardized, one-step AR-predicted from the causal
  lag buffer (initialised from the historical tail, carried forward without
  refitting), turned into a standardized innovation e, and expanded into
  N_Q per-point quantities (level, scale, tail, rank, dependence, AR-score
  terms).  Causal evidence is accumulated over trailing windows
  WINDOWS + the full online prefix, standardized against the historical null,
  plus CUSUM-type statistics with unknown change time.

The batch (vectorized over time) and the streaming (one point at a time)
implementations perform the same elementwise floating-point operations in the
same order so that both produce identical feature vectors.
"""
from __future__ import annotations

import math

import numpy as np

P_MAX = 8  # maximal AR order (fixed lag-buffer length)
CLIP_C = 8.0  # clipping of standardized innovations for moment features
WINDOWS = (8, 16, 32, 64, 128, 256)  # trailing windows; the full prefix is added
TAIL_Q = 0.95  # |e| rank threshold for the tail exceedance indicator
PAGE_DRIFT = 0.5  # drift of the Page CUSUM (in null-sd units)
EPS = 1e-6
SD_FLOOR = 1e-3
LRV_CAP = (1e-3, 1e3)

# per-point quantities (rows of the quantity matrix)
Q_NAMES = (
    "ec",  # 0 clipped standardized innovation (level)
    "ec2",  # 1 squared (scale)
    "abs_ec",  # 2 absolute (scale, robust-ish)
    "ec4",  # 3 fourth power (kurtosis numerator)
    "ec3",  # 4 third power (skewness numerator)
    "u",  # 5 signed midrank PIT of e in the historical innovations
    "v",  # 6 midrank of |e| in the historical |innovations|
    "tail",  # 7 indicator v > TAIL_Q
    "udisp",  # 8 (u - 0.5)^2 rank dispersion (shape)
    "r1",  # 9 ec_t * ec_{t-1} (lag-1 dependence)
    "r2",  # 10 ec_t * ec_{t-2}
    "r14",  # 11 ec_t * (ec_{t-1}+...+ec_{t-4}) (low-frequency power)
    "sgn",  # 12 sign(e_t) sign(e_{t-1})
    "z",  # 13 standardized raw value (level w.r.t. long-run variance)
    "zc2",  # 14 clipped raw value squared (raw scale)
    "g1",  # 15 ec_t * z_{t-1} (AR score, lag 1)
    "g2",  # 16 ec_t * z_{t-2} (AR score, lag 2)
)
N_Q = len(Q_NAMES)
N_ELAG = 4  # innovation lags needed by r14
# quantities that also get a log-ratio feature and the shape features
LOG_Q = (1, 2, 14)
CUSUM_Q = (0, 1, 6, 5)  # level, scale, robust scale, robust level
N_WIN = len(WINDOWS) + 1
N_PER_WIN = N_Q + len(LOG_Q) + 2  # z-types + log ratios + kurt + skew
N_CUSUM = 4 * len(CUSUM_Q)  # range up/down, page up/down
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
N_STEP = 2  # t, log1p(t)
N_FEATURES = N_WIN * N_PER_WIN + N_CUSUM + N_CONTEXT + N_STEP


def feature_names() -> list[str]:
    names = []
    wins = [f"w{w}" for w in WINDOWS] + ["wfull"]
    for w in wins:
        for q in Q_NAMES:
            names.append(f"{w}_z_{q}")
        for qi in LOG_Q:
            names.append(f"{w}_log_{Q_NAMES[qi]}")
        names.append(f"{w}_logkurt")
        names.append(f"{w}_skew")
    for qi in CUSUM_Q:
        q = Q_NAMES[qi]
        names += [f"cus_up_{q}", f"cus_dn_{q}", f"page_up_{q}", f"page_dn_{q}"]
    names += list(CONTEXT_NAMES)
    names += ["t", "log1p_t"]
    assert len(names) == N_FEATURES
    return names


# --------------------------------------------------------------------------
# historical setup
# --------------------------------------------------------------------------
def _levinson(r: np.ndarray, p_max: int):
    """Levinson-Durbin on autocovariances r[0..p_max]; returns (phis, sig2)."""
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
    n = sorted_ref.shape[0]
    lo = np.searchsorted(sorted_ref, values, side="left")
    hi = np.searchsorted(sorted_ref, values, side="right")
    return (lo + hi) / (2.0 * n)


def _quantities(sgn_e, ec, z, zc, u, v, ec_lags, z_lags, sgn_prev):
    """Assemble the N_Q per-point quantities.

    Every argument is either a scalar (streaming) or a 1-D array over time
    (batch); ``ec_lags[k-1]`` is ec_{t-k}, ``z_lags[k-1]`` is z_{t-k}.
    Elementwise operations only, identical in both modes.
    """
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
    """Historical setup: standardization, AR fit, innovation null reference."""
    x = np.asarray(x_hist, dtype=np.float64)
    n = x.shape[0]
    mu = float(np.sum(x) / n)
    xc = x - mu
    var = float(np.sum(xc * xc) / n)
    sd = math.sqrt(var) if var > 1e-24 else 1.0
    z = xc / sd
    # Yule-Walker autocovariances (biased) and AIC order selection
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
    # innovations on the historical, using the fixed P_MAX-lag filter
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
    # rows with all lags available: e-index t >= N_ELAG
    L = N_ELAG
    ec_lags = [ec[L - k : T - k] for k in range(1, N_ELAG + 1)]
    zz = z[P_MAX:]  # z aligned with e
    zfull = z  # z_{t-k} for e-index t is z[P_MAX + t - k]
    z_lags = [zfull[P_MAX + L - k : P_MAX + T - k] for k in (1, 2)]
    zc_h = np.clip(zz, -CLIP_C, CLIP_C)
    q = _quantities(
        sgn_e[L:], ec[L:], zz[L:], zc_h[L:], u[L:], v[L:], ec_lags, z_lags,
        sgn_e[L - 1 : T - 1],
    )
    Qm = np.stack(q)  # (N_Q, T-L)
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
    # raw-level quantity z: standardize by sqrt(lrv) instead of its own sd
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
        "z_tail": z[n - P_MAX :][::-1].copy(),  # z_{t-1}, z_{t-2}, ...
        "ec_tail": ec[T - N_ELAG :][::-1].copy(),  # ec_{t-1}, ...
        "sgn_tail": float(sgn_e[T - 1]),
    }


# --------------------------------------------------------------------------
# features from window sums (shared elementwise formulas)
# --------------------------------------------------------------------------
def _window_features(S, n, const):
    """S: (N_Q, K, ...) window sums; n: (K, ...) window lengths (float).

    Returns array (N_PER_WIN * K, ...) in the fixed feature order.
    """
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
    blocks = [zf] + [l[None] for l in logs] + [logkurt[None], skew[None]]
    F = np.concatenate(blocks, axis=0)  # (N_PER_WIN, K, ...)
    # reorder to window-major: feature index = k * N_PER_WIN + j
    F = np.swapaxes(F, 0, 1)  # (K, N_PER_WIN, ...)
    return F.reshape((N_WIN * N_PER_WIN,) + F.shape[2:])


# --------------------------------------------------------------------------
# batch (vectorized over time) computation for training / OOF
# --------------------------------------------------------------------------
def batch_features(const: dict, x_online) -> np.ndarray:
    """Return the (T, N_FEATURES) float32 feature matrix for one series."""
    x = np.asarray(x_online, dtype=np.float64)
    T = x.shape[0]
    z = (x - const["mu"]) / const["sd"]
    phi = const["phi"]
    z_tail = const["z_tail"]  # z_{t-1}, ..., z_{t-P}
    zfull = np.concatenate([z_tail[::-1], z])  # chronological
    pred = np.zeros(T)
    for k in range(1, P_MAX + 1):
        pred += phi[k - 1] * zfull[P_MAX - k : P_MAX - k + T]
    e = (z - pred) / const["s_eps"]
    ec = np.clip(e, -CLIP_C, CLIP_C)
    u = _midrank(const["sorted_e"], e)
    v = _midrank(const["sorted_abs"], np.abs(e))
    ecfull = np.concatenate([const["ec_tail"][::-1], ec])
    ec_lags = [ecfull[N_ELAG - k : N_ELAG - k + T] for k in range(1, N_ELAG + 1)]
    z_lags = [zfull[P_MAX - k : P_MAX - k + T] for k in (1, 2)]
    sgn_e = np.sign(e)
    sgnfull = np.concatenate([[const["sgn_tail"]], sgn_e])
    sgn_prev = sgnfull[:T]
    zc = np.clip(z, -CLIP_C, CLIP_C)
    Qm = np.stack(_quantities(sgn_e, ec, z, zc, u, v, ec_lags, z_lags, sgn_prev))  # (N_Q, T)
    C = np.zeros((N_Q, T + 1))
    np.cumsum(Qm, axis=1, out=C[:, 1:])
    m = np.arange(1, T + 1)
    idx = np.empty((N_WIN, T), dtype=np.int64)
    for k, w in enumerate(WINDOWS):
        idx[k] = np.maximum(m - w, 0)
    idx[N_WIN - 1] = 0
    n = (m[None, :] - idx).astype(np.float64)  # (K, T)
    S = C[:, m][:, None, :] - C[:, idx]  # (N_Q, K, T)
    Fw = _window_features(S, n, const)  # (N_WIN*N_PER_WIN, T)
    # CUSUM statistics
    cus = np.empty((N_CUSUM, T))
    sqm = np.sqrt(m.astype(np.float64))
    for j, qi in enumerate(CUSUM_Q):
        y = (Qm[qi] - const["mu0"][qi]) / const["sd0"][qi]
        Sy = np.cumsum(y)
        runmin = np.minimum.accumulate(np.minimum(Sy, 0.0))
        runmax = np.maximum.accumulate(np.maximum(Sy, 0.0))
        cus[4 * j] = (Sy - runmin) / sqm
        cus[4 * j + 1] = (runmax - Sy) / sqm
        pu = 0.0
        pd = 0.0
        pus = np.empty(T)
        pds = np.empty(T)
        for t in range(T):
            yt = y[t]
            pu = max(0.0, (pu + yt) - PAGE_DRIFT)
            pd = max(0.0, (pd - yt) - PAGE_DRIFT)
            pus[t] = pu
            pds[t] = pd
        cus[4 * j + 2] = pus
        cus[4 * j + 3] = pds
    ctx = np.repeat(const["context"][:, None], T, axis=1)
    tt = np.arange(T, dtype=np.float64)
    step = np.stack([tt, np.log1p(tt)])
    F = np.concatenate([Fw, cus, ctx, step], axis=0)  # (N_FEATURES, T)
    return np.ascontiguousarray(F.T).astype(np.float32)


# --------------------------------------------------------------------------
# streaming computation (inference)
# --------------------------------------------------------------------------
_WIN_ARR = np.array(WINDOWS, dtype=np.int64)


class StreamState:
    __slots__ = (
        "const", "m", "cap", "C", "zlag", "eclag", "sgn_prev",
        "Sy", "runmin", "runmax", "pu", "pd", "cus_mu", "cus_sd",
    )

    def __init__(self, const: dict):
        self.const = const
        self.m = 0
        self.cap = 1024
        self.C = np.zeros((N_Q, self.cap))
        self.zlag = [float(v) for v in const["z_tail"]]  # z_{t-1}, ...
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


def stream_start(x_hist) -> StreamState:
    return StreamState(fit_hist(x_hist))


def stream_step(st: StreamState, x_t) -> np.ndarray:
    """Consume one online point; return the float32 feature vector."""
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
    # update lag buffers
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
    S = C[:, m][:, None] - C[:, idx]  # (N_Q, K)
    Fw = _window_features(S, n, const)  # (N_WIN*N_PER_WIN,)
    # CUSUM
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
