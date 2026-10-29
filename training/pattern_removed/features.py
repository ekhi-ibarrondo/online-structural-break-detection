"""Feature engine of pattern_removed: numpy+stdlib only; no online-length interface."""

from __future__ import annotations
import math
import numpy as np

W = np.array([1, 4, 16, 64, 256], dtype=np.int64)
PROBS = np.linspace(1 / 128, 127 / 128, 33)
TRANSFORMS = [
    "level",
    "energy",
    "absolute",
    "difference_energy",
    "lag1_product",
    "ar64_residual",
    "ar64_energy",
    "seasonal_residual",
    "seasonal_energy",
]
A_NAMES = [
    "focus_variance",
    "sanso",
    "sr_variance",
    "focus_mean",
    "hermite",
    "pot",
    "rank_quantile",
    "bn_iid",
    "bn_bartlett",
    "sign_cusum",
    "garch_score",
    "garch_reset",
    "garch_level",
    "garch_variance",
    "ar_quadratic_iid",
    "ar_quadratic_bartlett",
    "bridge_square",
    "bridge_rank",
]
FEATURE_NAMES = (
    [f"D_w{w}_{s}_{c}" for w in W for s in TRANSFORMS for c in ["pit_logit", "effect"]]
    + ["log1p_step", "step_over_step_plus64"]
    + ["A_" + s + "_pit_logit" for s in A_NAMES]
)
RATIOS = np.array([0.5, 0.75, 1.5, 2, 3])
SR_A = -0.5 * np.log(RATIOS)
SR_B = 0.5 * (1 - 1 / RATIOS)


def pit_logit(x, knots):
    """Rowwise monotone quantile interpolation with fixed Pareto tails."""
    x = np.asarray(x, dtype=np.float64)
    # knots (...,33); x (...,). Search ties deterministically to the right.
    k = np.minimum(31, np.maximum(0, np.sum(x[..., None] >= knots, axis=-1) - 1))
    a = np.take_along_axis(knots, k[..., None], axis=-1)[..., 0]
    b = np.take_along_axis(knots, (k + 1)[..., None], axis=-1)[..., 0]
    d = np.maximum(b - a, 1e-12)
    u = PROBS[k] + (PROBS[k + 1] - PROBS[k]) * (x - a) / d
    low_scale = np.maximum((knots[..., 1] - knots[..., 0]) * PROBS[0] / (PROBS[1] - PROBS[0]), 1e-6)
    high_scale = np.maximum((knots[..., -1] - knots[..., -2]) * (1 - PROBS[-1]) / (PROBS[-1] - PROBS[-2]), 1e-6)
    u = np.where(x < knots[..., 0], PROBS[0] / (1 + np.maximum(knots[..., 0] - x, 0) / low_scale), u)
    u = np.where(x >= knots[..., -1], 1 - (1 - PROBS[-1]) / (1 + np.maximum(x - knots[..., -1], 0) / high_scale), u)
    u = np.clip(u, 1e-6, 1 - 1e-6)
    return np.log(u) - np.log1p(-u)


def covariance(a, longrun=False):
    a = np.asarray(a, dtype=np.float64)
    if a.ndim == 1:
        a = a[:, None]
    a = a - a.mean(axis=0)
    n = len(a)
    c = a.T @ a / max(1, n)
    if longrun:
        bw = min(16, max(1, int(n ** (1 / 3))))
        for j in range(1, min(bw + 1, n)):
            v = a[j:].T @ a[:-j] / n
            c += (1 - j / (bw + 1)) * (v + v.T)
    return (c + c.T) * 0.5


def whitener(a, longrun=False):
    c = covariance(a, longrun)
    eig, v = np.linalg.eigh(c)
    floor = max(1e-6, float(np.trace(c) / len(c)) * 0.01)
    return (v * (1 / np.sqrt(np.maximum(eig, floor)))) @ v.T


def ar_fit(z, p):
    p = min(p, max(1, len(z) - 1))
    n = len(z)
    ac = np.array([float(z[j:] @ z[: n - j]) / max(1, n) if j < n else 0 for j in range(p + 1)])
    c = ac[np.abs(np.arange(p)[:, None] - np.arange(p))]
    c.flat[:: p + 1] += max(1e-8, 0.01 * ac[0])
    coef = np.linalg.solve(c, ac[1:])
    return coef


def lags_matrix(z, p):
    pad = np.r_[np.zeros(p), z]
    return np.stack([pad[p - j : p - j + len(z)] for j in range(1, p + 1)], axis=1)


def garch_fit(z):
    aa = np.array([0, 0.03, 0.03, 0.03, 0.1, 0.1, 0.2])
    bb = np.array([0, 0.6, 0.8, 0.95, 0.6, 0.8, 0.6])
    var = max(float(np.mean(z * z)), 1e-8)
    oo = (1 - aa - bb) * var
    h = np.full(7, var)
    obj = np.zeros(7)
    hs = np.empty((len(z), 7))
    prev = var
    for i, zz in enumerate(z):
        h = oo + aa * prev + bb * h
        hs[i] = h
        obj += np.log(h) + zz * zz / h
        prev = zz * zz
    k = int(np.argmin(obj))
    a = float(aa[k])
    b = float(bb[k])
    o = float(oo[k])
    hv = hs[:, k]
    score = np.empty((len(z), 3))
    der = np.zeros(3)
    prev = var
    hp = var
    for i, zz in enumerate(z):
        der = np.array([1.0, prev, hp]) + b * der
        score[i] = 0.5 * (zz * zz / hv[i] - 1) * der / hv[i]
        hp = hv[i]
        prev = zz * zz
    return o, a, b, hv, score, der


class Hull:
    """Monotone upper/lower hull of cumulative sufficient statistic."""

    def __init__(self):
        self.lo = [(0.0, 0.0)]
        self.hi = [(0.0, 0.0)]
        self.t = 0
        self.total = 0.0

    @staticmethod
    def _push(h, x, y, sign):
        while len(h) > 1:
            a, b = h[-2], h[-1]
            cross = (b[0] - a[0]) * (y - b[1]) - (b[1] - a[1]) * (x - b[0])
            if sign * cross > 0:
                break
            h.pop()
        h.append((x, y))

    def step(self, v, variance=False):
        self.t += 1
        self.total += float(v)
        pts = np.asarray(self.lo + self.hi)
        length = self.t - pts[:, 0]
        s = self.total - pts[:, 1]
        if variance:
            q = np.maximum(s / length, 1e-12)
            val = float(np.max(0.5 * length * (q - 1 - np.log(q))))
        else:
            val = float(np.max(0.5 * s * s / length))
        self._push(self.lo, self.t, self.total, 1)
        self._push(self.hi, self.t, self.total, -1)
        return max(0.0, val)


def suffix_grid(t):
    out = list(range(1, min(16, t) + 1))
    v = 16
    while v < t:
        v = int(math.ceil(v * 1.5))
        if v < t:
            out.append(v)
    if not out or out[-1] != t:
        out.append(t)
    return np.array(out, dtype=np.int64)


class FeatureState:
    def __init__(self, x_historical):
        x = np.asarray(x_historical, dtype=np.float64)
        if x.ndim != 1 or len(x) == 0 or not np.isfinite(x).all():
            raise ValueError("historical must be nonempty finite vector")
        self.n_hist = len(x)
        self.mean = float(x.mean())
        self.sd = max(1e-8, float(x.std()))
        z = np.clip((x - self.mean) / self.sd, -12, 12)
        # Pad only estimator algebra for exceptionally short histories; n_hist remains real.
        zh = z if len(z) >= 4 else np.pad(z, (4 - len(z), 0), mode="edge")
        self.ar64 = ar_fit(zh, min(64, max(1, self.n_hist // 8)))
        self.ar = ar_fit(zh, min(4, max(1, self.n_hist // 64)))
        self.p = len(self.ar)
        candidates = [s for s in [4, 8, 16, 32, 64, 128] if s <= self.n_hist // 4] or [1]
        ac = [float(zh[s:] @ zh[:-s]) / max(1, float(zh[:-s] @ zh[:-s])) for s in candidates]
        sk = int(np.argmax(np.abs(ac)))
        self.season = candidates[sk]
        self.season_beta = float(np.clip(ac[sk], -0.98, 0.98))
        lag64 = lags_matrix(zh, len(self.ar64))
        lag = lag64[:, : self.p]
        eh = zh - lag @ self.ar
        valid = slice(min(max(len(self.ar64), self.p, self.season), len(zh) // 2), None)
        self.emean = float(eh[valid].mean())
        self.esd = max(1e-6, float(eh[valid].std()))
        eh = (eh - self.emean) / self.esd
        self.abs_e = np.sort(np.abs(eh[valid]))
        self.zquant = np.quantile(zh, [0.1, 0.5, 0.9])
        self.pot = float(np.quantile(np.abs(zh), 0.9))
        self.omega, self.alpha, self.beta, hv, gs, self.gder = garch_fit(zh)
        self.gh = float(hv[-1])
        self.gprev = float(zh[-1] ** 2)
        self.emean2 = float(np.mean(zh * zh))
        self.emean2 = max(1e-8, self.emean2)
        self.varscale = max(1e-6, float(np.var(zh * zh / self.emean2)) / 2)
        self.gqmean = max(1e-8, float(np.mean(zh * zh / hv)))
        self.gqsd = max(1e-6, float(np.std(zh * zh / hv)))
        self.gvarscale = max(1e-6, float(np.var(zh * zh / hv / self.gqmean)) / 2)
        self.sanso_sd = math.sqrt(max(1e-6, float(covariance(zh * zh, True)[0, 0])))
        sp = np.sign(eh[1:]) * np.sign(eh[:-1])
        self.signmean = float(sp.mean())
        self.signsd = max(1e-6, float(sp.std()))
        self.eprev = float(eh[-1])
        eclip = np.minimum(eh * eh, 25)
        erank = (
            np.searchsorted(self.abs_e, np.abs(eh), side="left") + np.searchsorted(self.abs_e, np.abs(eh), side="right")
        ) / (2 * len(self.abs_e))
        bridges = np.stack([eclip, erank], axis=1)[valid]
        self.bmean = bridges.mean(axis=0)
        self.bsd = np.maximum(bridges.std(axis=0), 1e-6)
        h3 = (zh**3 - 3 * zh) / math.sqrt(6)
        h4 = (zh**4 - 6 * zh**2 + 3) / math.sqrt(24)
        herm = np.stack([h3, h4], axis=1)
        pot = np.stack([(np.abs(zh) > self.pot).astype(float), np.maximum(np.abs(zh) - self.pot, 0)], axis=1)
        rank = (zh[:, None] <= self.zquant).astype(float)
        ars = np.column_stack([eh[:, None] * lag, eh * eh])
        groups = [herm, pot, rank, ars, gs]
        self.rawmean = np.concatenate([g[valid].mean(axis=0) for g in groups])
        nr = sum(g.shape[1] for g in groups)
        nc = nr + self.p + 1
        self.lin = np.zeros((nr, nc))
        r = 0
        c = 0
        self.slices = []
        for gi, g in enumerate(groups):
            d = g.shape[1]
            self.lin[r : r + d, c : c + d] = whitener(g[valid])
            self.slices.append(slice(c, c + d))
            c += d
            if gi == 3:
                self.long_slice = slice(c, c + d)
                self.lin[r : r + d, c : c + d] = whitener(g[valid], True)
                c += d
            r += d
        arres = np.clip(zh - lag64 @ self.ar64, -12, 12)
        sl = lags_matrix(zh, self.season)[:, -1]
        seares = np.clip(zh - self.season_beta * sl, -12, 12)
        prev = np.r_[0, zh[:-1]]
        dif = np.clip(zh - prev, -12, 12)
        dh = np.column_stack(
            [zh, zh * zh, np.abs(zh), dif * dif, zh * prev, arres, arres * arres, seares, seares * seares]
        )[valid]
        cs = np.vstack([np.zeros(9), np.cumsum(dh, axis=0)])
        knots = []
        mu = []
        sd = []
        for w in W:
            v = min(int(w), len(dh))
            roll = (cs[v:] - cs[:-v]) / v
            knots.append(np.quantile(roll, PROBS, axis=0).T)
            mu.append(roll.mean(axis=0))
            sd.append(np.maximum(roll.std(axis=0), 1e-6))
        self.dknots = np.concatenate(knots, axis=0)
        self.dmean = np.concatenate(mu)
        self.dsd = np.concatenate(sd)
        self.zbuffer = np.zeros(max(128, len(self.ar64), self.season) + 1)
        tail = zh[-len(self.zbuffer) :]
        self.zbuffer[-len(tail) :] = tail
        self.cap = 512
        self.dcs = np.zeros((self.cap + 256, 9))
        tail = dh[-256:]
        if len(tail) < 256:
            tail = np.pad(tail, ((256 - len(tail), 0), (0, 0)), mode="edge")
        self.dcs[1:257] = np.cumsum(tail, axis=0)
        self.pcs = np.zeros((self.cap, nc))
        self.bcs = np.zeros((self.cap, 2))
        self.t = 0
        self.hulls = [Hull(), Hull(), Hull()]
        self.sr = np.full(5, -np.inf)
        self.gsr = np.full(5, -np.inf)
        self.sum_square = 0.0
        self.sum_gq = 0.0
        self.signplus = 0.0
        self.signminus = 0.0
        self.greset = np.zeros(3)
        self.gcount = 0

    def step(self, x_t):
        x_t = float(x_t)
        if not math.isfinite(x_t):
            raise ValueError("online point must be finite")
        z = max(-12.0, min(12.0, (x_t - self.mean) / self.sd))
        self.t += 1
        t = self.t
        if t >= self.cap:
            self.dcs = np.pad(self.dcs, ((0, self.cap), (0, 0)))
            self.pcs = np.pad(self.pcs, ((0, self.cap), (0, 0)))
            self.bcs = np.pad(self.bcs, ((0, self.cap), (0, 0)))
            self.cap *= 2
        prev = float(self.zbuffer[-1])
        lag = self.zbuffer[-self.p :][::-1]
        e = (z - float(lag @ self.ar) - self.emean) / self.esd
        ar = max(-12.0, min(12.0, z - float(self.zbuffer[-len(self.ar64) :][::-1] @ self.ar64)))
        se = max(-12.0, min(12.0, z - self.season_beta * self.zbuffer[-self.season]))
        dif = max(-12.0, min(12.0, z - prev))
        d = np.array([z, z * z, abs(z), dif * dif, z * prev, ar, ar * ar, se, se * se])
        idx = 256 + t
        self.dcs[idx] = self.dcs[idx - 1] + d
        dm = ((self.dcs[idx] - self.dcs[idx - W]) / W[:, None]).ravel()
        out = np.empty(110, dtype=np.float64)
        out[:90:2] = pit_logit(dm, self.dknots)
        out[1:90:2] = np.clip((dm - self.dmean) / self.dsd, -30, 30)
        out[90] = math.log1p(t)
        out[91] = t / (t + 64)
        hp = self.gh
        self.gh = self.omega + self.alpha * self.gprev + self.beta * hp
        self.gder = np.array([1.0, self.gprev, hp]) + self.beta * self.gder
        gq = z * z / self.gh
        gs = 0.5 * (gq - 1) * self.gder / self.gh
        self.gprev = z * z
        h3 = (z**3 - 3 * z) / math.sqrt(6)
        h4 = (z**4 - 6 * z * z + 3) / math.sqrt(24)
        raw = np.r_[
            h3,
            h4,
            float(abs(z) > self.pot),
            max(abs(z) - self.pot, 0),
            (z <= self.zquant).astype(float),
            e * lag,
            e * e,
            gs,
        ]
        prim = (raw - self.rawmean) @ self.lin
        self.pcs[t] = self.pcs[t - 1] + prim
        lengths = suffix_grid(t)
        ss = self.pcs[t] - self.pcs[t - lengths]
        ss /= np.sqrt(lengths[:, None])
        sq = ss * ss
        a = np.empty(18)
        a[0] = self.hulls[0].step(z * z / self.emean2, True) / self.varscale
        self.sum_square += z * z - self.emean2
        a[1] = abs(self.sum_square) / (self.sanso_sd * math.sqrt(t * (1 + t / self.n_hist)))
        self.sr = np.logaddexp(0, self.sr) + SR_A + SR_B * z * z / self.emean2
        a[2] = max(0.0, float(np.logaddexp.reduce(self.sr)) - math.log(5 * t))
        a[3] = self.hulls[1].step(z)
        a[4] = np.max(np.sum(sq[:, self.slices[0]], axis=1))
        a[5] = np.max(np.sum(sq[:, self.slices[1]], axis=1))
        a[6] = np.max(np.sum(sq[:, self.slices[2]], axis=1))
        asi = self.slices[3]
        asl = self.long_slice
        a[7] = np.max(np.abs(ss[:, asi.start : asi.stop - 1]))
        a[8] = np.max(np.abs(ss[:, asl.start : asl.stop - 1]))
        sign = (
            (1 if e > 0 else -1 if e < 0 else 0) * (1 if self.eprev > 0 else -1 if self.eprev < 0 else 0)
            - self.signmean
        ) / self.signsd
        self.eprev = e
        self.signplus = max(0.0, self.signplus + sign - 0.25)
        self.signminus = max(0.0, self.signminus - sign - 0.25)
        a[9] = max(self.signplus, self.signminus)
        gp = prim[self.slices[4]]
        gpsum = self.pcs[t, self.slices[4]]
        a[10] = float(gpsum @ gpsum) / t
        self.greset += gp
        self.gcount += 1
        a[11] = max(0.0, float(self.greset @ self.greset) / self.gcount - 0.5 * self.gcount)
        if a[11] <= 0:
            self.greset[:] = 0
            self.gcount = 0
        self.sum_gq += (gq - self.gqmean) / self.gqsd
        a[12] = self.sum_gq / math.sqrt(t)
        self.gsr = np.logaddexp(0, self.gsr) + SR_A + SR_B * gq / self.gqmean
        a[13] = max(
            self.hulls[2].step(gq / self.gqmean, True) / self.gvarscale,
            float(np.logaddexp.reduce(self.gsr)) - math.log(5 * t),
            0.0,
        )
        a[14] = np.max(np.sum(sq[:, asi], axis=1))
        a[15] = np.max(np.sum(sq[:, asl], axis=1))
        ae = abs(e)
        rank = (np.searchsorted(self.abs_e, ae, side="left") + np.searchsorted(self.abs_e, ae, side="right")) / (
            2 * len(self.abs_e)
        )
        bv = (np.array([min(e * e, 25), rank]) - self.bmean) / self.bsd
        self.bcs[t] = self.bcs[t - 1] + bv
        if t == 1:
            a[16:] = bv
        else:
            ks = np.unique(np.r_[lengths[lengths < t], t - lengths[lengths < t], t // 2])
            u = ks / t
            a[16:] = np.max(
                np.abs(self.bcs[ks] - u[:, None] * self.bcs[t]) / (math.sqrt(t) * (u * (1 - u))[:, None] ** 0.25),
                axis=0,
            )
        out[92:] = np.log1p(np.maximum(a, 0))
        out[[104, 108, 109]] = np.arcsinh(a[[12, 16, 17]])
        self.zbuffer[:-1] = self.zbuffer[1:]
        self.zbuffer[-1] = z
        return out
