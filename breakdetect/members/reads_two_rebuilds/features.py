"""Inputs of the reads_two_rebuilds and reads_three_others members besides the surprises and
pattern_removed vectors and the detector traces.

``SummaryStream`` gives 36 running statistics of AR(5) innovations per point; ``context4`` gives
four constants of the history (squared-value autocorrelation, kurtosis and two long-run
variances of the squares).
"""

from __future__ import annotations

import math

import numpy as np

from ...detectors.common import bartlett_omega4, bartlett_omega4_prewhitened

P_AR = 5
HALF_LIVES = (10, 40, 150)


def series_context(x_hist) -> tuple[float, float]:
    """Lag-1 autocorrelation of the squared z-scores and kurtosis of the history."""
    a = np.asarray(x_hist, dtype=np.float64)
    mu = float(a.mean())
    sd = float(a.std(ddof=1))
    if not np.isfinite(sd) or sd < 1e-8:
        sd = 1e-8
    z = (a - mu) / sd
    s = z * z
    s0 = s[:-1] - s[:-1].mean()
    s1 = s[1:] - s[1:].mean()
    den = float(np.sqrt(float((s0 * s0).sum()) * float((s1 * s1).sum())))
    rho1_sq = float((s0 * s1).sum() / den) if den > 0.0 else 0.0
    m2 = float((z * z).mean())
    m4 = float((z**4).mean())
    kurt = m4 / (m2 * m2) if m2 > 0.0 else 3.0
    return rho1_sq, kurt


def standardized_hist(x_hist) -> np.ndarray:
    """History z-scored with its mean and sample sd (floored at 1e-8)."""
    x_h = np.asarray(x_hist, dtype=np.float64)
    mu_h = float(x_h.mean())
    sd_h = float(x_h.std(ddof=1)) if x_h.size > 1 else 1.0
    sd_h = max(sd_h, 1e-8)
    return (x_h - mu_h) / sd_h


def context4(h: np.ndarray) -> list[float]:
    """[rho1 of squares, kurtosis, omega4, prewhitened omega4] of the history."""
    r, ku = series_context(h)
    a = standardized_hist(h)
    return [float(r), float(ku), float(bartlett_omega4(a)), float(bartlett_omega4_prewhitened(a))]


class SummaryStream:
    """Running moments, CUSUM ranges and exponential averages of AR(5) innovations (36 values)."""

    def __init__(self, h: np.ndarray) -> None:
        p = P_AR
        X = np.column_stack([h[p - k - 1 : len(h) - k - 1] for k in range(p)])
        yv = h[p:]
        phi = np.linalg.lstsq(X, yv, rcond=None)[0]
        eh = yv - X @ phi
        se = eh.std() + 1e-12
        uh = eh / se
        self.phi = [float(v) for v in phi]
        self.se = float(se)
        self.lags = [float(h[-1 - k]) for k in range(p)]
        self.mabs_h = float(np.mean(np.abs(uh)) + 1e-12)
        self.m4_h = float(np.mean(uh**4) + 1e-12)
        self.rate_h = float(np.clip(np.mean(np.abs(uh) > 2.5), 1e-4, 0.5))
        self.v2_h = float(np.var(uh**2) + 1e-12)
        self.acf1 = float(np.corrcoef(h[1:], h[:-1])[0, 1])
        kurt_raw_h = float(np.mean(h**4) / (np.mean(h**2) ** 2 + 1e-12))
        self.rate_den = self.rate_h * (1 - self.rate_h)
        self.ctx6 = [float(np.log(len(h))), self.acf1, kurt_raw_h, float(np.sum(phi)), float(np.log(se)), self.rate_h]
        self.up1 = float(uh[-1])
        self.up2 = float(uh[-2])
        self.oprev = float(h[-1])
        self.t = 0
        self.c1 = self.c2 = self.ca = self.c4 = self.cd1 = self.cd2 = self.svar = 0.0
        self.cex = 0
        self.cmax = 0.0
        self.ro1 = self.ro2 = self.rod = 0.0
        self.mn_sv = self.mx_sv = self.mn_c1 = self.mx_c1 = self.mn_d = self.mx_d = 0.0
        self.ew_a = []
        for hl in HALF_LIVES:
            a = 1.0 - 0.5 ** (1.0 / hl)
            self.ew_a.append((a, 1.0 - a, math.sqrt(hl)))
        self.ew_y = [[0.0, 0.0, 0.0, 0.0] for _ in HALF_LIVES]

    def step(self, o: float) -> list:
        """Consume one online value and return the 36 statistics."""
        ph, lg = self.phi, self.lags
        pred = ph[0] * lg[0] + ph[1] * lg[1] + ph[2] * lg[2] + ph[3] * lg[3] + ph[4] * lg[4]
        u = (o - pred) / self.se
        lg.insert(0, o)
        lg.pop()
        self.t += 1
        tt = float(self.t)
        u2 = u * u
        au = abs(u)
        d1 = u * self.up1
        self.c1 += u
        self.c2 += u2
        self.ca += au
        self.c4 += u2 * u2
        self.cd1 += d1
        self.cd2 += u * self.up2
        if au > 2.5:
            self.cex += 1
        self.svar += u2 - 1.0
        if au > self.cmax:
            self.cmax = au
        self.ro1 += o
        self.ro2 += o * o
        self.rod += o * self.oprev
        self.oprev = o
        self.up2 = self.up1
        self.up1 = u
        sv, c1, cd1 = self.svar, self.c1, self.cd1
        self.mn_sv = min(self.mn_sv, sv)
        self.mx_sv = max(self.mx_sv, sv)
        self.mn_c1 = min(self.mn_c1, c1)
        self.mx_c1 = max(self.mx_c1, c1)
        self.mn_d = min(self.mn_d, cd1)
        self.mx_d = max(self.mx_d, cd1)
        sq = math.sqrt(tt)
        dv = math.sqrt(tt * self.v2_h)
        log = math.log
        f = [
            math.log1p(tt - 1),
            c1 / sq,
            log(self.c2 / tt + 1e-9),
            log(self.ca / tt / self.mabs_h + 1e-9),
            log(self.c4 / tt / self.m4_h + 1e-9),
            cd1 / sq,
            self.cd2 / sq,
            (self.cex / tt - self.rate_h) / math.sqrt(self.rate_den / tt),
            log(self.cmax + 1e-9),
            (sv - self.mn_sv) / dv,
            (self.mx_sv - sv) / dv,
            (c1 - self.mn_c1) / sq,
            (self.mx_c1 - c1) / sq,
            (cd1 - self.mn_d) / sq,
            (self.mx_d - cd1) / sq,
            self.ro1 / sq,
            log(self.ro2 / tt + 1e-9),
            self.rod / max(self.ro2, 1e-9) - self.acf1,
        ]
        for (a, om, sh), y in zip(self.ew_a, self.ew_y, strict=True):
            y[0] = y[0] * om + a * u
            y[1] = y[1] * om + a * u2
            y[2] = y[2] * om + a * d1
            y[3] = y[3] * om + a * au
            nrm = 1.0 - math.pow(om, tt)
            f.append(y[0] / nrm * sh)
            f.append(log(y[1] / nrm + 1e-9))
            f.append(y[2] / nrm * sh)
            f.append(y[3] / nrm / self.mabs_h)
        f.extend(self.ctx6)
        return f
