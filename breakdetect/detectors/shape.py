"""Distribution-shape traces: quantile mixture ("qms"), tail exceedances ("pot"), Hermite CUSUM."""

from __future__ import annotations

import math

import numpy as np

from ..kernels import shape as kernels
from .common import WINSOR_C, bartlett_omega4, null_substreams, standardize

QMS_LEVELS = (0.01, 0.05, 0.5, 0.95, 0.99)
QMS_LOGIT_GRID = (-2.0, -1.0, -0.5, 0.5, 1.0, 2.0)
POT_LEVELS = (0.01, 0.05, 0.95, 0.99)
HERMITE_K = 0.25
HERMITE_YCAP = 8.0


def _order_statistic(a_sorted: np.ndarray, level: float) -> tuple[float, float]:
    """Empirical ``level`` quantile of the history and the fraction of points at or below it."""
    n = a_sorted.shape[0]
    idx = min(max(math.ceil(level * n) - 1, 0), n - 1)
    q = float(a_sorted[idx])
    return q, float(np.searchsorted(a_sorted, q, side="right"))


class ShapeTrace:
    """One shape detector, z-scored against circular null lanes of the history."""

    def __init__(self, name: str, history) -> None:
        mu_h, sd_h, a_hist = standardize(history)
        omega4 = bartlett_omega4(a_hist)
        n_null = int(a_hist.shape[0])
        starts, factors = null_substreams(n_null, omega4, True)
        n_lanes = len(starts) + 1
        idx = (np.arange(n_null)[:, None] + np.asarray(starts, dtype=np.int64)[None, :]) % n_null
        vals_null = a_hist[idx] * np.asarray(factors, dtype=np.float64)[None, :]
        a_sorted = np.sort(a_hist)
        n = a_sorted.shape[0]
        lo, hi = 1.0 / (n + 1.0), n / (n + 1.0)

        stats = np.empty(n_lanes, dtype=np.float64)
        if name == "qms":
            thresholds, theta0 = [], []
            for level in QMS_LEVELS:
                q, k_leq = _order_statistic(a_sorted, level)
                thresholds.append(q)
                theta0.append(min(max(k_leq / n, lo), hi))
            q_arr = np.asarray(thresholds, dtype=np.float64)
            th0 = np.asarray(theta0, dtype=np.float64)
            grid = np.asarray(QMS_LOGIT_GRID, dtype=np.float64)
            logit0 = np.log(th0) - np.log1p(-th0)
            th1 = 1.0 / (1.0 + np.exp(-(logit0[:, None] + grid[None, :])))
            l1 = np.log(th1) - np.log(th0)[:, None]
            l0 = np.log1p(-th1) - np.log1p(-th0)[:, None]
            logr = np.full((n_lanes, q_arr.size, grid.size), -700.0, dtype=np.float64)
            work = np.empty(max(n_lanes, q_arr.size), dtype=np.float64)
            extra = (logr, q_arr, l1, l0, math.log(grid.size), np.empty(q_arr.size * grid.size))
            self._kernel = kernels.qms_step
        elif name == "pot":
            thresholds, theta0, is_hi = [], [], []
            for level in POT_LEVELS:
                q, k_leq = _order_statistic(a_sorted, level)
                upper = level >= 0.5
                rate_leq = k_leq / n
                rate = (1.0 - rate_leq) if upper else rate_leq
                thresholds.append(q)
                theta0.append(min(max(rate, lo), hi))
                is_hi.append(upper)
            m = len(thresholds)
            work = np.empty(n_lanes, dtype=np.float64)
            extra = (
                np.zeros((n_lanes, m, kernels.POT_CAP + 2), dtype=np.int64),
                np.zeros((n_lanes, m, kernels.POT_CAP + 2), dtype=np.int64),
                np.ones((n_lanes, m), dtype=np.int64),
                np.zeros((n_lanes, m), dtype=np.int64),
                np.zeros(1, dtype=np.int64),
                np.asarray(thresholds, dtype=np.float64),
                np.asarray(theta0, dtype=np.float64),
                np.asarray([1 if v else 0 for v in is_hi], dtype=np.int64),
            )
            self._kernel = kernels.pot_step
        elif name == "hermite":
            if a_sorted.size < 2:
                raise ValueError("a_hist_sorted must be 1-D with >= 2 points")
            cap = WINSOR_C * float(max(abs(a_sorted[0]), abs(a_sorted[-1])))
            h2 = a_sorted * a_sorted
            h3 = h2 * a_sorted - 3.0 * a_sorted
            h4 = h2 * h2 - 6.0 * h2 + 3.0
            sd3 = float(h3.std(ddof=1))
            sd4 = float(h4.std(ddof=1))
            work = np.empty(max(n_lanes, 4), dtype=np.float64)
            extra = (
                np.zeros((n_lanes, 4), dtype=np.float64),
                cap,
                float(h3.mean()),
                float(h4.mean()),
                1.0 / sd3 if sd3 > 1e-8 else 1.0,
                1.0 / sd4 if sd4 > 1e-8 else 1.0,
                HERMITE_K,
                HERMITE_YCAP,
            )
            self._kernel = kernels.hermite_step
        else:
            raise ValueError(f"unknown shape detector {name!r}")
        self._args = (
            float(mu_h),
            float(sd_h),
            n_null,
            np.ascontiguousarray(vals_null, dtype=np.float64),
            np.zeros(1, dtype=np.int64),
            np.empty(n_lanes, dtype=np.float64),
            stats,
            work,
            *extra,
        )

    def step(self, x: float) -> float:
        """Consume one online point and return the standardized statistic."""
        z, status = self._kernel(float(x), *self._args)
        if status:
            raise ValueError("NaN online point")
        return z
