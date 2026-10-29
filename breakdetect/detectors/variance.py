"""Variance traces: FOCuS, Sanso kappa-2 and a Shiryaev-Roberts mixture over variance ratios."""

from __future__ import annotations

import math

import numpy as np

from ..kernels import variance as kernels
from .common import FOCUS_MAX_MAIN, FOCUS_MAX_NULL, WINSOR_C, bartlett_omega4, null_substreams, standardize

DETECTORS = ("focus_var", "sanso_k2", "mix_sr_var")
SR_GRID = (0.25, 0.5, 0.7, 1.5, 2.0, 4.0)
_KINDS = {"focus_var": kernels.KIND_FOCUS, "sanso_k2": kernels.KIND_SANSO, "mix_sr_var": kernels.KIND_MIX}


class VarianceTrace:
    """Mean pivotal z of the selected variance detectors, calibrated on injected null sub-streams."""

    def __init__(self, names: tuple[str, ...], history) -> None:
        mu_h, sd_h, a_hist = standardize(history)
        omega4 = bartlett_omega4(a_hist)
        n_hist = int(a_hist.shape[0])
        cap = WINSOR_C * float(np.max(np.abs(a_hist)))
        if "sanso_k2" in names and not (math.isfinite(float(omega4)) and float(omega4) > 0.0):
            raise ValueError(f"omega4 must be a finite positive float, got {omega4!r}")
        starts, factors = null_substreams(n_hist, omega4, True)
        n_sub = len(starts)
        n_streams = 1 + n_sub

        cap_m = cap_n = 2
        if "focus_var" in names:
            cap_m, cap_n = FOCUS_MAX_MAIN, FOCUS_MAX_NULL
        sanso_omega4, inv_n_main = 0.0, 0.0
        if "sanso_k2" in names:
            sanso_omega4, inv_n_main = float(omega4), 1.0 / float(n_hist)
        mx_c1 = np.zeros(1, dtype=np.float64)
        mx_c2 = np.zeros(1, dtype=np.float64)
        mx_logw = np.zeros(1, dtype=np.float64)
        mx_add = 0.0
        m_grid = 1
        if "mix_sr_var" in names:
            weights = [1.0 / len(SR_GRID)] * len(SR_GRID)
            m_grid = len(SR_GRID)
            mx_c1 = np.asarray([-0.5 * math.log(v) for v in SR_GRID], dtype=np.float64)
            mx_c2 = np.asarray([0.5 * (1.0 / v - 1.0) for v in SR_GRID], dtype=np.float64)
            mx_logw = np.asarray([math.log(v) for v in weights], dtype=np.float64)
            mx_add = float(-math.log1p(-0.0))

        self._args = (
            float(mu_h),
            float(sd_h),
            float(cap),
            np.zeros(1, dtype=np.int64),
            np.ascontiguousarray(a_hist, dtype=np.float64),
            np.asarray(starts, dtype=np.int64),
            np.asarray(factors, dtype=np.float64),
            np.zeros(n_sub, dtype=np.float64),
            np.asarray([_KINDS[nm] for nm in names], dtype=np.int64),
            np.zeros((len(names), n_sub), dtype=np.float64),
            np.zeros((1, 2, cap_m), dtype=np.int64),
            np.zeros((1, 2, cap_m), dtype=np.float64),
            np.ones((1, 2, cap_m), dtype=np.float64),
            np.array([[0, 1, 1]], dtype=np.int64),
            np.zeros(1, dtype=np.float64),
            np.zeros((n_sub, 2, cap_n), dtype=np.int64),
            np.zeros((n_sub, 2, cap_n), dtype=np.float64),
            np.ones((n_sub, 2, cap_n), dtype=np.float64),
            np.tile(np.array([0, 1, 1], dtype=np.int64), (n_sub, 1)),
            np.zeros(n_sub, dtype=np.float64),
            np.zeros(n_streams, dtype=np.int64),
            np.zeros((n_streams, 3), dtype=np.float64),
            sanso_omega4,
            inv_n_main,
            np.full((n_streams, m_grid), -700.0, dtype=np.float64),
            mx_c1,
            mx_c2,
            mx_logw,
            mx_add,
        )

    def step(self, x: float) -> float:
        """Consume one online point and return the mean z."""
        return kernels.variance_step(float(x), *self._args)
