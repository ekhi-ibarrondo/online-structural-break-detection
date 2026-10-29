"""numba build of ``kernels_src`` under fixed compilation pins.

1. ``numba_runtime`` is the first import that touches numba: it writes
   ``NUMBA_CPU_NAME=generic``, ``NUMBA_DISABLE_INTEL_SVML=1`` and
   ``NUMBA_NUM_THREADS=1`` before ``import numba`` and verifies the versions
   (0.67.0 / 0.49.0), SVML off, ``CPU_NAME``/``CPU_FEATURES`` and the real
   codegen target ``('generic', '')``.
2. There is no pure-Python production path: ``TRAINING_NUMBA=0`` raises ``RuntimeError``.
3. ``kernels_src.py`` is loaded by path as ``<package>.kernels_src__nb`` with
   ``_KERNEL_DECORATOR = numba_runtime.kernel_decorator`` injected before
   execution, so each kernel is ``njit(sig, cache=False, fastmath=False,
   boundscheck=False, nogil=False)`` and compiled eagerly at decoration
   (explicit signature).  Nothing is written to disk.
4. The compile time is recorded (``record_compile("order_not_size", ...)``); ``REPORT`` is the banner.
"""
from __future__ import annotations

import importlib.util
import time
from pathlib import Path

from . import numba_runtime as _rt

_rt.require_numba()
if not _rt.enabled():
    raise RuntimeError("the order_not_size kernels require numba: TRAINING_NUMBA=0 is not supported")

KERNELS = ("_bits", "_from_bits", "_hi", "_lo", "_set_hi", "_set_lo0", "log_p", "exp_p", "erfc_p",
           "ndtri_p", "ndtri_array_p", "nsum_p", "ndot_p", "mean_p", "yule_walker_p", "ar_fit_p",
           "ar1_filter_p", "garch_filter_p", "garch_loss_p", "_in_box", "garch_fit_p", "garch_block_p")

_T0 = time.perf_counter()
SRC_PATH = Path(__file__).resolve().parent / "kernels_src.py"
_spec = importlib.util.spec_from_file_location(f"{__package__}.kernels_src__nb", SRC_PATH)
SRC_MODULE = importlib.util.module_from_spec(_spec)
SRC_MODULE._KERNEL_DECORATOR = _rt.kernel_decorator
_spec.loader.exec_module(SRC_MODULE)

log_p = SRC_MODULE.log_p
exp_p = SRC_MODULE.exp_p
erfc_p = SRC_MODULE.erfc_p
ndtri_p = SRC_MODULE.ndtri_p
ndtri_array_p = SRC_MODULE.ndtri_array_p
nsum_p = SRC_MODULE.nsum_p
ndot_p = SRC_MODULE.ndot_p
mean_p = SRC_MODULE.mean_p
yule_walker_p = SRC_MODULE.yule_walker_p
ar_fit_p = SRC_MODULE.ar_fit_p
ar1_filter_p = SRC_MODULE.ar1_filter_p
garch_filter_p = SRC_MODULE.garch_filter_p
garch_loss_p = SRC_MODULE.garch_loss_p
garch_fit_p = SRC_MODULE.garch_fit_p
garch_block_p = SRC_MODULE.garch_block_p

COMPILE_SECONDS = time.perf_counter() - _T0
_rt.record_compile("order_not_size", COMPILE_SECONDS, len(KERNELS))
REPORT = _rt.report_line
