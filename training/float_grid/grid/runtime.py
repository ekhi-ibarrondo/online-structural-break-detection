"""Verify the compilation target before compiling any numerical functions."""

import os
import sys

PREIMPORTED = "numba" in sys.modules
CPU_PIN = os.environ.get("TRAINING_NUMBA_CPU_NAME", "generic")
if not PREIMPORTED:
    os.environ["NUMBA_CPU_NAME"] = CPU_PIN
    os.environ["NUMBA_DISABLE_INTEL_SVML"] = "1"
    os.environ["NUMBA_NUM_THREADS"] = "1"

import llvmlite  # noqa: E402
import numba  # noqa: E402
from llvmlite import binding  # noqa: E402
from numba.core.registry import cpu_target  # noqa: E402


def verify_runtime():
    """Fail closed, including when another module initialized numba first."""
    if numba.__version__ != "0.67.0" or llvmlite.__version__ != "0.49.0":
        raise RuntimeError("Requires numba 0.67.0 and llvmlite 0.49.0")
    config = numba.config
    if config.USING_SVML or not config.DISABLE_INTEL_SVML:
        raise RuntimeError("SVML must be disabled")
    # config only: numba.get_num_threads() would launch the threading layer in the pre-fork parent
    if config.CPU_NAME != CPU_PIN or config.NUMBA_NUM_THREADS != 1:
        raise RuntimeError("Compilation target or thread count does not match the runtime pins")
    codegen = cpu_target.target_context.codegen()
    actual = (codegen._get_host_cpu_name(), codegen._tm_features)
    if CPU_PIN == "generic" and (config.CPU_FEATURES not in (None, "") or actual != ("generic", "")):
        raise RuntimeError("The generic target requires empty CPU features")
    return actual


ACTUAL_TARGET = verify_runtime()


def compiled(signature):
    """The single decorator policy for all numerical functions."""
    return numba.njit(signature, cache=False, fastmath=False, boundscheck=False, nogil=False, error_model="numpy")


def runtime_report():
    import platform

    import numpy as np

    from .kernels import COMPILATION_SECONDS

    return (
        f"python={platform.python_version()} numpy={np.__version__} "
        f"numba={numba.__version__} llvmlite={llvmlite.__version__} "
        f"CPU_NAME={numba.config.CPU_NAME!r} CPU_FEATURES={numba.config.CPU_FEATURES or ''!r} "
        f"target={ACTUAL_TARGET!r} USING_SVML={numba.config.USING_SVML} "
        f"host_cpu={binding.get_host_cpu_name()} preimported={PREIMPORTED} "
        f"threads={numba.config.NUMBA_NUM_THREADS} eager_compilation_seconds={COMPILATION_SECONDS:.9f}"
    )
