"""Numba configuration shared by every compiled kernel.

Kernels are compiled eagerly (explicit signatures) for the generic x86-64 CPU, without
Intel SVML and without a threading layer. Compiling for the host CPU could emit FMA or
vector instructions, and SVML replaces the libm transcendentals; either would change the
last bits of the results from one machine to another.
"""

from __future__ import annotations

import os
import sys

PINS = {
    "NUMBA_CPU_NAME": "generic",
    "NUMBA_DISABLE_INTEL_SVML": "1",
    "NUMBA_NUM_THREADS": "1",
}
REQUIRED_VERSIONS = {"numba": "0.67.0", "llvmlite": "0.49.0"}

for _name, _value in PINS.items():
    os.environ[_name] = _value

_PREIMPORTED = "numba" in sys.modules

import llvmlite  # noqa: E402
import numba  # noqa: E402
from numba.core.registry import cpu_target  # noqa: E402


def _verify() -> tuple[str, str]:
    """Check versions and pins; return the (cpu, features) of the active code generator."""
    observed = {"numba": numba.__version__, "llvmlite": llvmlite.__version__}
    if observed != REQUIRED_VERSIONS:
        raise RuntimeError(f"breakdetect requires {REQUIRED_VERSIONS}, found {observed}")
    if _PREIMPORTED:
        numba.config.reload_config()
    config = numba.config
    if config.USING_SVML or not config.DISABLE_INTEL_SVML:
        raise RuntimeError("Intel SVML must be disabled (NUMBA_DISABLE_INTEL_SVML=1)")
    if config.CPU_NAME != "generic" or (config.CPU_FEATURES or "") != "" or config.NUMBA_NUM_THREADS != 1:
        raise RuntimeError("numba was configured before breakdetect could pin the generic CPU target")
    codegen = cpu_target.target_context.codegen()
    target = (str(codegen._get_host_cpu_name()), str(codegen._tm_features))
    if target != ("generic", ""):
        raise RuntimeError(f"numba code generator targets {target}, expected ('generic', '')")
    return target


TARGET = _verify()

F8 = numba.types.float64
I8 = numba.types.int64
A1 = numba.types.Array(F8, 1, "C")
A2 = numba.types.Array(F8, 2, "C")
A3 = numba.types.Array(F8, 3, "C")
I1 = numba.types.Array(I8, 1, "C")
I2 = numba.types.Array(I8, 2, "C")
I3 = numba.types.Array(I8, 3, "C")


def kernel(signature, error_model: str = "python"):
    """Decorator compiling ``signature`` eagerly with the fixed, deterministic flag set."""
    return numba.njit(
        signature,
        cache=False,
        fastmath=False,
        boundscheck=False,
        nogil=False,
        error_model=error_model,
    )
