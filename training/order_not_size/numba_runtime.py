"""numba runtime pins, the ``TRAINING_NUMBA`` switch and the stderr report.

* Environment pins written BEFORE ``import numba``:
  ``NUMBA_CPU_NAME=generic`` (codegen for the generic x86-64 model; numba
  also sets ``CPU_FEATURES=""`` for ``generic``, so no AVX/FMA can be
  emitted and every machine compiles the same code),
  ``NUMBA_DISABLE_INTEL_SVML=1`` (never route ``exp``/``log`` to Intel SVML;
  the process libm is the one CPython uses), ``NUMBA_NUM_THREADS=1`` (no
  threading layer is ever used).  ``TRAINING_NUMBA_CPU_NAME`` overrides the CPU
  name for A/B comparisons and nothing else.
* ``TRAINING_NUMBA`` (env, read once at import): ``1`` (default) = kernels in
  *require* mode -- if numba/llvmlite cannot be imported, their versions are
  not exactly ``REQUIRED_VERSIONS``, ``numba.config.USING_SVML`` is True,
  the CPU-name pin did not take, or the real codegen target
  (``cpu_target.target_context.codegen()``: cpu name + LLVM feature string)
  is not ``generic`` with an empty feature string, importing this module
  raises ``RuntimeError`` instead of silently changing the arithmetic;
  ``0`` = pure-Python reference path, numba not imported at all.
* ``set_enabled(flag)`` / ``enabled()``: a per-call toggle to run both paths
  in the same process.  Callers read ``enabled()`` at series setup time only
  (a series never mixes paths).
* ``report(file=sys.stderr)``: one line with the active path, versions,
  ``numba.core.config.CPU_NAME``, ``llvmlite.binding.get_host_cpu_name()``,
  the real codegen target (``target_cpu`` / ``target_features``), whether
  numba was already imported before the pins (``numba_preimported``),
  ``USING_SVML`` and the eager-compilation time recorded by each kernels
  module (``record_compile``).

This module imports no numpy so that the pins can precede any numba import
in the calling process.  If numba was ALREADY imported when this module
loads, ``numba.config.reload_config()`` re-reads the pinned environment; the
pins still take effect as long as nothing has been compiled yet (the target
machine is created at the first compilation).
"""

from __future__ import annotations

import os
import sys
import time
from collections.abc import Callable
from typing import Any

__all__ = [
    "PINS",
    "REQUIRED_VERSIONS",
    "KernelCompileRecord",
    "codegen_target",
    "compile_records",
    "enabled",
    "kernel_decorator",
    "mode_from_env",
    "record_compile",
    "report",
    "report_line",
    "require_numba",
    "set_enabled",
]

_CPU_NAME = os.environ.get("TRAINING_NUMBA_CPU_NAME", "generic")

# Exact versions the kernels were validated with (requirements.txt pins the
# same pair).  Single source of truth: ``require_numba``
# enforces it and the banner prints the observed versions next to it.
REQUIRED_VERSIONS: dict[str, str] = {"numba": "0.67.0", "llvmlite": "0.49.0"}

# The pins, in the order they are written.  ``NUMBA_CPU_NAME`` is the only
# one with an escape hatch (``TRAINING_NUMBA_CPU_NAME``).
PINS: dict[str, str] = {
    "NUMBA_CPU_NAME": _CPU_NAME,
    "NUMBA_DISABLE_INTEL_SVML": "1",
    "NUMBA_NUM_THREADS": "1",
}

_FALSE_VALUES = ("0", "", "false", "no", "off")


def mode_from_env(value: str | None) -> bool:
    """``TRAINING_NUMBA`` semantics: unset / ``1`` / anything else truthy -> kernels;
    ``0``, empty, ``false``, ``no``, ``off`` (case-insensitive) -> Python."""
    if value is None:
        return True
    return value.strip().lower() not in _FALSE_VALUES


class KernelCompileRecord:
    """Eager-compilation record of one kernels module (name, seconds, #kernels)."""

    __slots__ = ("n_kernels", "name", "seconds")

    def __init__(self, name: str, seconds: float, n_kernels: int) -> None:
        self.name = name
        self.seconds = float(seconds)
        self.n_kernels = int(n_kernels)

    def __repr__(self) -> str:  # pragma: no cover - cosmetic
        return f"KernelCompileRecord({self.name!r}, {self.seconds:.3f}s, {self.n_kernels})"


_records: list[KernelCompileRecord] = []
_numba_mod: Any = None
_enabled: bool = False
_import_error: str | None = None
# Filled by ``require_numba``: (target cpu name, LLVM feature string) of the
# codegen actually used for our kernels, and whether numba had been imported
# before the pins were written.
_target: tuple[str, str] | None = None
_numba_preimported: bool | None = None


def _apply_pins() -> None:
    for k, v in PINS.items():
        os.environ[k] = v


def require_numba() -> Any:
    """Import numba under the pins and verify them; raise ``RuntimeError`` otherwise.

    Idempotent: the imported module is cached.  Called at import when
    ``TRAINING_NUMBA`` is on, and by ``set_enabled(True)``.
    """
    global _numba_mod, _import_error, _target, _numba_preimported
    if _numba_mod is not None:
        return _numba_mod
    _apply_pins()
    already = "numba" in sys.modules
    _numba_preimported = already
    try:
        import llvmlite
        import llvmlite.binding as _llb  # noqa: F401
        import numba as _nb
    except Exception as exc:  # ImportError, OSError from a broken llvmlite, ...
        _import_error = f"{type(exc).__name__}: {exc}"
        raise RuntimeError(
            "TRAINING_NUMBA=1 (require mode) but numba/llvmlite cannot be imported: "
            f"{_import_error}"
        ) from exc
    observed = {"numba": str(_nb.__version__), "llvmlite": str(llvmlite.__version__)}
    if observed != REQUIRED_VERSIONS:
        raise RuntimeError(
            "TRAINING_NUMBA=1 (require mode) but the numba/llvmlite versions are not the validated "
            f"pair: observed {observed} != required {REQUIRED_VERSIONS} (the kernels were "
            "validated bit-exact against these exact wheels; check that pip applied "
            "requirements.txt)"
        )
    if already:
        # Re-read the pinned environment (only effective before any compile).
        _nb.config.reload_config()
    cfg = _nb.config
    if bool(getattr(cfg, "USING_SVML", False)):
        raise RuntimeError(
            "TRAINING_NUMBA=1 (require mode) but numba.config.USING_SVML is True: Intel SVML "
            "would replace the process libm (NUMBA_DISABLE_INTEL_SVML=1 pin did not take)"
        )
    if str(getattr(cfg, "CPU_NAME", None)) != PINS["NUMBA_CPU_NAME"]:
        raise RuntimeError(
            f"TRAINING_NUMBA=1 (require mode) but numba.config.CPU_NAME={cfg.CPU_NAME!r} != "
            f"{PINS['NUMBA_CPU_NAME']!r} (pin did not take: numba imported and configured "
            "before this module?)"
        )
    if PINS["NUMBA_CPU_NAME"] == "generic" and (getattr(cfg, "CPU_FEATURES", None) or "") != "":
        raise RuntimeError(
            f"TRAINING_NUMBA=1 (require mode) but numba.config.CPU_FEATURES={cfg.CPU_FEATURES!r} "
            "is not empty under NUMBA_CPU_NAME=generic (host ISA extensions would leak "
            "into the kernels)"
        )
    target = _codegen_target()
    if PINS["NUMBA_CPU_NAME"] == "generic" and target != ("generic", ""):
        raise RuntimeError(
            "TRAINING_NUMBA=1 (require mode) but the REAL codegen target is "
            f"cpu={target[0]!r} features={target[1]!r} != ('generic', ''): host ISA "
            "extensions would be emitted into the kernels (numba imported/configured "
            f"before the pins? numba_preimported={already})"
        )
    _target = target
    _numba_mod = _nb
    return _nb


def _codegen_target() -> tuple[str, str]:
    """``(cpu_name, feature_string)`` of the codegen numba will use for our kernels.

    Read from the live ``JITCPUCodegen`` (``numba.core.registry.cpu_target
    .target_context.codegen()``; numba 0.67: ``_get_host_cpu_name()`` returns
    ``config.CPU_NAME`` when set, ``_tm_features`` is the feature string the
    target machine was created with — ``''`` under ``NUMBA_CPU_NAME=generic``,
    the full host list under ``host``).  Touching ``target_context`` creates
    the target machine NOW, i.e. under the pins already verified above and
    before any kernel is compiled — the same moment the first eager
    compilation would have created it.  Private attributes: if numba renames
    them the strings say so instead of raising, so a ``generic`` pin then
    FAILS the comparison in ``require_numba`` (a loud failure of the check
    is preferable to a silent pass).
    """
    try:
        from numba.core.registry import cpu_target

        cg = cpu_target.target_context.codegen()
    except Exception as exc:  # pragma: no cover - defensive
        return (f"<codegen error: {type(exc).__name__}: {exc}>", "<unknown>")
    if hasattr(cg, "_get_host_cpu_name"):
        try:
            cpu = str(cg._get_host_cpu_name())
        except Exception as exc:  # pragma: no cover - defensive
            cpu = f"<error: {exc}>"
    else:  # pragma: no cover - numba API drift
        cpu = "<no _get_host_cpu_name>"
    feats = getattr(cg, "_tm_features", None)
    if feats is None:  # pragma: no cover - numba API drift
        feats = "<no _tm_features>"
    return (cpu, str(feats))


def codegen_target() -> tuple[str, str] | None:
    """The ``(cpu, features)`` recorded by ``require_numba`` (None on the Python path)."""
    return _target


def enabled() -> bool:
    """True when the detector wrappers must build kernel state at series setup."""
    return _enabled


def set_enabled(flag: bool) -> bool:
    """Toggle the kernel path for the setups that follow; returns the previous value.

    ``True`` imports/verifies numba (raises ``RuntimeError`` in the same
    cases as require mode); ``False`` never imports anything.
    """
    global _enabled
    prev = _enabled
    if flag:
        require_numba()
    _enabled = bool(flag)
    return prev


def kernel_decorator(signature: Any) -> Callable[[Callable], Any]:
    """``@njit(signature, cache=False, fastmath=False, boundscheck=False, nogil=False)``.

    Explicit signature => eager compilation at decoration time (import of
    the kernels module).  ``parallel`` is never set.  Every kernel of every
    pass goes through this one helper so the flags cannot drift.
    """
    nb = require_numba()
    return nb.njit(signature, cache=False, fastmath=False, boundscheck=False, nogil=False)


def record_compile(name: str, seconds: float, n_kernels: int) -> None:
    """Called by a kernels module after its eager compilation."""
    _records.append(KernelCompileRecord(name, seconds, n_kernels))


def compile_records() -> tuple[KernelCompileRecord, ...]:
    return tuple(_records)


def report_line() -> str:
    """The one-line runtime report (also what ``report()`` prints)."""
    if not _enabled:
        extra = f" (numba import error: {_import_error})" if _import_error else ""
        return f"numba_runtime: path=python TRAINING_NUMBA={os.environ.get('TRAINING_NUMBA', '<unset>')}{extra}"
    nb = _numba_mod
    try:
        import llvmlite
        import llvmlite.binding as llb

        host = llb.get_host_cpu_name()
        llv = llvmlite.__version__
    except Exception as exc:  # pragma: no cover - defensive
        host = f"<error: {exc}>"
        llv = "?"
    comp = ", ".join(f"{r.name}={r.seconds:.2f}s/{r.n_kernels}k" for r in _records) or "none"
    tcpu, tfeat = _target if _target is not None else ("<unchecked>", "<unchecked>")
    return (
        f"numba_runtime: path=numba numba={nb.__version__} llvmlite={llv} "
        f"required={REQUIRED_VERSIONS['numba']}/{REQUIRED_VERSIONS['llvmlite']} "
        f"CPU_NAME={nb.config.CPU_NAME!r} CPU_FEATURES={nb.config.CPU_FEATURES!r} "
        f"target_cpu={tcpu!r} target_features={tfeat!r} "
        f"numba_preimported={_numba_preimported} "
        f"host_cpu={host!r} USING_SVML={bool(nb.config.USING_SVML)} "
        f"NUMBA_NUM_THREADS={os.environ.get('NUMBA_NUM_THREADS')} "
        f"pid={os.getpid()} eager_compile[{comp}]"
    )


def report(file: Any = None) -> str:
    """Print ``report_line()`` to ``file`` (default stderr) and return it."""
    line = report_line()
    print(line, file=file if file is not None else sys.stderr, flush=True)
    return line


# --------------------------------------------------------------------------
# Import-time behaviour: read TRAINING_NUMBA once; require mode raises here.
# --------------------------------------------------------------------------
_T0 = time.perf_counter()
if mode_from_env(os.environ.get("TRAINING_NUMBA")):
    require_numba()
    _enabled = True
IMPORT_SECONDS: float = time.perf_counter() - _T0
