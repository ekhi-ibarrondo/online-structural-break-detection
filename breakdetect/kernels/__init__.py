"""Numba kernels. Importing this package applies the numba pins before any kernel is compiled."""

from . import runtime

__all__ = ["runtime"]
