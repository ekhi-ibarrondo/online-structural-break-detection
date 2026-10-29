"""Online structural-break detection: a committee of ten members, one score per online point.

>>> import breakdetect
>>> model = breakdetect.load_model("weights")
>>> scores = model.score(history, online)            # or: for s in model.score_series(...)

Scores are bit-for-bit reproducible on x86-64 Linux with Python 3.12 and the pinned
numpy/numba versions; loading the model limits BLAS to one thread because multi-threaded
reductions change the order (and thus the rounding) of floating-point sums.
"""

from __future__ import annotations

from .weights import default_weights_dir, verify_manifest

__all__ = ["load_model"]


def load_model(weights_dir=None, verify: bool = True):
    """Load the committee (default directory: ``$BREAKDETECT_WEIGHTS``, else the repository's ``weights/``)."""
    from threadpoolctl import threadpool_limits

    from .committee import Committee

    root = default_weights_dir() if weights_dir is None else weights_dir
    if verify:
        verify_manifest(root)
    threadpool_limits(limits=1)
    return Committee.load(root)
