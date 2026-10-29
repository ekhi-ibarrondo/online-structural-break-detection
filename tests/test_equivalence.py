"""The package reproduces, bit for bit, the scores of the submitted model on the golden series.

The bits were checked on x86-64 Linux without AVX-512. Elsewhere the scores may differ in the
last digits: Windows and macOS have another maths library behind ``log``, ``exp`` and similar
functions, and numpy has separate AVX-512 versions of them. There the test only checks that
the scores agree to ``ATOL`` (on Windows the largest difference on these series is about 1e-5).
"""

from __future__ import annotations

import platform
import sys

import numpy as np
from conftest import requires_weights


def _has_avx512() -> bool:
    try:
        from numpy._core._multiarray_umath import __cpu_features__
    except ImportError:
        return False
    return bool(__cpu_features__.get("AVX512F"))


EXACT = sys.platform.startswith("linux") and platform.machine().lower() in ("x86_64", "amd64") and not _has_avx512()
ATOL = 1e-4


@requires_weights
def test_golden_scores_are_reproduced_exactly(model, golden):
    for i, (history, online, expected) in enumerate(golden):
        scores = model.score(history, online)
        assert scores.dtype == np.float64
        assert scores.shape == expected.shape, f"series {i}"
        diff = np.max(np.abs(scores - expected))
        if EXACT:
            assert np.array_equal(scores, expected), f"series {i}: max |diff| = {diff}"
        else:
            assert diff <= ATOL, f"series {i}: max |diff| = {diff}"
