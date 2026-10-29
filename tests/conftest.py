"""Shared fixtures. Tests that need the model weights are skipped when they are not installed."""

from __future__ import annotations

import os
import sys
from pathlib import Path

import numpy as np
import pytest

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

for _var in ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS", "NUMEXPR_NUM_THREADS"):
    os.environ.setdefault(_var, "1")

from breakdetect.weights import default_weights_dir  # noqa: E402

GOLDEN = ROOT / "tests" / "golden" / "golden20.npz"
SKIP_REASON = (
    "model weights not found: run scripts/download_weights.py or set BREAKDETECT_WEIGHTS to the weights directory"
)


def weights_available() -> bool:
    return (default_weights_dir() / "committee.json").is_file()


requires_weights = pytest.mark.skipif(not weights_available(), reason=SKIP_REASON)


@pytest.fixture(scope="session")
def model():
    if not weights_available():
        pytest.skip(SKIP_REASON)
    import breakdetect

    return breakdetect.load_model()


@pytest.fixture(scope="session")
def golden():
    """The 20 short synthetic series and the scores the original submission produced on them."""
    g = np.load(GOLDEN, allow_pickle=False)
    ho, oo = g["history_offsets"], g["online_offsets"]
    series = [
        (
            g["history"][ho[i] : ho[i + 1]].astype(np.float64),
            g["online"][oo[i] : oo[i + 1]].astype(np.float64),
            g["scores"][oo[i] : oo[i + 1]],
        )
        for i in range(len(ho) - 1)
    ]
    return series
