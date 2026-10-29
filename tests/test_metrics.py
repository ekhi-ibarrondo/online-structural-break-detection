"""TS-AUC on a toy case worked out by hand."""

from __future__ import annotations

import numpy as np
import pytest

from breakdetect.metrics import labels_from_break, points, ts_auc, ts_auc_bruteforce, ts_auc_decomposed

# Four series A, B, C, D (one array per series, one value per online step).
# Step 0: A=0.9 is positive; B=0.2, C=0.5 and D=0.9 are negative.
#         A beats B and C (2 pairs) and ties D (0.5): 2.5 concordant out of 3 pairs.
# Step 1: A=0.4 and B=0.4 are positive, C=0.6 is negative: 0 concordant out of 2 pairs.
# Steps 2 and 3: no negative series left, so no pairs.
# TS-AUC = (2.5 + 0) / (3 + 2) = 0.5, i.e. 50 points.
SCORES = [np.array([0.9, 0.4, 0.7, 0.1]), np.array([0.2, 0.4, 0.3]), np.array([0.5, 0.6]), np.array([0.9])]
LABELS = [np.array([1, 1, 1, 1]), np.array([0, 1, 1]), np.array([0, 0]), np.array([0])]


def test_toy_case_by_hand():
    assert ts_auc(SCORES, LABELS) == pytest.approx(0.5, abs=1e-15)
    assert points(SCORES, LABELS) == pytest.approx(50.0, abs=1e-12)
    d = ts_auc_decomposed(SCORES, LABELS)
    assert d["w"][:2].tolist() == [3.0, 2.0]
    assert d["auc"][0] == pytest.approx(2.5 / 3.0)
    assert d["auc"][1] == 0.0


def test_matches_pair_counting_on_random_data():
    rng = np.random.default_rng(0)
    scores, labels = [], []
    for _ in range(30):
        n = int(rng.integers(5, 40))
        tau = int(rng.integers(-1, n))
        scores.append(np.round(rng.random(n), 1))
        labels.append(labels_from_break(tau, n))
    assert ts_auc(scores, labels) == pytest.approx(ts_auc_bruteforce(scores, labels), abs=1e-12)


def test_labels_from_break():
    assert labels_from_break(None, 3).tolist() == [0, 0, 0]
    assert labels_from_break(-1, 3).tolist() == [0, 0, 0]
    assert labels_from_break(1, 3).tolist() == [0, 1, 1]
    with pytest.raises(ValueError):
        labels_from_break(3, 3)
