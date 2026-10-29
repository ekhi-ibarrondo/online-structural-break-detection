"""Scores do not depend on the process, on the order of the series or on earlier series."""

from __future__ import annotations

import subprocess
import sys

import numpy as np
from conftest import ROOT, requires_weights

_SCRIPT = """
import sys
sys.path.insert(0, {root!r})
import numpy as np
import breakdetect
from breakdetect.synthetic import generate
model = breakdetect.load_model()
out = [model.score(s.history, s.online) for s in generate(n_series=3, seed=5, online_range=(20, 60))]
np.save(sys.argv[1], np.concatenate(out))
"""


@requires_weights
def test_two_fresh_processes_give_identical_scores(tmp_path):
    outputs = []
    for k in range(2):
        path = tmp_path / f"run{k}.npy"
        subprocess.run([sys.executable, "-c", _SCRIPT.format(root=str(ROOT)), str(path)], check=True, timeout=1800)
        outputs.append(np.load(path))
    assert np.array_equal(outputs[0], outputs[1])


@requires_weights
def test_no_state_carries_over_between_series(model, golden):
    history, online, expected = golden[3]
    first = model.score(history, online)
    other_history, other_online, _ = golden[7]
    model.score(other_history, other_online)
    again = model.score(history, online)
    assert np.array_equal(first, again)
    assert np.array_equal(first, expected)
