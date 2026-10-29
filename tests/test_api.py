"""Public API: weights check, causal streaming, the CrunchDAO entry point and the synthetic generator."""

from __future__ import annotations

import numpy as np
import pytest
from conftest import requires_weights

from breakdetect.synthetic import BREAK_KINDS, generate
from breakdetect.weights import verify_manifest


def test_synthetic_generator_is_seeded_and_float32():
    a = generate(n_series=6, seed=3)
    b = generate(n_series=6, seed=3)
    for s, t in zip(a, b, strict=True):
        assert np.array_equal(s.history, t.history) and np.array_equal(s.online, t.online)
        assert 1000 <= len(s.history) <= 5000 and 10 <= len(s.online) <= 999
        assert np.array_equal(s.history.astype(np.float32).astype(np.float64), s.history)
        assert s.kind in (*BREAK_KINDS, "none")
        assert (s.break_index is None) == (s.kind == "none")


def test_missing_manifest_is_reported(tmp_path):
    with pytest.raises(FileNotFoundError):
        verify_manifest(tmp_path)


def test_corrupted_weights_are_rejected(tmp_path):
    (tmp_path / "a.bin").write_bytes(b"abc")
    (tmp_path / "MANIFEST.sha256").write_text("0" * 64 + "  a.bin\n", encoding="utf-8")
    with pytest.raises(ValueError):
        verify_manifest(tmp_path)


@requires_weights
def test_scores_are_causal_and_one_per_point(model, golden):
    history, online, expected = golden[0]
    consumed = []

    def stream():
        for x in online:
            consumed.append(x)
            yield x

    for k, score in enumerate(model.score_series(history, stream()), start=1):
        assert len(consumed) == k
        assert 0.0 <= score <= 1.0
        assert score == expected[k - 1]
    assert len(consumed) == len(online)


@requires_weights
def test_crunchdao_entry_point(model, golden):
    import main

    gen = main.infer(iter([(h.tolist(), o.tolist()) for h, o, _ in golden[:2]]), None)
    assert next(gen) is None
    scores = np.array(list(gen))
    expected = np.concatenate([e for _, _, e in golden[:2]])
    assert np.array_equal(scores, expected)
    assert main.train(None) is None
