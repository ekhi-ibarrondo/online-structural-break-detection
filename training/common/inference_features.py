"""Bridge to the inference package for the streaming inputs of reads_two_rebuilds and reads_three_others.

Both members are trained on the per-point features that the committee computes
at inference time.  Most of them have a training copy in this folder
(surprises ``surprises/features.py``, pattern_removed
``pattern_removed/features.py``, slow_and_fast ``slow_and_fast/features.py``,
order_not_size ``order_not_size/scorer.py``; each is identical, statement for
statement, to its inference counterpart).  The remaining inputs -- the 36
summary features, the 22 detector traces and the 4 per-series context values
-- live only in the inference package, together with the detector library they
need.  They are imported from there (``pip install -e .`` at the repository
root), so training and inference share one implementation:
``SummaryStream`` and ``context4`` from ``SUMMARY_MODULE``, ``TraceSet`` from
``TRACE_MODULE``.
"""
from __future__ import annotations

import importlib

SUMMARY_MODULE = "breakdetect.members.reads_two_rebuilds.features"
TRACE_MODULE = "breakdetect.detectors.traces"


def _import(name: str):
    try:
        return importlib.import_module(name)
    except ImportError as exc:
        raise ImportError(
            f"cannot import {name!r}: install the inference package (pip install -e . at the repository root)"
        ) from exc


class StreamInputs:
    """Constructors of the per-series summary stream, trace stream and context values."""

    def __init__(self):
        summary_mod, trace_mod = _import(SUMMARY_MODULE), _import(TRACE_MODULE)
        self.summary = summary_mod.SummaryStream
        self.traces = trace_mod.TraceSet
        self.context = summary_mod.context4
