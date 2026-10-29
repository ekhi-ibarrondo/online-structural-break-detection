"""The "Reads 3 others" member: LightGBM stacking that reads three other members."""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np

from ...detectors.traces import TraceSet
from ..order_not_size.columns import ColumnEngine
from ..reads_two_rebuilds.features import SummaryStream, context4
from ..slow_and_fast.features import FeatureEngine
from .forest import Forest

WIDTH = 468


class State:
    """Per-series state of every input block of the head."""

    __slots__ = ("column_state", "context", "engine_state", "features", "summary", "traces")


class ReadsThreeOthers:
    """Forest over [surprises | pattern_removed | summary | traces | context | slow_and_fast features |
    order_not_size columns] (468 inputs)."""

    feature_inputs = ("surprises", "pattern_removed")

    def __init__(self, model_dirs) -> None:
        (model_dir,) = model_dirs
        spec = json.loads((Path(model_dir) / "head.json").read_text())
        if spec.get("width") != WIDTH:
            raise ValueError("head specification")
        names = spec["trace_names"]
        if (
            len(names) != 22
            or len(set(names)) != 22
            or not all(isinstance(n, str) and n.startswith("z_") for n in names)
        ):
            raise ValueError("head trace order")
        self.trace_keys = tuple(n[2:] for n in names)
        self.forest = Forest.load(model_dir)
        if self.forest.n_features != WIDTH:
            raise ValueError("head width")
        self.engine = FeatureEngine()
        self.columns = ColumnEngine()

    def start(self, history) -> State:
        h = np.asarray(history, dtype=np.float64)
        st = State()
        st.summary = SummaryStream(h)
        st.traces = TraceSet(h, self.trace_keys)
        st.context = np.asarray(context4(h), np.float32)
        st.engine_state = self.engine.start(h)
        st.column_state = self.columns.start(h)
        st.features = np.empty(WIDTH, np.float32)
        return st

    def step(self, st: State, x: float, inputs) -> tuple[list[float], None]:
        """Raw forest score of the point."""
        if len(inputs) != 2 or np.shape(inputs[0]) != (182,) or np.shape(inputs[1]) != (110,):
            raise ValueError("shared input shape")
        x = float(x)
        st.features[:182] = inputs[0]
        st.features[182:292] = np.asarray(inputs[1], np.float32)
        st.features[292:328] = np.asarray(st.summary.step(x), np.float32)
        st.features[328:350] = np.asarray(st.traces.step(x), np.float64).astype(np.float32)
        st.features[350:354] = st.context
        st.features[354:450] = self.engine.step_features(st.engine_state, x).astype(np.float32)
        st.features[450:468] = self.columns.step_columns(st.column_state, x).astype(np.float32)
        if not np.isfinite(st.features).all():
            raise ValueError("nonfinite head input")
        return [self.forest.predict_one(st.features)], None
