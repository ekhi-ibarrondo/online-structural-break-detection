"""The "Reads 2 rebuilds" member: supervised GRU on 354 features from surprises, pattern_removed and detectors."""

from __future__ import annotations

import json
import os

import numpy as np

from ...detectors.traces import TraceSet
from ...kernels import reads_two_rebuilds as kernels
from .features import SummaryStream, context4

LEARNER_FORMAT = "reads_two_rebuilds/v1"
SOURCES = ("surprises", "pattern_removed", "summary", "traces", "context")
WIDTHS = {"surprises": 182, "pattern_removed": 110, "summary": 36, "traces": 22, "context": 4}
N_FEATURES = 354
HIDDEN = 128
HEAD = 64
NET_KEYS = ("Wih0", "Whh0", "bih0", "bhh0", "Wih1", "Whh1", "bih1", "bhh1", "W1", "b1", "W2", "b2")
_EXPECTED_NAMES = [f"{nm}_{j}" for nm in SOURCES for j in range(WIDTHS[nm])]


def load_learner(path: str) -> dict:
    """Weights, input median/scale and column names of one bagged network."""
    z = np.load(path, allow_pickle=False)
    meta = json.loads(str(z["meta_json"]))
    if meta.get("format") != LEARNER_FORMAT:
        raise ValueError(f"reads_two_rebuilds learner {path}: unexpected format {meta.get('format')!r}")
    P = {v: np.ascontiguousarray(z[v], dtype=np.float64) for v in NET_KEYS}
    out = dict(
        P=P,
        med=np.asarray(z["med"], np.float64),
        sc=np.asarray(z["sc"], np.float64),
        names=[str(x) for x in z["names"]],
        meta=meta,
    )
    if not (len(out["names"]) == meta["n_features"] == P["Wih0"].shape[1]):
        raise ValueError(f"reads_two_rebuilds learner {path}: feature count mismatch")
    return out


class _Net:
    """Weights of one learner, transposed for the compiled forward pass."""

    __slots__ = ("WT", "b", "b2", "w2")

    def __init__(self, P: dict) -> None:
        self.WT = [np.ascontiguousarray(P[k].T) for k in ("Wih0", "Whh0", "Wih1", "Whh1", "W1")]
        self.b = [P[k] for k in ("bih0", "bhh0", "bih1", "bhh1", "b1")]
        self.w2 = np.ascontiguousarray(P["W2"][0])
        self.b2 = float(P["b2"][0])


class State:
    """Per-series state: summary statistics, detector traces, history context and GRU hidden states."""

    __slots__ = ("a", "context32", "gh", "gi", "h1", "h2", "summary", "traces")

    def __init__(self, summary, traces, context32, n_bags: int) -> None:
        self.summary = summary
        self.traces = traces
        self.context32 = context32
        self.h1 = [np.zeros(HIDDEN) for _ in range(n_bags)]
        self.h2 = [np.zeros(HIDDEN) for _ in range(n_bags)]
        self.gi = [np.empty(3 * HIDDEN) for _ in range(n_bags)]
        self.gh = [np.empty(3 * HIDDEN) for _ in range(n_bags)]
        self.a = [np.empty(HEAD) for _ in range(n_bags)]


class ReadsTwoRebuilds:
    """Three bagged GRU networks; their logits are averaged before the CDF map."""

    feature_inputs = ("surprises", "pattern_removed")

    def __init__(self, model_dirs) -> None:
        (model_dir,) = model_dirs
        with open(os.path.join(model_dir, "head.json"), encoding="utf-8") as fh:
            spec = json.load(fh)
        if tuple(spec.get("sources", ())) != SOURCES or spec.get("quant") != "fp16":
            raise ValueError(f"reads_two_rebuilds head spec malformed: {spec!r}")
        names = list(spec["trace_names"])
        if len(names) != WIDTHS["traces"] or not all(isinstance(n, str) and n.startswith("z_") for n in names):
            raise ValueError("reads_two_rebuilds head: 22 trace names 'z_*' expected")
        self.trace_keys = [nm[2:] for nm in names]
        learners = [load_learner(os.path.join(model_dir, fn)) for fn in spec["bags"]]
        if len(learners) != 3:
            raise ValueError("reads_two_rebuilds head: exactly 3 bags expected")
        for learner in learners:
            if learner["names"] != _EXPECTED_NAMES:
                raise ValueError("reads_two_rebuilds learner: column names do not follow the input layout")
        for learner in learners[1:]:
            if not (
                np.array_equal(learner["med"], learners[0]["med"]) and np.array_equal(learner["sc"], learners[0]["sc"])
            ):
                raise ValueError("reads_two_rebuilds head: med/sc differ between bags")
        self.med = learners[0]["med"]
        self.sc = learners[0]["sc"]
        if self.med.shape != (N_FEATURES,) or not (
            np.isfinite(self.med).all() and np.isfinite(self.sc).all() and (self.sc > 0).all()
        ):
            raise ValueError("reads_two_rebuilds head: med/sc malformed")
        self.nets = tuple(_Net(learner["P"]) for learner in learners)
        self.n_bags = len(self.nets)

    def start(self, history) -> State:
        h = np.asarray(history, dtype=np.float64)
        summary = SummaryStream(h)
        traces = TraceSet(h, self.trace_keys)
        context32 = np.array(context4(h), np.float64).astype(np.float32)
        return State(summary, traces, context32, self.n_bags)

    def step(self, st: State, x: float, inputs) -> tuple[list[float], None]:
        """Mean logit of the bags; the input vector is quantized to float16 as in training."""
        xf = float(x)
        surprises, pattern_removed = inputs
        x32 = np.empty(N_FEATURES, dtype=np.float32)
        x32[0:182] = surprises
        x32[182:292] = pattern_removed.astype(np.float32)
        x32[292:328] = np.array(st.summary.step(xf)).astype(np.float32)
        x32[328:350] = np.array(st.traces.step(xf), np.float64).astype(np.float32)
        x32[350:354] = st.context32
        x64 = np.ascontiguousarray(x32, dtype=np.float64)
        z = np.empty(N_FEATURES)
        kernels.transform(x64, self.med, self.sc, z)
        xq = z.astype(np.float32).astype(np.float16).astype(np.float64)
        acc = 0.0
        for b, net in enumerate(self.nets):
            WT, bb = net.WT, net.b
            acc += kernels.net_step(
                xq,
                st.h1[b],
                st.h2[b],
                WT[0],
                bb[0],
                WT[1],
                bb[1],
                WT[2],
                bb[2],
                WT[3],
                bb[3],
                WT[4],
                bb[4],
                net.w2,
                net.b2,
                st.gi[b],
                st.gh[b],
                st.a[b],
            )
        return [acc / self.n_bags], None
