"""Committee fusion: members' probabilities averaged in logit space with integer votes.

At online step t every member turns its raw score(s) into a probability with its per-step
CDF maps (mean over its maps). The committee takes the log-odds of each probability,
repeats it ``multiplicity`` times, averages the list with a left fold in member order and
maps the mean back with a logistic function, clipped to [1e-9, 1 - 1e-9] and rounded to
float32. A member that raises or returns a value outside [0, 1] is dropped for the rest of
the series (and so is any member that reads its features); the remaining votes still form
the mean. Every series is processed on its own: no state is shared between series and the
length of the online segment is never used.
"""

from __future__ import annotations

import json
from collections.abc import Iterable, Iterator
from pathlib import Path

import numpy as np

from .calibration import load_map
from .members import FACTORIES

MAX_ONLINE_POINTS = 4096


class Member:
    """A loaded member: its scorer, its maps, the raw score feeding each map and its votes."""

    __slots__ = ("inputs", "map_inputs", "maps", "multiplicity", "name", "scorer")

    def __init__(self, name, scorer, maps, map_inputs, multiplicity, inputs) -> None:
        self.name = name
        self.scorer = scorer
        self.maps = maps
        self.map_inputs = map_inputs
        self.multiplicity = multiplicity
        self.inputs = inputs


def _member_logit(v: float) -> np.float64:
    """Log-odds of a member probability clipped to [1e-6, 1 - 1e-6]."""
    p = np.clip(np.float64(v), 1e-6, 1 - 1e-6)
    return np.log(p / (1.0 - p))


def _fuse(votes: list[np.float64]) -> float:
    """Left-fold mean of the votes, then a branch-stable logistic in numpy float64."""
    acc = np.float64(votes[0])
    for v in votes[1:]:
        acc += v
    raw = np.asarray([acc / float(len(votes))], dtype=np.float64)
    out = np.empty_like(raw, dtype=np.float64)
    pos = raw >= 0
    out[pos] = 1.0 / (1.0 + np.exp(-raw[pos]))
    ex = np.exp(raw[~pos])
    out[~pos] = ex / (1.0 + ex)
    np.clip(out, 1e-9, 1.0 - 1e-9, out=out)
    return float(out.astype(np.float32).astype(np.float64)[0])


def _as_float(x) -> float:
    """The point as a Python float; a size-1 array gives its value, anything else NaN."""
    try:
        return float(x)
    except MemoryError:
        raise
    except Exception:
        try:
            xa = np.asarray(x, dtype=np.float64).reshape(-1)
            return float(xa[0]) if xa.size == 1 else float("nan")
        except MemoryError:
            raise
        except Exception:
            return float("nan")


class Committee:
    """The full model. ``score_series`` yields one score in [0, 1] per online point."""

    def __init__(self, members: list[Member]) -> None:
        self.members = members

    @classmethod
    def load(cls, weights_dir) -> Committee:
        """Build every member and map listed in ``committee.json``."""
        root = Path(weights_dir)
        config = json.loads((root / "committee.json").read_text(encoding="utf-8"))
        if config.get("format") != "breakdetect-committee/1":
            raise ValueError("unknown committee format")
        members: list[Member] = []
        position: dict[str, int] = {}
        for spec in config["members"]:
            name = spec["name"]
            scorer = FACTORIES[name]([root / d for d in spec["models"]])
            inputs = tuple(position[n] for n in scorer.feature_inputs)
            maps = tuple(load_map(root / p) for p in spec["maps"])
            multiplicity = int(spec["multiplicity"])
            if multiplicity < 1 or len(maps) != len(spec["map_inputs"]):
                raise ValueError(f"member {name}: invalid multiplicity or maps")
            position[name] = len(members)
            members.append(Member(name, scorer, maps, tuple(spec["map_inputs"]), multiplicity, inputs))
        return cls(members)

    def score_series(self, history: Iterable[float], online: Iterable[float]) -> Iterator[float]:
        """Score each online point after seeing only the history and the points before it."""
        members = self.members
        n_members = len(members)
        alive = [False] * n_members
        states: list = [None] * n_members
        feats: list = [None] * n_members
        try:
            x_hist = np.asarray(history, dtype=np.float64)
        except MemoryError:
            raise
        except Exception:
            x_hist = None
        for m in range(n_members if x_hist is not None else 0):
            me = members[m]
            if me.inputs and not all(alive[i] for i in me.inputs):
                continue
            try:
                states[m] = me.scorer.start(x_hist)
                alive[m] = True
            except MemoryError:
                raise
            except Exception:
                states[m] = None
        n_points = 0
        for x in online:
            n_points += 1
            if n_points > MAX_ONLINE_POINTS:
                score = 0.5
            else:
                t = n_points - 1
                xf = _as_float(x)
                votes: list[np.float64] = []
                for m in range(n_members):
                    if not alive[m]:
                        continue
                    me = members[m]
                    if me.inputs and not all(alive[i] for i in me.inputs):
                        alive[m] = False
                        states[m] = None
                        continue
                    try:
                        raw, feats[m] = me.scorer.step(states[m], xf, tuple(feats[i] for i in me.inputs))
                        maps = me.maps
                        idx = me.map_inputs
                        # Builtin sum: compensated (Neumaier) since Python 3.12, so the rounding is version-specific.
                        v = sum(maps[q].evaluate(t, raw[idx[q]]) for q in range(len(maps))) / float(len(maps))
                    except MemoryError:
                        raise
                    except Exception:
                        alive[m] = False
                        states[m] = None
                        continue
                    if not (0.0 <= v <= 1.0):
                        alive[m] = False
                        states[m] = None
                        continue
                    votes.extend([_member_logit(v)] * me.multiplicity)
                score = _fuse(votes) if votes else 0.5
            if not (0.0 <= score <= 1.0):
                score = 0.5
            yield score

    def score(self, history: Iterable[float], online: Iterable[float]) -> np.ndarray:
        """All scores of a series as a float64 array."""
        return np.fromiter(self.score_series(history, online), dtype=np.float64)
