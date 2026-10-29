"""Committee members. Each factory takes the list of model directories of one member.

A member scorer implements ``start(history) -> state`` and
``step(state, x, inputs) -> (raw scores, feature vector or None)``; ``feature_inputs``
names the earlier members whose feature vectors it reads at the same point.
"""

from __future__ import annotations

from .blind_to_size import BlindToSize
from .broken_or_not import BrokenOrNot
from .float_grid import FloatGrid
from .order_not_size import OrderNotSize
from .pattern_removed import PatternRemoved
from .reads_three_others import ReadsThreeOthers
from .reads_two_rebuilds import ReadsTwoRebuilds
from .slow_and_fast import SlowAndFast
from .surprises import Surprises
from .which_one_broke import WhichOneBroke

FACTORIES = {
    "surprises": Surprises,
    "pattern_removed": PatternRemoved,
    "order_not_size": OrderNotSize,
    "slow_and_fast": SlowAndFast,
    "reads_two_rebuilds": ReadsTwoRebuilds,
    "float_grid": FloatGrid,
    "reads_three_others": ReadsThreeOthers,
    "broken_or_not": BrokenOrNot,
    "which_one_broke": WhichOneBroke,
    "blind_to_size": BlindToSize,
}
