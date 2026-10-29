"""The "Blind to size" member: volatility-scaled pre-training."""

from __future__ import annotations

import json
from pathlib import Path

from ..sequence_net import SequenceNet


class BlindToSize(SequenceNet):
    """Inputs are divided by a strictly past EWMA volatility (half-life 50) before the network."""

    def __init__(self, model_dirs) -> None:
        (model_dir,) = model_dirs
        root = Path(model_dir)
        config = json.loads((root / "network.json").read_text(encoding="utf-8"))
        super().__init__(
            root / "weights.npz", config["offset_b"], root / "fallback.npy", scale_blind=True, strict=False
        )
