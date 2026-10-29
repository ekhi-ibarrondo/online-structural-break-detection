"""The "Broken or not?" member: causal GRU pre-trained on simulated breaks, pointwise loss."""

from __future__ import annotations

import json
from pathlib import Path

from ..sequence_net import SequenceNet


class BrokenOrNot(SequenceNet):
    """A degenerate history or point drops the member for the rest of the series."""

    def __init__(self, model_dirs) -> None:
        (model_dir,) = model_dirs
        root = Path(model_dir)
        config = json.loads((root / "network.json").read_text(encoding="utf-8"))
        super().__init__(
            root / "weights.npz", config["offset_b"], root / "fallback.npy", scale_blind=False, strict=True
        )
