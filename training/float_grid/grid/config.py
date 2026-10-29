"""Model configuration of the float-grid member (the member has no trained weights)."""

import json
from pathlib import Path

from .api import FORMAT, SCORE_NAME, TRANSCENDENTALS
from .constants import constants_hex


def model_config():
    return {
        "format": FORMAT,
        "score": SCORE_NAME,
        "constants": constants_hex(),
        "transcendentals": {name: dict(details) for name, details in TRANSCENDENTALS.items()},
        "description": "Log likelihood ratio of membership in a discrete float grid fitted on the history.",
    }


def write_model_config(path):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(model_config(), sort_keys=True, indent=1) + "\n", encoding="utf-8", newline="\n")
