"""Location and integrity check of the model weights directory."""

from __future__ import annotations

import hashlib
import os
from pathlib import Path

MANIFEST = "MANIFEST.sha256"
ENV_VAR = "BREAKDETECT_WEIGHTS"


def default_weights_dir() -> Path:
    """``$BREAKDETECT_WEIGHTS`` if set, else ``weights/`` next to the repository root."""
    env = os.environ.get(ENV_VAR)
    if env:
        return Path(env)
    return Path(__file__).resolve().parent.parent / "weights"


def sha256_file(path: Path) -> str:
    """Hex sha256 of a file, read in 1 MiB chunks."""
    h = hashlib.sha256()
    with open(path, "rb") as fh:
        for chunk in iter(lambda: fh.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def verify_manifest(weights_dir) -> int:
    """Check the sha256 of every file listed in ``MANIFEST.sha256``; return the number of files."""
    root = Path(weights_dir)
    manifest = root / MANIFEST
    if not manifest.is_file():
        raise FileNotFoundError(f"{manifest} not found: download the weights first (scripts/download_weights.py)")
    n = 0
    for line in manifest.read_text(encoding="utf-8").splitlines():
        if not line.strip():
            continue
        digest, rel = line.split(maxsplit=1)
        path = root / rel
        if not path.is_file():
            raise FileNotFoundError(f"weights file {rel} listed in {MANIFEST} is missing")
        if sha256_file(path) != digest:
            raise ValueError(f"weights file {rel} does not match its sha256 in {MANIFEST}")
        n += 1
    return n
