"""Download the model weights from the GitHub Release and verify them against MANIFEST.sha256.

usage: python scripts/download_weights.py [--dest weights] [--url URL] [--force]
"""

from __future__ import annotations

import argparse
import hashlib
import shutil
import sys
import tarfile
import tempfile
import urllib.request
from pathlib import Path

RELEASE_URL = "https://github.com/ekhi-ibarrondo/online-structural-break-detection/releases/download/v1.0.0"
ARCHIVE = "breakdetect-weights.tar"
ARCHIVE_SHA256 = "af77b2898a16b37ac92e36b230e1c898f51952ea8510171bdc9154e8a7d86c15"
MANIFEST = "MANIFEST.sha256"


def sha256_file(path: Path) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as fh:
        for chunk in iter(lambda: fh.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def verify(dest: Path) -> int:
    """Check every file listed in the manifest; return the number of files checked."""
    n = 0
    for line in (dest / MANIFEST).read_text(encoding="utf-8").splitlines():
        if not line.strip():
            continue
        digest, rel = line.split(maxsplit=1)
        path = dest / rel
        if not path.is_file() or sha256_file(path) != digest:
            raise SystemExit(f"weights file {rel} is missing or corrupted")
        n += 1
    return n


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--dest", type=Path, default=Path(__file__).resolve().parent.parent / "weights")
    parser.add_argument("--url", default=f"{RELEASE_URL}/{ARCHIVE}")
    parser.add_argument("--force", action="store_true", help="replace an existing destination directory")
    args = parser.parse_args()
    if args.dest.exists() and not args.force:
        sys.exit(f"{args.dest} already exists (use --force to replace it)")
    with tempfile.TemporaryDirectory() as tmp:
        archive = Path(tmp) / ARCHIVE
        print(f"downloading {args.url}")
        with urllib.request.urlopen(args.url) as response, open(archive, "wb") as fh:
            shutil.copyfileobj(response, fh)
        if sha256_file(archive) != ARCHIVE_SHA256:
            sys.exit(f"{ARCHIVE} does not match its published sha256: the download is incomplete or altered")
        staging = Path(tmp) / "weights"
        with tarfile.open(archive) as tar:
            tar.extractall(staging, filter="data")
        n = verify(staging)
        if args.dest.exists():
            shutil.rmtree(args.dest)
        shutil.move(str(staging), args.dest)
    print(f"{n} files verified in {args.dest}")


if __name__ == "__main__":
    main()
