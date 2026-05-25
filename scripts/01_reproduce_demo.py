"""Reproduce the upstream demo on the public Ovenbird sample dataset.

Expects the upstream repo cloned at `external/upstream` and the pre-trained
checkpoint available under `external/upstream/checkpoints/`.

Usage:
    uv run scripts/01_reproduce_demo.py
"""

from __future__ import annotations

import sys
from pathlib import Path

UPSTREAM = Path(__file__).resolve().parents[1] / "external" / "upstream"


def main() -> int:
    if not UPSTREAM.exists():
        print(
            f"Upstream repo not found at {UPSTREAM}. Clone it with:\n"
            "  git clone https://github.com/sammlapp/ovenbird-individual-recognition.git "
            f"{UPSTREAM}",
            file=sys.stderr,
        )
        return 1

    print(f"Found upstream at {UPSTREAM}. (Demo runner not yet implemented; see M1.)")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
