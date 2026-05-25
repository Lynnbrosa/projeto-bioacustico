"""Command-line entry point for the `bioacid` console script."""

from __future__ import annotations

import argparse
import sys
from collections.abc import Sequence

from bioacid import __version__


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="bioacid",
        description="Bioacoustic individual identification pipeline.",
    )
    parser.add_argument(
        "--version",
        action="version",
        version=f"bioacid {__version__}",
    )
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    parser = build_parser()
    parser.parse_args(argv)
    return 0


if __name__ == "__main__":
    sys.exit(main())
