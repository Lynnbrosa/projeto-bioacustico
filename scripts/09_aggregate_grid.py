"""Aggregate per-run ``result.json`` files into the M3 table and figure.

Lets us regenerate ``reports/experiment_table.md`` + ``reports/figures/m3_grid.png``
after fixing a single failing config, without re-running the whole grid.

Usage:
    uv run --extra ml --extra dev scripts/09_aggregate_grid.py
"""

from __future__ import annotations

import sys
from pathlib import Path

from bioacid.reporting import load_grid_results, write_experiment_figure, write_experiment_table

ROOT = Path(__file__).resolve().parents[1]
GRID_DIR = ROOT / "reports" / "runs" / "m3_grid"
TABLE_PATH = ROOT / "reports" / "experiment_table.md"
FIGURE_PATH = ROOT / "reports" / "figures" / "m3_grid.png"


def main() -> int:
    if not GRID_DIR.exists():
        print(f"missing {GRID_DIR}", file=sys.stderr)
        return 1

    runs = load_grid_results(GRID_DIR)
    write_experiment_table(runs, TABLE_PATH)
    print(f"table: {TABLE_PATH.relative_to(ROOT)}")
    write_experiment_figure(runs, FIGURE_PATH)
    print(f"figure: {FIGURE_PATH.relative_to(ROOT)}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
