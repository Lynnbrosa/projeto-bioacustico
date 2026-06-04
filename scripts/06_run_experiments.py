"""M3: Run the experiment grid and write per-run artifacts.

Cartesian product of ``BACKBONES x LOSSES`` (the original M3 deliverable)
plus ablations from the M3 winner (SpecAugment, SupCon, ConvNeXt-Tiny).

Each run writes ``backbone.pth`` + ``embeddings.npy`` + ``result.json`` under
``reports/runs/m3_grid/<exp_id>/``. The aggregate table + figure are written
via :mod:`bioacid.reporting` so the layout is the same as
``scripts/09_aggregate_grid.py``.

Usage:
    uv run --extra ml --extra dev scripts/06_run_experiments.py
"""

from __future__ import annotations

import sys
import time
from pathlib import Path

from bioacid.experiment import run_experiment
from bioacid.losses import LossName
from bioacid.models import BackboneName
from bioacid.reporting import load_grid_results, write_experiment_figure, write_experiment_table
from bioacid.train import TrainConfig

ROOT = Path(__file__).resolve().parents[1]
UPSTREAM = ROOT / "external" / "upstream"
CSV_PATH = UPSTREAM / "sample_data" / "labeled_clips_sample.csv"
GRID_DIR = ROOT / "reports" / "runs" / "m3_grid"
TABLE_PATH = ROOT / "reports" / "experiment_table.md"
FIGURE_PATH = ROOT / "reports" / "figures" / "m3_grid.png"

EPOCHS = 10
BATCH_SIZE = 32
LR = 1e-3

BACKBONES: list[BackboneName] = ["resnet18", "resnet50", "efficientnet_b0"]
LOSSES: list[LossName] = ["cross_entropy", "arcface"]


def grid() -> list[tuple[str, TrainConfig]]:
    """All configurations to run, in display order."""

    def cfg(**overrides: object) -> TrainConfig:
        base: dict[str, object] = {"epochs": EPOCHS, "batch_size": BATCH_SIZE, "lr": LR}
        return TrainConfig(**base, **overrides)  # type: ignore[arg-type]

    cartesian = [
        (f"{backbone}__{loss}", cfg(backbone=backbone, loss=loss))
        for backbone in BACKBONES
        for loss in LOSSES
    ]
    ablations = [
        ("resnet18__cross_entropy__specaug", cfg(spec_augment=True)),
        ("resnet18__supcon", cfg(loss="supcon")),
        ("convnext_tiny__cross_entropy", cfg(backbone="convnext_tiny")),
    ]
    return cartesian + ablations


def main() -> int:
    if not UPSTREAM.exists():
        print(f"upstream not found at {UPSTREAM}; clone first", file=sys.stderr)
        return 1

    GRID_DIR.mkdir(parents=True, exist_ok=True)
    configurations = grid()

    t_total = time.time()
    for exp_id, config in configurations:
        print(f"\n=== {exp_id} ===")
        try:
            run_experiment(
                config,
                train_csv=CSV_PATH,
                audio_root=UPSTREAM,
                output_dir=GRID_DIR / exp_id,
                device="cpu",
            )
        except Exception as exc:
            print(f"FAILED {exp_id}: {exc}", file=sys.stderr)
            (GRID_DIR / exp_id).mkdir(parents=True, exist_ok=True)
            (GRID_DIR / exp_id / "error.txt").write_text(repr(exc))

    print(f"\n=== grid finished in {time.time() - t_total:.1f}s ===")

    runs = load_grid_results(GRID_DIR)
    write_experiment_table(runs, TABLE_PATH)
    print(f"table: {TABLE_PATH.relative_to(ROOT)}")
    write_experiment_figure(runs, FIGURE_PATH)
    print(f"figure: {FIGURE_PATH.relative_to(ROOT)}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
