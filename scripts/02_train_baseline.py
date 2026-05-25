"""M2 / M3: Train a single backbone+loss baseline on the sample.

Runs one configuration of :func:`bioacid.experiment.run_experiment` and prints
the clustering metrics. Defaults: ResNet18 + cross-entropy, 10 epochs.

For the full M3 grid (multiple backbones and losses), see
``scripts/04_run_experiments.py``.

Usage:
    uv run --extra ml --extra dev scripts/02_train_baseline.py
"""

from __future__ import annotations

import sys
from pathlib import Path

from bioacid.evaluate import format_metrics
from bioacid.experiment import run_experiment
from bioacid.train import TrainConfig

ROOT = Path(__file__).resolve().parents[1]
UPSTREAM = ROOT / "external" / "upstream"
CSV_PATH = UPSTREAM / "sample_data" / "labeled_clips_sample.csv"
RUN_DIR = ROOT / "reports" / "runs" / "m2_baseline"


def main() -> int:
    if not UPSTREAM.exists():
        print(f"upstream not found at {UPSTREAM}; clone first", file=sys.stderr)
        return 1

    config = TrainConfig(
        backbone="resnet18",
        loss="cross_entropy",
        epochs=10,
        batch_size=32,
        lr=1e-3,
    )

    result = run_experiment(
        config,
        train_csv=CSV_PATH,
        audio_root=UPSTREAM,
        output_dir=RUN_DIR,
        device="cpu",
    )

    print()
    print(format_metrics(result.metrics))
    print(f"\ntrain {result.train_seconds:.1f}s · embed {result.embed_seconds:.1f}s")
    print(f"artifacts: {RUN_DIR.relative_to(ROOT)}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
