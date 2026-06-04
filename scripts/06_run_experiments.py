"""M3: Run the experiment grid and aggregate results.

Two phases:

1. **Baseline grid** — Cartesian product of ``BACKBONES x LOSSES`` (the
   original M3 deliverable).
2. **Ablation grid** — incremental modifications to the M3 winner
   (ResNet18 + CE): SpecAugment on, SupCon, ConvNeXt-Tiny.

Each run writes per-config artifacts (``backbone.pth``, ``embeddings.npy``,
``result.json``) under ``reports/runs/m3_grid/<exp_id>/``. The aggregate
table and figure go to ``reports/experiment_table.md`` and
``reports/figures/m3_grid.png``.

Usage:
    uv run --extra ml --extra dev scripts/06_run_experiments.py
"""

from __future__ import annotations

import sys
import time
from dataclasses import dataclass
from pathlib import Path

from bioacid.experiment import ExperimentResult, run_experiment
from bioacid.losses import LossName
from bioacid.models import BackboneName
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


@dataclass
class GridEntry:
    """A single configuration to run, with a human-friendly ID."""

    exp_id: str
    config: TrainConfig


def baseline_grid() -> list[GridEntry]:
    """Original M3 grid: backbone x loss."""
    return [
        GridEntry(
            exp_id=f"{backbone}__{loss}",
            config=TrainConfig(
                backbone=backbone, loss=loss, epochs=EPOCHS, batch_size=BATCH_SIZE, lr=LR
            ),
        )
        for backbone in BACKBONES
        for loss in LOSSES
    ]


def ablation_grid() -> list[GridEntry]:
    """Incremental ablations from the M3 winner (ResNet18 + CE)."""
    return [
        GridEntry(
            exp_id="resnet18__cross_entropy__specaug",
            config=TrainConfig(
                backbone="resnet18",
                loss="cross_entropy",
                epochs=EPOCHS,
                batch_size=BATCH_SIZE,
                lr=LR,
                spec_augment=True,
            ),
        ),
        GridEntry(
            exp_id="resnet18__supcon",
            config=TrainConfig(
                backbone="resnet18",
                loss="supcon",
                epochs=EPOCHS,
                batch_size=BATCH_SIZE,
                lr=LR,
            ),
        ),
        GridEntry(
            exp_id="convnext_tiny__cross_entropy",
            config=TrainConfig(
                backbone="convnext_tiny",
                loss="cross_entropy",
                epochs=EPOCHS,
                batch_size=BATCH_SIZE,
                lr=LR,
            ),
        ),
    ]


def main() -> int:
    if not UPSTREAM.exists():
        print(f"upstream not found at {UPSTREAM}; clone first", file=sys.stderr)
        return 1

    GRID_DIR.mkdir(parents=True, exist_ok=True)
    grid = baseline_grid() + ablation_grid()
    results_by_id: dict[str, ExperimentResult] = {}

    t_total = time.time()
    for entry in grid:
        print(f"\n=== {entry.exp_id} ===")
        try:
            result = run_experiment(
                entry.config,
                train_csv=CSV_PATH,
                audio_root=UPSTREAM,
                output_dir=GRID_DIR / entry.exp_id,
                device="cpu",
            )
            results_by_id[entry.exp_id] = result
        except Exception as exc:
            print(f"FAILED {entry.exp_id}: {exc}", file=sys.stderr)
            (GRID_DIR / entry.exp_id).mkdir(parents=True, exist_ok=True)
            (GRID_DIR / entry.exp_id / "error.txt").write_text(repr(exc))

    print(f"\n=== grid finished in {time.time() - t_total:.1f}s ===")
    _write_table(grid, results_by_id)
    _write_figure(grid, results_by_id)
    return 0


def _write_table(grid: list[GridEntry], results: dict[str, ExperimentResult]) -> None:
    header = (
        "| Experiment | Backbone | Loss | SpecAug | ARI | NMI | FMI | Hungarian | "
        "Purity | Train top-1 | Train (s) |\n"
        "| --- | --- | --- | :-: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |\n"
    )
    rows: list[str] = []
    for entry in grid:
        result = results.get(entry.exp_id)
        if result is None:
            rows.append(
                f"| {entry.exp_id} | {entry.config.backbone} | {entry.config.loss} | "
                f"{'yes' if entry.config.spec_augment else 'no'} | "
                "FAILED | FAILED | FAILED | FAILED | FAILED | - | - |\n"
            )
            continue
        m = result.metrics
        rows.append(
            f"| {entry.exp_id} | {result.config.backbone} | {result.config.loss} | "
            f"{'yes' if result.config.spec_augment else 'no'} | "
            f"{m.ari:.3f} | {m.nmi:.3f} | {m.fmi:.3f} | "
            f"{m.hungarian_accuracy:.3f} | {m.purity:.3f} | "
            f"{result.final_train_top1:.3f} | {result.train_seconds:.0f} |\n"
        )

    body = (
        "# M3 — Matriz de experimentos (baseline + ablações)\n\n"
        f"Grid: `{len(grid)}` runs, {EPOCHS} épocas, batch {BATCH_SIZE}, lr {LR}.\n\n"
        "Treino feito no split `val` (70 clipes, 7 indivíduos); avaliação por "
        "clustering em todos os 100 clipes (10 indivíduos — 3 não vistos no treino, "
        "open-set re-id em espírito).\n\n" + header + "".join(rows) + "\n## Artefatos\n\n"
        "- Pesos, embeddings e métricas em JSON por run: `reports/runs/m3_grid/<exp_id>/`.\n"
        "- Figura comparativa: `reports/figures/m3_grid.png`.\n\n"
        "## Limitações\n\n"
        "- Sample dataset (100 clipes) é insuficiente pra distinguir variâncias "
        "estatísticas entre configurações. Use estes números como sanity check do "
        "pipeline; conclusões científicas pedem o dataset PAM completo.\n"
    )
    TABLE_PATH.write_text(body)
    print(f"table written to {TABLE_PATH.relative_to(ROOT)}")


def _write_figure(grid: list[GridEntry], results: dict[str, ExperimentResult]) -> None:
    valid = [(entry, results[entry.exp_id]) for entry in grid if entry.exp_id in results]
    if not valid:
        return
    try:
        import matplotlib

        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
    except ImportError:
        print("matplotlib unavailable; skipping figure")
        return

    def _label(r: ExperimentResult) -> str:
        suffix = " (specaug)" if r.config.spec_augment else ""
        return f"{r.config.backbone}\n{r.config.loss}{suffix}"

    labels = [_label(r) for _, r in valid]
    hungarian = [r.metrics.hungarian_accuracy for _, r in valid]
    ari = [r.metrics.ari for _, r in valid]
    nmi = [r.metrics.nmi for _, r in valid]

    x = range(len(valid))
    fig, ax = plt.subplots(figsize=(max(8, len(valid) * 1.5), 4.5))
    width = 0.25
    ax.bar([i - width for i in x], hungarian, width, label="Hungarian acc")
    ax.bar(list(x), ari, width, label="ARI")
    ax.bar([i + width for i in x], nmi, width, label="NMI")
    ax.set_xticks(list(x))
    ax.set_xticklabels(labels, rotation=20, ha="right", fontsize=8)
    ax.set_ylim(0, 1)
    ax.set_ylabel("score")
    ax.set_title("M3 grid — clustering metrics by configuration")
    ax.legend(loc="upper right")
    ax.grid(axis="y", alpha=0.3)
    fig.tight_layout()
    FIGURE_PATH.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(FIGURE_PATH, dpi=120)
    plt.close(fig)
    print(f"figure written to {FIGURE_PATH.relative_to(ROOT)}")


if __name__ == "__main__":
    raise SystemExit(main())
