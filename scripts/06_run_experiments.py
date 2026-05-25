"""M3: Run the experiment grid and aggregate results.

Sweeps ``(backbone, loss)`` combinations defined below, runs each via
:func:`bioacid.experiment.run_experiment`, and writes:

- ``reports/runs/m3_grid/<exp_id>/`` — per-run artifacts (backbone.pth,
  embeddings.npy, result.json).
- ``reports/experiment_table.md`` — aggregated markdown table.
- ``reports/figures/m3_grid.png`` — bar chart of Hungarian accuracy per run.

Usage:
    uv run --extra ml --extra dev scripts/04_run_experiments.py
"""

from __future__ import annotations

import sys
import time
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

BACKBONES: list[BackboneName] = ["resnet18", "resnet50", "efficientnet_b0"]
LOSSES: list[LossName] = ["cross_entropy", "arcface"]
EPOCHS = 10
BATCH_SIZE = 32
LR = 1e-3


def main() -> int:
    if not UPSTREAM.exists():
        print(f"upstream not found at {UPSTREAM}; clone first", file=sys.stderr)
        return 1

    GRID_DIR.mkdir(parents=True, exist_ok=True)
    results: list[ExperimentResult] = []

    t_total = time.time()
    for backbone in BACKBONES:
        for loss in LOSSES:
            exp_id = f"{backbone}__{loss}"
            print(f"\n=== {exp_id} ===")
            config = TrainConfig(
                backbone=backbone,
                loss=loss,
                epochs=EPOCHS,
                batch_size=BATCH_SIZE,
                lr=LR,
            )
            try:
                result = run_experiment(
                    config,
                    train_csv=CSV_PATH,
                    audio_root=UPSTREAM,
                    output_dir=GRID_DIR / exp_id,
                    device="cpu",
                )
                results.append(result)
            except Exception as exc:
                print(f"FAILED {exp_id}: {exc}", file=sys.stderr)
                (GRID_DIR / exp_id).mkdir(parents=True, exist_ok=True)
                (GRID_DIR / exp_id / "error.txt").write_text(repr(exc))

    print(f"\n=== grid finished in {time.time() - t_total:.1f}s ===")
    _write_table(results)
    _write_figure(results)
    return 0


def _write_table(results: list[ExperimentResult]) -> None:
    header = (
        "| Backbone | Loss | ARI | NMI | FMI | Hungarian | Purity | Train top-1 | Train (s) |\n"
        "| --- | --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: |\n"
    )
    rows: list[str] = []
    for r in results:
        m = r.metrics
        rows.append(
            f"| {r.config.backbone} | {r.config.loss} | "
            f"{m.ari:.3f} | {m.nmi:.3f} | {m.fmi:.3f} | "
            f"{m.hungarian_accuracy:.3f} | {m.purity:.3f} | "
            f"{r.final_train_top1:.3f} | {r.train_seconds:.0f} |\n"
        )

    body = (
        "# M3 — Matriz de experimentos\n\n"
        f"Grid: `{len(results)}` runs, {EPOCHS} épocas, batch {BATCH_SIZE}, lr {LR}.\n\n"
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


def _write_figure(results: list[ExperimentResult]) -> None:
    if not results:
        return
    try:
        import matplotlib

        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
    except ImportError:
        print("matplotlib unavailable; skipping figure")
        return

    labels = [f"{r.config.backbone}\n{r.config.loss}" for r in results]
    hungarian = [r.metrics.hungarian_accuracy for r in results]
    ari = [r.metrics.ari for r in results]
    nmi = [r.metrics.nmi for r in results]

    x = range(len(results))
    fig, ax = plt.subplots(figsize=(max(8, len(results) * 1.5), 4.5))
    width = 0.25
    ax.bar([i - width for i in x], hungarian, width, label="Hungarian acc")
    ax.bar(list(x), ari, width, label="ARI")
    ax.bar([i + width for i in x], nmi, width, label="NMI")
    ax.set_xticks(list(x))
    ax.set_xticklabels(labels, rotation=20, ha="right")
    ax.set_ylim(0, 1)
    ax.set_ylabel("score")
    ax.set_title("M3 grid — clustering metrics by (backbone, loss)")
    ax.legend(loc="lower right")
    ax.grid(axis="y", alpha=0.3)
    fig.tight_layout()
    FIGURE_PATH.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(FIGURE_PATH, dpi=120)
    plt.close(fig)
    print(f"figure written to {FIGURE_PATH.relative_to(ROOT)}")


if __name__ == "__main__":
    raise SystemExit(main())
