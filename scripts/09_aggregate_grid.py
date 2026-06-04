"""Aggregate per-run ``result.json`` files into the M3 table and figure.

Lets us regenerate ``reports/experiment_table.md`` + ``reports/figures/m3_grid.png``
after fixing a single failing config, without re-running the whole grid.

Usage:
    uv run --extra ml --extra dev scripts/09_aggregate_grid.py
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
GRID_DIR = ROOT / "reports" / "runs" / "m3_grid"
TABLE_PATH = ROOT / "reports" / "experiment_table.md"
FIGURE_PATH = ROOT / "reports" / "figures" / "m3_grid.png"


# Order to display in the table; rest come at the end alphabetically.
PREFERRED_ORDER = [
    "resnet18__cross_entropy",
    "resnet18__arcface",
    "resnet50__cross_entropy",
    "resnet50__arcface",
    "efficientnet_b0__cross_entropy",
    "efficientnet_b0__arcface",
    "resnet18__cross_entropy__specaug",
    "resnet18__supcon",
    "convnext_tiny__cross_entropy",
]


def main() -> int:
    if not GRID_DIR.exists():
        print(f"missing {GRID_DIR}", file=sys.stderr)
        return 1

    runs: list[tuple[str, dict[str, object]]] = []
    for exp_dir in sorted(GRID_DIR.iterdir()):
        result_path = exp_dir / "result.json"
        if not result_path.exists():
            print(f"  skipping {exp_dir.name}: no result.json")
            continue
        runs.append((exp_dir.name, json.loads(result_path.read_text())))

    runs.sort(key=lambda r: (_order_index(r[0]), r[0]))

    _write_table(runs)
    _write_figure(runs)
    return 0


def _order_index(exp_id: str) -> int:
    if exp_id in PREFERRED_ORDER:
        return PREFERRED_ORDER.index(exp_id)
    return len(PREFERRED_ORDER)


def _write_table(runs: list[tuple[str, dict[str, object]]]) -> None:
    header = (
        "| Experiment | Backbone | Loss | SpecAug | ARI | NMI | FMI | Hungarian | "
        "Purity | Train top-1 | Train (s) |\n"
        "| --- | --- | --- | :-: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |\n"
    )
    rows = []
    for exp_id, payload in runs:
        cfg = payload["config"]
        m = payload["metrics"]
        rows.append(
            f"| {exp_id} | {cfg['backbone']} | {cfg['loss']} | "
            f"{'yes' if cfg['spec_augment'] else 'no'} | "
            f"{m['ari']:.3f} | {m['nmi']:.3f} | {m['fmi']:.3f} | "
            f"{m['hungarian_accuracy']:.3f} | {m['purity']:.3f} | "
            f"{payload['final_train_top1']:.3f} | "
            f"{payload['train_seconds']:.0f} |\n"
        )

    epochs = runs[0][1]["config"]["epochs"] if runs else "?"
    batch = runs[0][1]["config"]["batch_size"] if runs else "?"
    lr = runs[0][1]["config"]["lr"] if runs else "?"

    body = (
        "# M3 — Matriz de experimentos (baseline + ablações)\n\n"
        f"Grid: `{len(runs)}` runs, {epochs} épocas, batch {batch}, lr {lr}.\n\n"
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
    print(f"table: {TABLE_PATH.relative_to(ROOT)}")


def _write_figure(runs: list[tuple[str, dict[str, object]]]) -> None:
    if not runs:
        return
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    def label_of(exp_id: str, cfg: dict[str, object]) -> str:
        suffix = " (specaug)" if cfg["spec_augment"] else ""
        return f"{cfg['backbone']}\n{cfg['loss']}{suffix}"

    labels = [label_of(exp_id, payload["config"]) for exp_id, payload in runs]
    hungarian = [payload["metrics"]["hungarian_accuracy"] for _, payload in runs]
    ari = [payload["metrics"]["ari"] for _, payload in runs]
    nmi = [payload["metrics"]["nmi"] for _, payload in runs]

    x = range(len(runs))
    fig, ax = plt.subplots(figsize=(max(10, len(runs) * 1.4), 4.5))
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
    print(f"figure: {FIGURE_PATH.relative_to(ROOT)}")


if __name__ == "__main__":
    raise SystemExit(main())
