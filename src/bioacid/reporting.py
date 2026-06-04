"""Shared markdown table + bar-chart renderers for the M3 experiment grid.

The grid runner (``scripts/06_run_experiments.py``) and the aggregator
(``scripts/09_aggregate_grid.py``) both materialise per-config results
into ``reports/experiment_table.md`` + ``reports/figures/m3_grid.png``.

Keeping a single source of truth here avoids drift between the two
entry points. Heavy imports (matplotlib) are kept lazy.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

PREFERRED_ORDER: tuple[str, ...] = (
    "resnet18__cross_entropy",
    "resnet18__arcface",
    "resnet50__cross_entropy",
    "resnet50__arcface",
    "efficientnet_b0__cross_entropy",
    "efficientnet_b0__arcface",
    "resnet18__cross_entropy__specaug",
    "resnet18__supcon",
    "convnext_tiny__cross_entropy",
)


def load_grid_results(grid_dir: Path) -> list[tuple[str, dict[str, Any]]]:
    """Read every ``<exp_id>/result.json`` under ``grid_dir`` and sort it."""
    runs: list[tuple[str, dict[str, Any]]] = []
    for exp_dir in sorted(grid_dir.iterdir()):
        result_path = exp_dir / "result.json"
        if not result_path.exists():
            continue
        runs.append((exp_dir.name, json.loads(result_path.read_text())))
    return sort_runs(runs)


def sort_runs(runs: list[tuple[str, dict[str, Any]]]) -> list[tuple[str, dict[str, Any]]]:
    """Sort runs by the M3 preferred order, then alphabetically."""
    order = {name: i for i, name in enumerate(PREFERRED_ORDER)}
    return sorted(runs, key=lambda r: (order.get(r[0], len(order)), r[0]))


def write_experiment_table(runs: list[tuple[str, dict[str, Any]]], path: Path) -> None:
    """Write the M3 markdown table to ``path``.

    ``runs`` is a list of ``(exp_id, result_payload)`` where ``result_payload``
    matches ``ExperimentResult.as_dict()``.
    """
    if not runs:
        path.write_text("# M3 — Matriz de experimentos\n\n_no runs_\n")
        return

    epochs = runs[0][1]["config"]["epochs"]
    batch = runs[0][1]["config"]["batch_size"]
    lr = runs[0][1]["config"]["lr"]

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
    path.write_text(body)


def write_experiment_figure(runs: list[tuple[str, dict[str, Any]]], path: Path) -> None:
    """Write the M3 bar-chart figure to ``path``."""
    if not runs:
        return
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    def label_of(cfg: dict[str, Any]) -> str:
        suffix = " (specaug)" if cfg["spec_augment"] else ""
        return f"{cfg['backbone']}\n{cfg['loss']}{suffix}"

    labels = [label_of(payload["config"]) for _, payload in runs]
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
    path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(path, dpi=120)
    plt.close(fig)


__all__ = [
    "PREFERRED_ORDER",
    "load_grid_results",
    "sort_runs",
    "write_experiment_figure",
    "write_experiment_table",
]
