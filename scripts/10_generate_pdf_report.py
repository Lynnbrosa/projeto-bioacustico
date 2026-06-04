"""Generate a single PDF report summarising M0-M4 + HDBSCAN tuning.

Pulls already-computed artifacts from ``data/``, ``reports/`` and assembles
a multi-page PDF at ``reports/bioacid_report.pdf`` using matplotlib's PDF
backend (no external dependency beyond matplotlib).

Usage:
    uv run --extra ml --extra dev scripts/10_generate_pdf_report.py
"""

from __future__ import annotations

import json
import sys
from pathlib import Path
from textwrap import dedent

ROOT = Path(__file__).resolve().parents[1]
EMBEDDINGS_PATH = ROOT / "data" / "processed" / "sample_embeddings.parquet"
SWEEP_PATH = ROOT / "reports" / "runs" / "m2_cluster_sweep.json"
M2_CLUSTER_PATH = ROOT / "reports" / "runs" / "m2_cluster_metrics.json"
GRID_DIR = ROOT / "reports" / "runs" / "m3_grid"
FIG_DIR = ROOT / "reports" / "figures"
OUTPUT_PDF = ROOT / "reports" / "bioacid_report.pdf"


def main() -> int:
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from matplotlib.backends.backend_pdf import PdfPages

    if not EMBEDDINGS_PATH.exists():
        print(f"missing {EMBEDDINGS_PATH}; run scripts/01_reproduce_demo.py first", file=sys.stderr)
        return 1

    m2_metrics = _load_json(M2_CLUSTER_PATH)
    sweep = _load_json(SWEEP_PATH)
    grid_results = _load_grid_results(GRID_DIR)

    OUTPUT_PDF.parent.mkdir(parents=True, exist_ok=True)
    with PdfPages(OUTPUT_PDF) as pdf:
        _cover_page(pdf, plt)
        _non_technical_page(pdf, plt, m2_metrics, sweep, grid_results)
        _summary_page(pdf, plt, m2_metrics, sweep, grid_results)
        _figure_page(pdf, plt, FIG_DIR / "m3_grid.png", "M3 grid — métricas por configuração")
        _figure_page(
            pdf,
            plt,
            FIG_DIR / "embedding_umap_2d.png",
            "UMAP 2d dos embeddings (checkpoint upstream)",
        )
        _figure_page(
            pdf,
            plt,
            FIG_DIR / "embedding_tsne_2d.png",
            "t-SNE 2d dos embeddings — o espaço onde HDBSCAN bate 0.81",
        )
        _figure_page(
            pdf,
            plt,
            FIG_DIR / "distance_histogram.png",
            "Distância cosseno: pares same vs different individual",
        )
        _figure_page(
            pdf,
            plt,
            FIG_DIR / "cluster_confusion.png",
            "HDBSCAN confusion (t-SNE 2d, min_cluster_size=4)",
        )
        _conclusion_page(pdf, plt, m2_metrics, sweep)

    print(f"PDF written to {OUTPUT_PDF.relative_to(ROOT)}")
    return 0


def _load_json(path: Path) -> dict[str, object]:
    if not path.exists():
        return {}
    return json.loads(path.read_text())


def _load_grid_results(grid_dir: Path) -> list[tuple[str, dict[str, object]]]:
    runs: list[tuple[str, dict[str, object]]] = []
    if not grid_dir.exists():
        return runs
    for exp_dir in sorted(grid_dir.iterdir()):
        result_path = exp_dir / "result.json"
        if result_path.exists():
            runs.append((exp_dir.name, json.loads(result_path.read_text())))
    return runs


def _text_page(pdf, plt, lines: list[str], title: str | None = None) -> None:  # type: ignore[no-untyped-def]
    fig, ax = plt.subplots(figsize=(8.5, 11))
    ax.axis("off")
    y = 0.97
    if title:
        ax.text(
            0.5,
            y,
            title,
            ha="center",
            va="top",
            fontsize=18,
            fontweight="bold",
            transform=ax.transAxes,
        )
        y -= 0.06
    for line in lines:
        size = 9
        weight = "normal"
        prefix = ""
        if line.startswith("# "):
            size, weight, line = 14, "bold", line[2:]
        elif line.startswith("## "):
            size, weight, line = 12, "bold", line[3:]
        elif line.startswith("- "):
            prefix = "  •  "
            line = line[2:]
        elif line.startswith("```"):
            continue  # skip fence markers
        font = "monospace" if line.startswith(("    ", "\t")) else "sans-serif"
        ax.text(
            0.05,
            y,
            prefix + line.rstrip(),
            ha="left",
            va="top",
            fontsize=size,
            fontweight=weight,
            family=font,
            transform=ax.transAxes,
            wrap=True,
        )
        y -= 0.018 * (1 + size / 10)
        if y < 0.05:
            break
    pdf.savefig(fig)
    plt.close(fig)


def _figure_page(pdf, plt, image_path: Path, caption: str) -> None:  # type: ignore[no-untyped-def]
    if not image_path.exists():
        return
    from matplotlib.image import imread

    fig, ax = plt.subplots(figsize=(8.5, 11))
    ax.axis("off")
    img = imread(image_path)
    ax.imshow(img)
    ax.set_title(caption, fontsize=12, pad=14)
    pdf.savefig(fig, bbox_inches="tight")
    plt.close(fig)


def _cover_page(pdf, plt) -> None:  # type: ignore[no-untyped-def]
    lines = [
        "",
        "",
        "Identificação individual de aves por som",
        "",
        "Pipeline open-source replicando Lapp et al. (bioRxiv 2025)",
        "estendido com tuning de HDBSCAN, matriz de experimentos",
        "(3 backbones x 2 losses + 3 ablações) e protocolo de",
        "extensão neotropical via Xeno-canto.",
        "",
        "",
        "─" * 60,
        "",
        "Repositório:  github.com/Lynnbrosa/projeto-bioacustico",
        "Stack:        Python 3.11 · PyTorch · OpenSoundscape · HDBSCAN",
        "Sample:       100 clipes Ovenbird (10 indivíduos)",
        "",
    ]

    fig, ax = plt.subplots(figsize=(8.5, 11))
    ax.axis("off")

    ax.text(
        0.5,
        0.85,
        "bioacid",
        ha="center",
        va="center",
        fontsize=42,
        fontweight="bold",
        transform=ax.transAxes,
    )
    ax.text(
        0.5,
        0.79,
        "Relatório técnico — milestones M0 a M4",
        ha="center",
        fontsize=14,
        transform=ax.transAxes,
        style="italic",
    )

    y = 0.70
    for line in lines:
        ax.text(0.1, y, line, ha="left", fontsize=11, transform=ax.transAxes, family="serif")
        y -= 0.028

    pdf.savefig(fig)
    plt.close(fig)


def _non_technical_page(  # type: ignore[no-untyped-def]
    pdf,
    plt,
    m2_metrics: dict,
    sweep: dict,
    grid_results: list,
) -> None:
    """Plain-language summary for someone without ML / bioacoustics background."""
    baseline_h = m2_metrics.get("hungarian_accuracy", 0.0)
    best_h = sweep.get("best", {}).get("hungarian_accuracy", 0.0)

    lines = [
        "# Em poucas palavras (versao nao-tecnica)",
        "",
        "## O que esse projeto faz",
        "",
        "Reconhece individuos diferentes de uma mesma especie de passaro pela gravacao",
        "do canto, como se fosse reconhecimento de voz. A gente roda um trecho de 2",
        "segundos por um modelo de IA, ele transforma o som numa 'impressao digital'",
        "numerica, e dois cantos do mesmo passaro tem impressoes parecidas — dois",
        "passaros diferentes, impressoes distantes.",
        "",
        "## Por que isso importa",
        "",
        "Pesquisadores plantam gravadores na floresta por meses. Hoje conseguem dizer",
        "'tem Ovenbird cantando aqui' mas nao 'eh o mesmo Ovenbird de ontem'. Saber",
        "quem eh quem permite contar individuos, estimar sobrevivencia, ver se o",
        "passaro voltou no ano seguinte — tudo sem precisar pegar e anilhar.",
        "",
        "## O que a gente fez nesse projeto",
        "",
        "- Replicou um pipeline de 2025 (Lapp et al.) que faz isso em Ovenbird.",
        "- Trocou todos os componentes pra codigo proprio, em Python, com testes.",
        "- Gerou um conjunto de 'impressoes digitais' do dataset publico de exemplo.",
        "- Testou variacoes (arquiteturas e formas de treinar) e fez um relatorio honesto",
        "  sobre o que funcionou e o que nao funcionou com poucos dados.",
        "- Deixou pronto pra rodar com som de aves brasileiras (sabia-laranjeira)",
        "  baixadas do Xeno-canto.",
        "",
        "## Numeros principais (sem jargao)",
        "",
        "- Acerto ao perguntar 'qual o canto mais parecido?' (entre 100): 98%",
        f"- Acerto ao agrupar os cantos automaticamente:                    {int(baseline_h * 100)}%",
        f"  ... e depois de afinar o algoritmo de agrupamento:              {int(best_h * 100)}%",
        "",
        "Em outras palavras: o modelo 'sabe' quem eh quem (98%), mas precisa de um",
        "passo extra de organizacao pra explicar isso pra gente em forma de grupos.",
        "Esse passo extra foi onde a maior melhoria apareceu (+15 pontos).",
        "",
        "## O que ficou pendente",
        "",
        "- Rodar tudo com o passaro brasileiro: o codigo esta pronto, mas baixar",
        "  audio do Xeno-canto exige internet aberta, que esse ambiente nao tem.",
        "- O dataset publico tem so 100 cantos de 10 individuos. Com mais dados, as",
        "  comparacoes entre arquiteturas viram conclusoes estatisticas, nao chutes.",
    ]
    _text_page(pdf, plt, lines)


def _summary_page(  # type: ignore[no-untyped-def]
    pdf,
    plt,
    m2_metrics: dict,
    sweep: dict,
    grid_results: list,
) -> None:
    baseline_h = m2_metrics.get("hungarian_accuracy", 0.0)
    best_h = sweep.get("best", {}).get("hungarian_accuracy", 0.0)
    best_cfg = sweep.get("best", {})

    lines: list[str] = [
        "# Resumo tecnico",
        "",
        "## M1 — Reprodução do demo upstream",
        "",
        "Carregamento do checkpoint pré-treinado (ResNet18, 234 classes) e geração",
        "de embeddings 512-d para 100 clipes da sample pública. Tempo total: ~2 s em CPU.",
        "",
        "- 1-NN leave-one-out accuracy (cosseno): **0.980**",
        "- Confirma que o extrator codifica identidade individual com alta separabilidade.",
        "",
        "## M2 — Cluster metrics no checkpoint upstream",
        "",
        "Defaults (UMAP→5d, min_cluster_size=5):",
        f"  ARI {m2_metrics.get('ari', 0):.3f}  ·  NMI {m2_metrics.get('nmi', 0):.3f}  "
        f"·  FMI {m2_metrics.get('fmi', 0):.3f}  ·  Hungarian {baseline_h:.3f}",
        "",
        "## HDBSCAN sweep (180 configs)",
        "",
        f"Melhor: reduction={best_cfg.get('reduction', 'n/a')}, "
        f"dims={best_cfg.get('dims', '?')}, "
        f"min_cluster_size={best_cfg.get('min_cluster_size', '?')}, "
        f"min_samples={best_cfg.get('min_samples', '?')}",
        "",
        f"  Hungarian {best_h:.3f}  (+{(best_h - baseline_h):.3f} vs. baseline)",
        f"  ARI       {best_cfg.get('ari', 0):.3f}",
        f"  NMI       {best_cfg.get('nmi', 0):.3f}",
        "",
        "## M3 — Grid de experimentos",
        "",
        f"Total de runs: {len(grid_results)}. Treino no split val da sample",
        "(70 clipes, 7 indivíduos), avaliação clustering nos 100 clipes (10 indiv).",
        "",
    ]

    if grid_results:
        lines.append("## Top 3 runs do grid (por Hungarian)")
        lines.append("")
        sorted_runs = sorted(
            grid_results,
            key=lambda r: r[1].get("metrics", {}).get("hungarian_accuracy", 0),
            reverse=True,
        )[:3]
        for exp_id, payload in sorted_runs:
            m = payload.get("metrics", {})
            lines.append(
                f"  {exp_id:<40} Hungarian {m.get('hungarian_accuracy', 0):.3f}  "
                f"ARI {m.get('ari', 0):.3f}"
            )

    lines.append("")
    lines.append("## M4 — Extensão neotropical (sabiá-laranjeira)")
    lines.append("")
    lines.append("Cliente Xeno-canto + heurística de pseudo-label (recordist+date+locality)")
    lines.append("implementados em ``bioacid/xeno_canto.py``. 13 testes offline. Execução")
    lines.append("end-to-end requer máquina com acesso a xeno-canto.org.")

    _text_page(pdf, plt, lines)


def _conclusion_page(pdf, plt, m2_metrics: dict, sweep: dict) -> None:  # type: ignore[no-untyped-def]
    baseline = m2_metrics.get("hungarian_accuracy", 0.0)
    best = sweep.get("best", {}).get("hungarian_accuracy", 0.0)
    delta = best - baseline

    lines = (
        dedent(f"""
        # Conclusões

        ## Reprodução

        - Pipeline upstream reproduzido sem regressão: 1-NN LOO 0.980 sobre embeddings
          do checkpoint pré-treinado.

        ## Onde estava o gargalo

        - Métrica Hungarian no M2 baseline: {baseline:.3f}.
        - 1-NN LOO no mesmo embedding: 0.980.
        - Gap explicado integralmente pelo clustering, não pelo extrator.

        ## Tuning do HDBSCAN

        - Sweep de 180 configurações (reduction x dims x min_cluster x min_samples).
        - Best: t-SNE 2d + min_cluster_size=4 + min_samples=1.
        - Hungarian sobe pra {best:.3f} (+{delta:.3f}).

        ## Matriz de experimentos M3

        - ResNet18 lidera (com CE ou ArcFace) em sample pequena.
        - ResNet50, EfficientNet-B0 e ConvNeXt-Tiny colapsam por overfitting.
        - SpecAugment não ajuda com 70 clipes de treino.
        - SupCon underperformed (esperado: precisa de batch maior).

        ## Limitações

        - 100 clipes da sample são insuficientes pra distinguir variâncias entre
          configurações com significância estatística.
        - Conclusões científicas finais pedem o dataset PAM completo (não público).

        ## Próximos passos sugeridos

        - Rodar M4 em ambiente com xeno-canto.org acessível.
        - Estender o grid HDBSCAN pra outros backbones treinados.
        - Curar subset manualmente rotulado pra calibrar a heurística de pseudo-label.
    """)
        .strip()
        .splitlines()
    )

    _text_page(pdf, plt, lines)


if __name__ == "__main__":
    raise SystemExit(main())
