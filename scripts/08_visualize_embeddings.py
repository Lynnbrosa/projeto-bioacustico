"""M2/M3 visualization: embedding plots and cluster diagnostics.

Reads ``data/processed/sample_embeddings.parquet`` (produced by
``scripts/01_reproduce_demo.py``) and writes three figures:

- ``reports/figures/embedding_umap_2d.png`` — UMAP scatter, coloured by
  true individual.
- ``reports/figures/embedding_tsne_2d.png`` — t-SNE scatter; this is the
  space where HDBSCAN scored Hungarian 0.81 (see m2_hdbscan_sweep.md).
- ``reports/figures/distance_histogram.png`` — within- vs between-individual
  cosine distance distributions, showing the discriminability gap.
- ``reports/figures/cluster_confusion.png`` — confusion matrix after
  Hungarian-aligning HDBSCAN clusters to true individuals.

Usage:
    uv run --extra ml --extra dev scripts/08_visualize_embeddings.py
"""

from __future__ import annotations

import sys
import warnings
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
EMBEDDINGS_PATH = ROOT / "data" / "processed" / "sample_embeddings.parquet"
FIG_DIR = ROOT / "reports" / "figures"


def main() -> int:
    if not EMBEDDINGS_PATH.exists():
        print(f"missing {EMBEDDINGS_PATH}; run scripts/01_reproduce_demo.py first", file=sys.stderr)
        return 1

    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    df = pd.read_parquet(EMBEDDINGS_PATH)
    feature_cols = [c for c in df.columns if c.startswith("f") and c[1:].isdigit()]
    features = df[feature_cols].to_numpy(dtype=np.float32)
    labels = df["aiid_label"].to_numpy()
    print(
        f"loaded {features.shape[0]} embeddings (dim {features.shape[1]}), {len(np.unique(labels))} individuals"
    )

    FIG_DIR.mkdir(parents=True, exist_ok=True)

    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        _scatter_umap(features, labels, FIG_DIR / "embedding_umap_2d.png", plt)
        _scatter_tsne(features, labels, FIG_DIR / "embedding_tsne_2d.png", plt)
        _distance_histogram(features, labels, FIG_DIR / "distance_histogram.png", plt)
        _confusion_matrix(features, labels, FIG_DIR / "cluster_confusion.png", plt)

    print(f"figures written to {FIG_DIR.relative_to(ROOT)}/")
    return 0


def _scatter_umap(features: np.ndarray, labels: np.ndarray, path: Path, plt) -> None:  # type: ignore[no-untyped-def]
    import umap

    reduced = umap.UMAP(n_components=2, random_state=42).fit_transform(features)
    _scatter(reduced, labels, "UMAP 2d on Ovenbird sample embeddings", path, plt)


def _scatter_tsne(features: np.ndarray, labels: np.ndarray, path: Path, plt) -> None:  # type: ignore[no-untyped-def]
    from sklearn.manifold import TSNE

    reduced = TSNE(n_components=2, random_state=42, init="pca").fit_transform(features)
    _scatter(reduced, labels, "t-SNE 2d on Ovenbird sample embeddings", path, plt)


def _scatter(reduced: np.ndarray, labels: np.ndarray, title: str, path: Path, plt) -> None:  # type: ignore[no-untyped-def]
    fig, ax = plt.subplots(figsize=(7, 6))
    cmap = plt.get_cmap("tab10")
    unique_labels = np.unique(labels)
    for i, label in enumerate(unique_labels):
        mask = labels == label
        ax.scatter(
            reduced[mask, 0],
            reduced[mask, 1],
            s=40,
            alpha=0.85,
            color=cmap(i % 10),
            label=f"indiv {label}",
            edgecolor="white",
            linewidth=0.5,
        )
    ax.set_title(title)
    ax.set_xlabel("dim 1")
    ax.set_ylabel("dim 2")
    ax.legend(loc="best", fontsize=8, ncol=2)
    ax.set_aspect("equal", adjustable="datalim")
    ax.grid(alpha=0.2)
    fig.tight_layout()
    fig.savefig(path, dpi=120)
    plt.close(fig)
    print(f"  {path.relative_to(ROOT)}")


def _distance_histogram(features: np.ndarray, labels: np.ndarray, path: Path, plt) -> None:  # type: ignore[no-untyped-def]
    from sklearn.metrics.pairwise import cosine_distances

    distances = cosine_distances(features)
    n = features.shape[0]
    same_label = labels[:, None] == labels[None, :]
    diff_label = ~same_label
    upper = np.triu(np.ones((n, n), dtype=bool), k=1)
    within = distances[same_label & upper]
    between = distances[diff_label & upper]

    fig, ax = plt.subplots(figsize=(7, 4.5))
    ax.hist(within, bins=40, alpha=0.7, label=f"within ({len(within)} pairs)", color="#2ca02c")
    ax.hist(between, bins=40, alpha=0.7, label=f"between ({len(between)} pairs)", color="#d62728")
    ax.axvline(within.mean(), color="#2ca02c", linestyle="--", linewidth=1)
    ax.axvline(between.mean(), color="#d62728", linestyle="--", linewidth=1)
    ax.set_xlabel("cosine distance")
    ax.set_ylabel("pair count")
    ax.set_title("Pairwise cosine distance: same vs different individual")
    ax.legend(loc="best")
    ax.grid(alpha=0.2)
    fig.tight_layout()
    fig.savefig(path, dpi=120)
    plt.close(fig)
    print(f"  {path.relative_to(ROOT)}")


def _confusion_matrix(features: np.ndarray, labels: np.ndarray, path: Path, plt) -> None:  # type: ignore[no-untyped-def]
    from scipy.optimize import linear_sum_assignment
    from sklearn.cluster import HDBSCAN
    from sklearn.manifold import TSNE
    from sklearn.metrics import confusion_matrix as sk_confusion

    reduced = TSNE(n_components=2, random_state=42, init="pca").fit_transform(features)
    predicted = HDBSCAN(min_cluster_size=4, min_samples=1).fit_predict(reduced)

    if predicted.max() < 0:
        print(f"  skipping confusion ({path.name}): HDBSCAN found no clusters")
        return

    unique_truth = np.unique(labels)
    unique_pred = np.unique(predicted)
    cm = sk_confusion(labels, predicted, labels=unique_truth)
    # `labels=` makes rows match unique_truth order. Columns still cover
    # all predicted values automatically.
    _, col_ind = linear_sum_assignment(-cm)
    used = set(col_ind.tolist())
    extra_cols = [c for c in range(cm.shape[1]) if c not in used]
    column_order = col_ind.tolist() + extra_cols
    reordered = cm[:, column_order]
    col_labels = [str(unique_pred[c]) for c in column_order]
    row_labels = [str(t) for t in unique_truth]

    fig, ax = plt.subplots(figsize=(max(8, reordered.shape[1] * 0.4 + 2), 6))
    im = ax.imshow(reordered, cmap="Greens", aspect="auto")
    ax.set_xlabel("predicted cluster (Hungarian-aligned; -1 = noise)")
    ax.set_ylabel("true individual")
    ax.set_title("HDBSCAN confusion (t-SNE 2d, min_cluster_size=4)")
    ax.set_xticks(range(reordered.shape[1]))
    ax.set_xticklabels(col_labels, fontsize=8)
    ax.set_yticks(range(reordered.shape[0]))
    ax.set_yticklabels(row_labels, fontsize=8)
    for i in range(reordered.shape[0]):
        for j in range(reordered.shape[1]):
            value = int(reordered[i, j])
            if value > 0:
                ax.text(j, i, str(value), ha="center", va="center", fontsize=8)
    fig.colorbar(im, ax=ax, fraction=0.04, pad=0.04)
    fig.tight_layout()
    fig.savefig(path, dpi=120)
    plt.close(fig)
    print(f"  {path.relative_to(ROOT)}")


if __name__ == "__main__":
    raise SystemExit(main())
