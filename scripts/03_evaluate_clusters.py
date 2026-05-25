"""M2: Cluster the saved embeddings and report the official metrics.

Reads ``data/processed/sample_embeddings.parquet`` (produced by
``scripts/01_reproduce_demo.py``), runs UMAP → HDBSCAN, and prints the four
metrics the paper reports (ARI, NMI, FMI, Hungarian matching accuracy) plus
homogeneity/completeness/V-measure/purity.

This is the M2 deliverable that validates the ported pipeline against the
upstream checkpoint: identical embeddings → identical metrics.

Usage:
    uv run --extra ml --extra dev scripts/03_evaluate_clusters.py
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd

from bioacid.cluster import cluster_embeddings
from bioacid.evaluate import clustering_metrics, format_metrics

ROOT = Path(__file__).resolve().parents[1]
EMBEDDINGS_PATH = ROOT / "data" / "processed" / "sample_embeddings.parquet"
OUTPUT_JSON = ROOT / "reports" / "runs" / "m2_cluster_metrics.json"


def main() -> int:
    if not EMBEDDINGS_PATH.exists():
        print(
            f"Embeddings not found at {EMBEDDINGS_PATH}.\nRun scripts/01_reproduce_demo.py first.",
            file=sys.stderr,
        )
        return 1

    df = pd.read_parquet(EMBEDDINGS_PATH)
    feature_cols = [c for c in df.columns if c.startswith("f") and c[1:].isdigit()]
    features = df[feature_cols].to_numpy(dtype=np.float32)
    truth = df["aiid_label"].to_numpy()
    print(f"loaded {features.shape[0]} embeddings of dim {features.shape[1]}")
    print(f"truth: {len(np.unique(truth))} individuals")

    predicted, _ = cluster_embeddings(
        features,
        reduction_algorithm="umap",
        reduced_n_dimensions=5,
        min_cluster_size=5,
        random_state=42,
    )
    n_clusters = len(np.unique(predicted[predicted >= 0]))
    n_noise = int((predicted < 0).sum())
    print(f"HDBSCAN: {n_clusters} clusters, {n_noise} noise points")

    metrics = clustering_metrics(truth, predicted)
    print()
    print(format_metrics(metrics))

    OUTPUT_JSON.parent.mkdir(parents=True, exist_ok=True)
    OUTPUT_JSON.write_text(json.dumps(metrics.as_dict(), indent=2))
    print(f"\nmetrics saved to {OUTPUT_JSON.relative_to(ROOT)}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
