"""M2/M3: Sweep HDBSCAN + UMAP hyperparameters on the saved embeddings.

The M2 baseline used min_cluster_size=5, UMAP→5d and found 7 clusters from
10 individuals — homogeneity 0.80 but Hungarian only 0.70. The 1-NN LOO
accuracy on the same embeddings is 0.98, suggesting the bottleneck is the
clustering step, not the extractor.

This script sweeps:
- `reduction_algorithm` ∈ {none, umap, tsne}
- `reduced_n_dimensions` ∈ {2, 5, 10}
- `min_cluster_size` ∈ {2, 3, 4, 5, 7}
- `min_samples` ∈ {None, 1, 2, 3}

Picks the config with the best Hungarian accuracy and writes
``reports/runs/m2_cluster_sweep.json`` + an updated table.

Usage:
    uv run --extra ml --extra dev scripts/07_hdbscan_sweep.py
"""

from __future__ import annotations

import json
import sys
import time
import warnings
from itertools import product
from pathlib import Path

import numpy as np
import pandas as pd

from bioacid.cluster import hdbscan_cluster, reduce_dims
from bioacid.evaluate import clustering_metrics

ROOT = Path(__file__).resolve().parents[1]
EMBEDDINGS_PATH = ROOT / "data" / "processed" / "sample_embeddings.parquet"
OUTPUT_JSON = ROOT / "reports" / "runs" / "m2_cluster_sweep.json"
OUTPUT_MD = ROOT / "reports" / "m2_hdbscan_sweep.md"

REDUCTIONS = ["none", "umap", "tsne"]
DIMS = [2, 5, 10]
MIN_CLUSTER_SIZES = [2, 3, 4, 5, 7]
MIN_SAMPLES = [None, 1, 2, 3]


def main() -> int:
    if not EMBEDDINGS_PATH.exists():
        print(f"missing {EMBEDDINGS_PATH}; run scripts/01_reproduce_demo.py first", file=sys.stderr)
        return 1

    df = pd.read_parquet(EMBEDDINGS_PATH)
    feature_cols = [c for c in df.columns if c.startswith("f") and c[1:].isdigit()]
    features = df[feature_cols].to_numpy(dtype=np.float32)
    truth = df["aiid_label"].to_numpy()
    n_true = len(np.unique(truth))
    print(f"loaded {features.shape[0]} embeddings of dim {features.shape[1]}")
    print(f"truth: {n_true} individuals")

    runs: list[dict[str, object]] = []
    t0 = time.time()
    total = len(REDUCTIONS) * len(DIMS) * len(MIN_CLUSTER_SIZES) * len(MIN_SAMPLES)
    print(f"running {total} configurations...")

    cluster_param_pairs = list(product(MIN_CLUSTER_SIZES, MIN_SAMPLES))
    i = 0

    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        # Reduction is expensive (UMAP/t-SNE on 100 x 512 features takes
        # ~1 s per call); HDBSCAN on the reduced array is cheap. Compute
        # the reduction once per (reduction, dim) and sweep the HDBSCAN
        # params on the cached array.
        for reduction in REDUCTIONS:
            for dim in DIMS:
                if reduction == "none" and dim != DIMS[0]:
                    i += len(cluster_param_pairs)
                    continue
                try:
                    reduced = reduce_dims(
                        features,
                        algorithm=reduction,  # type: ignore[arg-type]
                        n_components=dim,
                        random_state=42,
                    )
                except Exception as exc:
                    for mcs, ms in cluster_param_pairs:
                        i += 1
                        runs.append(
                            {
                                "reduction": reduction,
                                "dims": dim,
                                "min_cluster_size": mcs,
                                "min_samples": ms,
                                "error": repr(exc),
                            }
                        )
                    continue

                for mcs, ms in cluster_param_pairs:
                    i += 1
                    try:
                        predicted = hdbscan_cluster(reduced, min_cluster_size=mcs, min_samples=ms)
                    except Exception as exc:
                        runs.append(
                            {
                                "reduction": reduction,
                                "dims": dim,
                                "min_cluster_size": mcs,
                                "min_samples": ms,
                                "error": repr(exc),
                            }
                        )
                        continue
                    metrics = clustering_metrics(truth, predicted)
                    runs.append(
                        {
                            "reduction": reduction,
                            "dims": dim,
                            "min_cluster_size": mcs,
                            "min_samples": ms,
                            "n_clusters": len(np.unique(predicted[predicted >= 0])),
                            "n_noise": int((predicted < 0).sum()),
                            **metrics.as_dict(),
                        }
                    )
                    if i % 20 == 0:
                        print(f"  {i}/{total}")

    elapsed = time.time() - t0
    print(f"done in {elapsed:.1f}s")

    valid = [r for r in runs if "error" not in r]
    valid.sort(key=lambda r: r["hungarian_accuracy"], reverse=True)  # type: ignore[arg-type, return-value]
    best = valid[0]
    print("\n=== BEST ===")
    print(json.dumps(best, indent=2, default=float))

    OUTPUT_JSON.parent.mkdir(parents=True, exist_ok=True)
    OUTPUT_JSON.write_text(
        json.dumps(
            {
                "n_embeddings": int(features.shape[0]),
                "n_true_individuals": int(n_true),
                "elapsed_s": elapsed,
                "runs": runs,
                "best": best,
            },
            indent=2,
            default=float,
        )
    )

    _write_table(valid[:15], best)
    return 0


def _write_table(top: list[dict[str, object]], best: dict[str, object]) -> None:
    lines = [
        "# M2 — HDBSCAN hyperparameter sweep",
        "",
        f"Best Hungarian accuracy: **{best['hungarian_accuracy']:.3f}** "
        f"(reduction={best['reduction']}, dims={best['dims']}, "
        f"min_cluster_size={best['min_cluster_size']}, min_samples={best['min_samples']}).",
        "",
        "Comparison to baseline (umap, dims=5, min_cluster_size=5, min_samples=None):",
        "",
        "| Setup | Hungarian | ARI | NMI | Purity | n_clusters |",
        "| --- | ---: | ---: | ---: | ---: | ---: |",
        "| Baseline (M2 default) | 0.700 | 0.651 | 0.887 | 0.700 | 7 |",
        f"| **Best from sweep** | **{best['hungarian_accuracy']:.3f}** | "
        f"{best['ari']:.3f} | {best['nmi']:.3f} | {best['purity']:.3f} | "
        f"{best['n_clusters']} |",
        "",
        "## Top 15 configurations",
        "",
        "| Reduction | Dims | min_clust | min_samp | Hungarian | ARI | NMI | Purity | Clusters | Noise |",
        "| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |",
    ]
    for r in top:
        lines.append(
            f"| {r['reduction']} | {r['dims']} | {r['min_cluster_size']} | "
            f"{r['min_samples']} | {r['hungarian_accuracy']:.3f} | "
            f"{r['ari']:.3f} | {r['nmi']:.3f} | {r['purity']:.3f} | "
            f"{r['n_clusters']} | {r['n_noise']} |"
        )
    lines.append("")
    lines.append(
        "Note: the upstream-checkpoint embeddings have a 1-NN LOO accuracy of 0.980 — "
        "they encode individual identity well. The clustering step's parameter choice "
        "translates that into discoverable structure."
    )
    OUTPUT_MD.write_text("\n".join(lines) + "\n")
    print(f"\ntable: {OUTPUT_MD.relative_to(ROOT)}")
    print(f"json:  {OUTPUT_JSON.relative_to(ROOT)}")


if __name__ == "__main__":
    raise SystemExit(main())
