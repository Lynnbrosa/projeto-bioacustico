"""Clustering metrics for individual identification.

Ports `bipartite_hungarian_matching_accuracy`, `cluster_purity` and the
aggregate `evaluate` helper from ``upstream/src/evaluation.py`` and re-exposes
them with full type hints.

Provides the four metrics required by M2 (ARI, NMI, FMI, Hungarian matching
accuracy) plus homogeneity, completeness, V-measure and purity for free since
sklearn computes them on the same call.
"""

from __future__ import annotations

from collections.abc import Sequence
from dataclasses import asdict, dataclass

import numpy as np
from numpy.typing import ArrayLike
from scipy.optimize import linear_sum_assignment
from sklearn.metrics import (
    accuracy_score,
    adjusted_rand_score,
    confusion_matrix,
    fowlkes_mallows_score,
    homogeneity_completeness_v_measure,
    normalized_mutual_info_score,
)


@dataclass(frozen=True, slots=True)
class ClusteringMetrics:
    """Standard cluster-vs-truth metrics. All values in [0, 1]."""

    ari: float
    nmi: float
    fmi: float
    homogeneity: float
    completeness: float
    v_measure: float
    purity: float
    hungarian_accuracy: float

    def as_dict(self) -> dict[str, float]:
        return asdict(self)


def bipartite_hungarian_matching_accuracy(
    true_labels: ArrayLike, predicted_labels: ArrayLike
) -> float:
    """Top-1 accuracy under the optimal one-to-one cluster→class assignment.

    Builds a confusion matrix and runs the Hungarian algorithm
    (`scipy.optimize.linear_sum_assignment`) to find the assignment of
    predicted clusters to true classes that maximises the number of correctly
    matched samples. Returns the fraction of samples covered by that
    assignment.

    This is the metric reported by Lapp et al. as "accuracy" in their tables.
    """
    true_array = np.asarray(true_labels)
    pred_array = np.asarray(predicted_labels)
    if true_array.shape != pred_array.shape:
        raise ValueError(f"shape mismatch: true={true_array.shape}, predicted={pred_array.shape}")

    cm = confusion_matrix(true_array, pred_array)
    row_ind, col_ind = linear_sum_assignment(-cm)
    total_correct = int(cm[row_ind, col_ind].sum())
    return total_correct / len(true_array)


def cluster_purity(true_labels: ArrayLike, predicted_labels: ArrayLike) -> float:
    """Fraction of samples that share the majority true label of their cluster."""
    true_array = np.asarray(true_labels)
    pred_array = np.asarray(predicted_labels)
    if true_array.shape != pred_array.shape:
        raise ValueError(f"shape mismatch: true={true_array.shape}, predicted={pred_array.shape}")

    majority: list[object] = []
    reordered_truth: list[object] = []
    for cluster_id in np.unique(pred_array):
        mask = pred_array == cluster_id
        members = true_array[mask]
        values, counts = np.unique(members, return_counts=True)
        majority_label = values[counts.argmax()]
        majority.extend([majority_label] * len(members))
        reordered_truth.extend(members.tolist())
    return float(accuracy_score(reordered_truth, majority))


def clustering_metrics(
    true_labels: ArrayLike,
    predicted_labels: ArrayLike,
) -> ClusteringMetrics:
    """Compute the full suite of clustering metrics in one shot."""
    true_array = np.asarray(true_labels)
    pred_array = np.asarray(predicted_labels)

    homogeneity, completeness, v_measure = homogeneity_completeness_v_measure(
        true_array, pred_array
    )
    return ClusteringMetrics(
        ari=float(adjusted_rand_score(true_array, pred_array)),
        nmi=float(normalized_mutual_info_score(true_array, pred_array)),
        fmi=float(fowlkes_mallows_score(true_array, pred_array)),
        homogeneity=float(homogeneity),
        completeness=float(completeness),
        v_measure=float(v_measure),
        purity=cluster_purity(true_array, pred_array),
        hungarian_accuracy=bipartite_hungarian_matching_accuracy(true_array, pred_array),
    )


def format_metrics(metrics: ClusteringMetrics) -> str:
    """Pretty-print metrics as a multi-line string for logs / reports."""
    rows: Sequence[tuple[str, float]] = (
        ("Adjusted Rand Index (ARI)", metrics.ari),
        ("Normalized Mutual Information (NMI)", metrics.nmi),
        ("Fowlkes-Mallows Index (FMI)", metrics.fmi),
        ("Homogeneity", metrics.homogeneity),
        ("Completeness", metrics.completeness),
        ("V-Measure", metrics.v_measure),
        ("Cluster Purity", metrics.purity),
        ("Hungarian Matching Accuracy", metrics.hungarian_accuracy),
    )
    width = max(len(label) for label, _ in rows)
    return "\n".join(f"{label:<{width}}  {value:.3f}" for label, value in rows)


__all__ = [
    "ClusteringMetrics",
    "bipartite_hungarian_matching_accuracy",
    "cluster_purity",
    "clustering_metrics",
    "format_metrics",
]
