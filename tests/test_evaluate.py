"""Unit tests for bioacid.evaluate."""

from __future__ import annotations

import numpy as np
import pytest

from bioacid.evaluate import (
    bipartite_hungarian_matching_accuracy,
    cluster_purity,
    clustering_metrics,
    format_metrics,
)


def test_perfect_clustering_hits_all_metrics_at_one() -> None:
    truth = np.array([0, 0, 1, 1, 2, 2])
    pred = np.array([5, 5, 9, 9, 2, 2])

    metrics = clustering_metrics(truth, pred)

    assert metrics.ari == pytest.approx(1.0)
    assert metrics.nmi == pytest.approx(1.0)
    assert metrics.fmi == pytest.approx(1.0)
    assert metrics.homogeneity == pytest.approx(1.0)
    assert metrics.completeness == pytest.approx(1.0)
    assert metrics.v_measure == pytest.approx(1.0)
    assert metrics.purity == pytest.approx(1.0)
    assert metrics.hungarian_accuracy == pytest.approx(1.0)


def test_random_clustering_far_below_one() -> None:
    rng = np.random.default_rng(0)
    truth = np.tile(np.arange(10), 10)
    pred = rng.integers(0, 10, size=truth.shape)

    metrics = clustering_metrics(truth, pred)

    assert metrics.ari < 0.2
    assert metrics.nmi < 0.3


def test_hungarian_accuracy_handles_label_remap() -> None:
    truth = np.array([0, 0, 0, 1, 1, 1, 2, 2, 2])
    # Predictions are a permutation of truth labels — Hungarian should recover 1.0
    pred = np.array([2, 2, 2, 0, 0, 0, 1, 1, 1])
    assert bipartite_hungarian_matching_accuracy(truth, pred) == pytest.approx(1.0)


def test_hungarian_accuracy_one_wrong_sample() -> None:
    truth = np.array([0, 0, 0, 1, 1, 1])
    pred = np.array([0, 0, 1, 1, 1, 1])
    # Best matching: cluster 0 -> class 0 (2 hits), cluster 1 -> class 1 (3 hits)
    # 1 sample misassigned out of 6
    assert bipartite_hungarian_matching_accuracy(truth, pred) == pytest.approx(5 / 6)


def test_purity_handles_majority() -> None:
    truth = np.array([0, 0, 1, 1, 1])
    pred = np.array([7, 7, 7, 8, 8])
    # Cluster 7: majority is 0 (truth=[0,0,1] -> majority 0) -> 2 correct
    # Cluster 8: majority is 1 -> 2 correct
    # 4 / 5 correct
    assert cluster_purity(truth, pred) == pytest.approx(4 / 5)


def test_shape_mismatch_raises() -> None:
    with pytest.raises(ValueError, match="shape mismatch"):
        bipartite_hungarian_matching_accuracy(np.array([0, 1, 2]), np.array([0, 1]))
    with pytest.raises(ValueError, match="shape mismatch"):
        cluster_purity(np.array([0, 1, 2]), np.array([0, 1]))


def test_format_metrics_returns_8_lines() -> None:
    truth = np.array([0, 0, 1, 1])
    pred = np.array([0, 0, 1, 1])
    formatted = format_metrics(clustering_metrics(truth, pred))
    assert len(formatted.splitlines()) == 8


def test_as_dict_roundtrip() -> None:
    truth = np.array([0, 0, 1, 1])
    pred = np.array([0, 0, 1, 1])
    metrics = clustering_metrics(truth, pred)
    d = metrics.as_dict()
    assert set(d.keys()) == {
        "ari",
        "nmi",
        "fmi",
        "homogeneity",
        "completeness",
        "v_measure",
        "purity",
        "hungarian_accuracy",
    }
    assert all(isinstance(v, float) for v in d.values())
