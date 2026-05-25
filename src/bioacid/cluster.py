"""Dimensionality reduction + clustering for individual identification.

Wraps the same UMAP/TSNE + HDBSCAN pipeline used by upstream's
``cluster_points`` into a small, type-hinted API.

UMAP, HDBSCAN and sklearn TSNE are imported lazily so the bioacid package
imports cleanly in dev-only environments (CI).
"""

from __future__ import annotations

from typing import Literal

import numpy as np
from numpy.typing import NDArray

ReductionAlgorithm = Literal["umap", "tsne", "none"]


def reduce_dims(
    embeddings: NDArray[np.floating],
    *,
    algorithm: ReductionAlgorithm = "umap",
    n_components: int = 5,
    random_state: int | None = None,
) -> NDArray[np.floating]:
    """Reduce ``embeddings`` to ``n_components`` dimensions.

    ``algorithm="none"`` returns the input unchanged (with a copy).
    """
    if algorithm == "none":
        return embeddings.copy()
    if algorithm == "umap":
        import umap

        reducer = umap.UMAP(n_components=n_components, random_state=random_state)
        return np.asarray(reducer.fit_transform(embeddings), dtype=embeddings.dtype)
    if algorithm == "tsne":
        from sklearn.manifold import TSNE

        reducer = TSNE(n_components=n_components, random_state=random_state)
        return np.asarray(reducer.fit_transform(embeddings), dtype=embeddings.dtype)
    raise ValueError(f"unknown reduction algorithm: {algorithm}")


def hdbscan_cluster(
    features: NDArray[np.floating],
    *,
    min_cluster_size: int = 5,
    min_samples: int | None = None,
) -> NDArray[np.intp]:
    """Run HDBSCAN and return the cluster labels (``-1`` for noise)."""
    from sklearn.cluster import HDBSCAN

    clusterer = HDBSCAN(min_cluster_size=min_cluster_size, min_samples=min_samples)
    labels = clusterer.fit_predict(features)
    return np.asarray(labels, dtype=np.intp)


def cluster_embeddings(
    embeddings: NDArray[np.floating],
    *,
    reduction_algorithm: ReductionAlgorithm = "umap",
    reduced_n_dimensions: int = 5,
    min_cluster_size: int = 5,
    min_samples: int | None = None,
    random_state: int | None = None,
) -> tuple[NDArray[np.intp], NDArray[np.floating]]:
    """End-to-end: reduce dims then HDBSCAN. Returns ``(labels, reduced)``."""
    reduced = reduce_dims(
        embeddings,
        algorithm=reduction_algorithm,
        n_components=reduced_n_dimensions,
        random_state=random_state,
    )
    labels = hdbscan_cluster(reduced, min_cluster_size=min_cluster_size, min_samples=min_samples)
    return labels, reduced


__all__ = ["ReductionAlgorithm", "cluster_embeddings", "hdbscan_cluster", "reduce_dims"]
