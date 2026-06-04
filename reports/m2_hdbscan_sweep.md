# M2 — HDBSCAN hyperparameter sweep

Best Hungarian accuracy: **0.810** (reduction=tsne, dims=2, min_cluster_size=4, min_samples=1).

Comparison to baseline (umap, dims=5, min_cluster_size=5, min_samples=None):

| Setup | Hungarian | ARI | NMI | Purity | n_clusters |
| --- | ---: | ---: | ---: | ---: | ---: |
| Baseline (M2 default) | 0.700 | 0.651 | 0.887 | 0.700 | 7 |
| **Best from sweep** | **0.810** | 0.821 | 0.935 | 0.900 | 11 |

## Top 15 configurations

| Reduction | Dims | min_clust | min_samp | Hungarian | ARI | NMI | Purity | Clusters | Noise |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| tsne | 2 | 4 | 1 | 0.810 | 0.821 | 0.935 | 0.900 | 11 | 1 |
| tsne | 2 | 4 | 2 | 0.810 | 0.821 | 0.935 | 0.900 | 11 | 1 |
| umap | 2 | 4 | 1 | 0.770 | 0.789 | 0.922 | 0.900 | 12 | 1 |
| umap | 2 | 4 | 2 | 0.770 | 0.789 | 0.922 | 0.900 | 12 | 1 |
| tsne | 2 | 2 | 3 | 0.760 | 0.774 | 0.904 | 0.880 | 12 | 3 |
| tsne | 2 | 3 | None | 0.760 | 0.774 | 0.904 | 0.880 | 12 | 3 |
| tsne | 2 | 3 | 3 | 0.760 | 0.774 | 0.904 | 0.880 | 12 | 3 |
| tsne | 2 | 3 | 1 | 0.750 | 0.776 | 0.914 | 0.900 | 13 | 1 |
| tsne | 2 | 3 | 2 | 0.750 | 0.776 | 0.914 | 0.900 | 13 | 1 |
| tsne | 2 | 4 | None | 0.710 | 0.647 | 0.882 | 0.710 | 7 | 1 |
| tsne | 2 | 5 | None | 0.710 | 0.647 | 0.882 | 0.710 | 7 | 1 |
| none | 2 | 2 | 3 | 0.700 | 0.618 | 0.850 | 0.700 | 7 | 6 |
| none | 2 | 3 | None | 0.700 | 0.618 | 0.850 | 0.700 | 7 | 6 |
| none | 2 | 3 | 1 | 0.700 | 0.626 | 0.857 | 0.700 | 7 | 4 |
| none | 2 | 3 | 2 | 0.700 | 0.626 | 0.857 | 0.700 | 7 | 4 |

Note: the upstream-checkpoint embeddings have a 1-NN LOO accuracy of 0.980 — they encode individual identity well. The clustering step's parameter choice translates that into discoverable structure.
