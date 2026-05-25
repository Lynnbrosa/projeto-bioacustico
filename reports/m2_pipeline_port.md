# M2 — Re-implementação mínima do pipeline

## Objetivo

Portar `dataset.py`, `model.py`, `preprocessor.py` e `evaluation.py` do upstream pro pacote `bioacid/`, com tipagem estrita e imports preguiçosos pra manter CI rápido. Cobrir as quatro métricas oficiais (ARI, NMI, FMI, Hungarian matching accuracy).

## Estrutura final

```
src/bioacid/
├── cli.py            # CLI mínima --version
├── cluster.py        # UMAP / TSNE + HDBSCAN
├── data.py           # AIIDLocalizedClipDataset + load_clip_table
├── evaluate.py       # ClusteringMetrics + funções de cluster vs truth
├── losses.py         # placeholder (M3)
├── models.py         # build_resnet18_1ch + Resnet18Classifier + load_ovenbird_checkpoint
├── preprocessor.py   # OvenbirdPreprocessor (bandpass 2-10 kHz, 2s, time jitter)
└── train.py          # train_supervised (cross-entropy)
```

## Resultados — `scripts/03_evaluate_clusters.py`

Carrega embeddings salvos pelo `scripts/01_reproduce_demo.py` (checkpoint upstream), aplica UMAP → HDBSCAN e reporta as métricas.

```
HDBSCAN: 7 clusters, 0 noise points

Adjusted Rand Index (ARI)            0.651
Normalized Mutual Information (NMI)  0.887
Fowlkes-Mallows Index (FMI)          0.728
Homogeneity                          0.797
Completeness                         1.000
V-Measure                            0.887
Cluster Purity                       0.700
Hungarian Matching Accuracy          0.700
```

Comentário: HDBSCAN num espaço UMAP 5d encontrou 7 clusters quando há 10 indivíduos. A `completeness=1.0` confirma que indivíduos não foram divididos entre clusters; o gargalo é a fusão de indivíduos próximos (`homogeneity=0.80`). 1-NN LOO no espaço 512d original (medido no M1) atinge 0.98 — sugere que o problema não é o extrator mas o ajuste de parâmetros do HDBSCAN. Tuning fica pro M3.

## `scripts/02_train_baseline.py`

End-to-end usando o nosso pacote:

1. Carrega CSV via `bioacid.data.load_clip_table`.
2. Filtra split `train` (~80 clipes, 10 indivíduos).
3. Treina `Resnet18Classifier` com cross-entropy via `bioacid.train.train_supervised`.
4. Embeda a amostra completa, roda UMAP+HDBSCAN, reporta métricas.

Esperado: métricas muito abaixo do checkpoint upstream (este foi treinado em 234 indivíduos, não 10). O propósito é validar que o pipeline portado roda — não bater números do paper na sample.

## Cobertura de testes

```
$ uv run pytest -q
..................                                                       [100%]
18 passed in 1.02s
```

- `tests/test_smoke.py` — 9 testes (import de todos os submódulos + CLI return code).
- `tests/test_evaluate.py` — 9 testes (perfeição, aleatoriedade, edge cases das métricas).

CI usa só extras `[dev]` (sem torch / opensoundscape). Módulos com deps pesadas (`models`, `train`, `data`, `preprocessor`, `cluster`) importam tudo dentro das funções pra que `import bioacid.models` funcione mesmo sem ML extras.

## Próximo passo

M3: matriz de experimentos (ResNet18 vs ResNet50 vs EfficientNet-B0 vs ConvNeXt-Tiny × cross-entropy vs ArcFace vs SupCon × mel vs linear-freq × SpecAugment on/off).
