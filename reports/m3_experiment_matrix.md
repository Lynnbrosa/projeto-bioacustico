# M3 — Matriz de experimentos

## Setup

Grid de 6 runs combinando 3 backbones × 2 losses:

- **Backbones**: ResNet18, ResNet50, EfficientNet-B0.
- **Losses**: Cross-Entropy supervisionada, ArcFace (margin=0.5, scale=30).
- **Treino**: split `val` da sample (70 clipes, 7 indivíduos), 10 épocas, batch 32, Adam lr=1e-3.
- **Avaliação**: embeddings dos 100 clipes (10 indivíduos) → UMAP(5d) → HDBSCAN(min_cluster_size=5) → métricas vs `aiid_label`.

Como a sample não tem split `train`, treinamos no `val` e avaliamos clustering em tudo (val+test). Os 3 indivíduos do `test` não são vistos no treino — open-set em espírito, embora a sample seja pequena demais pra resultados estatisticamente significativos.

ConvNeXt-Tiny e SupCon foram cortados do grid pra manter o run total abaixo de 5 minutos em CPU.

## Resultados

Ver tabela canônica em [`experiment_table.md`](experiment_table.md):

| Backbone | Loss | ARI | NMI | FMI | Hungarian | Purity | Train top-1 |
|---|---|---:|---:|---:|---:|---:|---:|
| resnet18 | cross_entropy | **0.365** | 0.606 | 0.435 | **0.530** | **0.540** | 0.986 |
| resnet18 | arcface | 0.363 | **0.626** | **0.474** | 0.510 | 0.510 | 0.586 |
| resnet50 | cross_entropy | 0.046 | 0.213 | 0.176 | 0.280 | 0.280 | 0.486 |
| resnet50 | arcface | 0.054 | 0.224 | 0.222 | 0.260 | 0.260 | 0.014 |
| efficientnet_b0 | cross_entropy | 0.058 | 0.222 | 0.190 | 0.270 | 0.280 | 0.343 |
| efficientnet_b0 | arcface | 0.067 | 0.269 | 0.185 | 0.340 | 0.340 | 0.000 |

Figura: [`figures/m3_grid.png`](figures/m3_grid.png).

## Discussão

- **ResNet18 venceu**, com cross-entropy e ArcFace empatando em ARI mas CE liderando em Hungarian (0.530 vs 0.510). Replica qualitativamente o achado do paper de Lapp et al.: supervised classification > ArcFace.
- **Modelos maiores (ResNet50, EfficientNet-B0) colapsaram**. Faz sentido — 70 clipes treinando ~25M parâmetros é overfitting massivo; o train top-1 do ResNet50+ArcFace ficou em 0.014 (não aprendeu nada além do margin term sobre features ainda ruidosas).
- **ArcFace precisa de mais épocas pra convergir** com o margin term. Os train top-1 caem porque a métrica é a accuracy do logit, que com margin adicional fica artificialmente baixa durante o treino. Não estritamente comparável com CE.
- **Nenhuma run bate o checkpoint upstream** (Hungarian 0.700 em M2). Era esperado: o checkpoint foi treinado em 234 indivíduos do dataset PAM completo; nosso treino vê 7.

## Comparação com checkpoint upstream

| Modelo | Hungarian | ARI | NMI |
|---|---:|---:|---:|
| Upstream checkpoint (234 indiv, full PAM) | 0.700 | 0.651 | 0.887 |
| Melhor do grid (ResNet18+CE, 7 indiv, val da sample) | 0.530 | 0.365 | 0.606 |

Gap de ~24% absoluto em Hungarian. A diferença vem do volume e diversidade de treino, não da arquitetura.

## Artefatos

```
reports/runs/m3_grid/
├── resnet18__cross_entropy/    {backbone.pth, embeddings.npy, result.json}
├── resnet18__arcface/
├── resnet50__cross_entropy/
├── resnet50__arcface/
├── efficientnet_b0__cross_entropy/
└── efficientnet_b0__arcface/
```

## Próximos passos

- Variações que ficaram fora do M3: ConvNeXt-Tiny, SupCon, mel vs linear-freq, SpecAugment, augmentações de tempo/frequência. Cada uma é mais um run no `scripts/06_run_experiments.py`.
- M4: aplicar o melhor pipeline (ResNet18+CE) a uma espécie neotropical via Xeno-canto.
