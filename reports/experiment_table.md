# M3 — Matriz de experimentos

Grid: `6` runs, 10 épocas, batch 32, lr 0.001.

Treino feito no split `val` (70 clipes, 7 indivíduos); avaliação por clustering em todos os 100 clipes (10 indivíduos — 3 não vistos no treino, open-set re-id em espírito).

| Backbone | Loss | ARI | NMI | FMI | Hungarian | Purity | Train top-1 | Train (s) |
| --- | --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| resnet18 | cross_entropy | 0.365 | 0.606 | 0.435 | 0.530 | 0.540 | 0.986 | 22 |
| resnet18 | arcface | 0.363 | 0.626 | 0.474 | 0.510 | 0.510 | 0.586 | 21 |
| resnet50 | cross_entropy | 0.046 | 0.213 | 0.176 | 0.280 | 0.280 | 0.486 | 58 |
| resnet50 | arcface | 0.054 | 0.224 | 0.222 | 0.260 | 0.260 | 0.014 | 54 |
| efficientnet_b0 | cross_entropy | 0.058 | 0.222 | 0.190 | 0.270 | 0.280 | 0.343 | 31 |
| efficientnet_b0 | arcface | 0.067 | 0.269 | 0.185 | 0.340 | 0.340 | 0.000 | 30 |

## Artefatos

- Pesos, embeddings e métricas em JSON por run: `reports/runs/m3_grid/<exp_id>/`.
- Figura comparativa: `reports/figures/m3_grid.png`.

## Limitações

- Sample dataset (100 clipes) é insuficiente pra distinguir variâncias estatísticas entre configurações. Use estes números como sanity check do pipeline; conclusões científicas pedem o dataset PAM completo.
