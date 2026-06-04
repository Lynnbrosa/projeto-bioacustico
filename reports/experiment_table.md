# M3 — Matriz de experimentos (baseline + ablações)

Grid: `9` runs, 10 épocas, batch 32, lr 0.001.

Treino feito no split `val` (70 clipes, 7 indivíduos); avaliação por clustering em todos os 100 clipes (10 indivíduos — 3 não vistos no treino, open-set re-id em espírito).

| Experiment | Backbone | Loss | SpecAug | ARI | NMI | FMI | Hungarian | Purity | Train top-1 | Train (s) |
| --- | --- | --- | :-: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| resnet18__cross_entropy | resnet18 | cross_entropy | no | 0.149 | 0.410 | 0.368 | 0.280 | 0.280 | 0.929 | 42 |
| resnet18__arcface | resnet18 | arcface | no | 0.321 | 0.556 | 0.416 | 0.540 | 0.540 | 0.786 | 33 |
| resnet50__cross_entropy | resnet50 | cross_entropy | no | 0.040 | 0.236 | 0.165 | 0.300 | 0.300 | 0.529 | 104 |
| resnet50__arcface | resnet50 | arcface | no | 0.014 | 0.187 | 0.141 | 0.250 | 0.250 | 0.000 | 90 |
| efficientnet_b0__cross_entropy | efficientnet_b0 | cross_entropy | no | 0.064 | 0.257 | 0.193 | 0.310 | 0.310 | 0.257 | 54 |
| efficientnet_b0__arcface | efficientnet_b0 | arcface | no | 0.063 | 0.255 | 0.185 | 0.300 | 0.300 | 0.000 | 50 |
| resnet18__cross_entropy__specaug | resnet18 | cross_entropy | yes | 0.141 | 0.343 | 0.245 | 0.350 | 0.350 | 0.900 | 68 |
| resnet18__supcon | resnet18 | supcon | no | 0.016 | 0.097 | 0.260 | 0.160 | 0.160 | 0.329 | 31 |
| convnext_tiny__cross_entropy | convnext_tiny | cross_entropy | no | 0.015 | 0.098 | 0.201 | 0.190 | 0.190 | 0.171 | 66 |

## Artefatos

- Pesos, embeddings e métricas em JSON por run: `reports/runs/m3_grid/<exp_id>/`.
- Figura comparativa: `reports/figures/m3_grid.png`.

## Limitações

- Sample dataset (100 clipes) é insuficiente pra distinguir variâncias estatísticas entre configurações. Use estes números como sanity check do pipeline; conclusões científicas pedem o dataset PAM completo.
