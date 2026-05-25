# M1 — Reprodução do demo upstream

## Objetivo

Rodar o demo do repositório de referência ([sammlapp/ovenbird-individual-recognition](https://github.com/sammlapp/ovenbird-individual-recognition)) com o checkpoint pré-treinado, gerar embeddings pra sample dataset pública e salvar como artefato.

## Setup

- Upstream clonado em `external/upstream/` via `git clone --depth 1`.
- Stack ML instalada via `uv sync --extra ml --extra dev`.
- Hardware: CPU (sandbox sem GPU acessível).

## Execução

Script: [`scripts/01_reproduce_demo.py`](../scripts/01_reproduce_demo.py).

```
$ uv run --extra ml --extra dev scripts/01_reproduce_demo.py
device: cpu
loaded feature extractor in 0.2s
sample: 100 clips, 10 individuals
embeddings in 1.8s, shape=(100, 512)
saved embeddings to data/processed/sample_embeddings.parquet
sanity check (1-NN LOO, cosine): 0.980
total: 2.1s
```

## Resultados

| Métrica | Valor |
| --- | --- |
| Clipes processados | 100 |
| Indivíduos | 10 (Ovenbird) |
| Dimensão do embedding | 512 |
| Tempo total (CPU) | ~2 s |
| 1-NN LOO accuracy (cosseno) | **0.980** |

A 1-NN LOO accuracy é um sanity check rápido, não uma métrica oficial — confirma que pares vindos do mesmo indivíduo são vizinhos mais próximos no espaço de embeddings. As métricas oficiais do paper (ARI, NMI, FMI, Hungarian matching) entram no M2 sobre clusterings HDBSCAN.

O resultado de 0.980 é consistente com os 0.96 reportados pelo paper sobre o dataset PAM completo (a sample é menor e mais limpa, logo o número aqui sobe).

## Ajustes feitos vs. demo upstream

O demo do upstream (`demo.ipynb`) chama `load_ovenbird_model()` que instancia `torchvision.models.resnet18(weights=IMAGENET1K_V1)` — isso dispara um download de pesos do `download.pytorch.org`, bloqueado pela política de rede do sandbox. Como o `state_dict` do checkpoint Ovenbird **sobrescreve todos** os parâmetros do ResNet18 logo em seguida, o download é desperdiçado.

Solução adotada no nosso script: replicamos a arquitetura `Resnet18_Classifier(num_classes=234)` localmente com `weights=None` e carregamos o checkpoint diretamente. O `state_dict` casa exatamente com a estrutura (chaves: `embedder.*` e `classifier.*`).

Outras simplificações:
- Não rodamos as visualizações TSNE / spectrograms do notebook (são exploratórias; ficam pra notebooks separados).
- Salvamos só os embeddings + label, formato parquet, pra serem reusados no M2.

## Artefato

[`data/processed/sample_embeddings.parquet`](../data/processed) — gitignored, gerado pelo script. Schema:

```
f000, f001, ..., f511, aiid_label
```

512 colunas float (features) + 1 coluna int (label do indivíduo). Indexado por `(file, start_time, end_time)`.

## Próximo passo

M2: re-implementar o pipeline mínimo (dataset, model, preprocessor, evaluation) no nosso pacote `bioacid` e validar que reproduzimos os mesmos embeddings sem depender de `external/upstream/src`.
