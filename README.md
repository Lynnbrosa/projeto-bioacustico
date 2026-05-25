# bioacid

Pipeline open-source de identificação individual de aves por som em dados de monitoramento acústico passivo (PAM).

Este projeto é uma replicação independente do estado da arte descrito em [Lapp et al. 2025](https://doi.org/10.1101/2025.06.26.661638) ("Automated identification of individual birds by song enables multi-year recapture from passive acoustic monitoring data"), estendida com uma extensão para pelo menos uma espécie neotropical usando dados públicos do [Xeno-canto](https://xeno-canto.org/).

## Objetivos

1. Reproduzir o pipeline de location-pseudo-labels do paper original sobre o sample dataset público de Ovenbird.
2. Comparar variações de backbone (ResNet18/50, EfficientNet-B0, ConvNeXt-Tiny) e loss (cross-entropy supervisionado, ArcFace, SupCon).
3. Estender o método para uma espécie brasileira (candidata default: sabiá-laranjeira, *Turdus rufiventris*) usando gravações públicas, substituindo a pseudo-label de localização por uma heurística baseada em metadado de gravação.
4. Entregar pipeline reproduzível por outros grupos.

## Estado

Pre-alpha. Bootstrap em andamento (M0). Veja [milestones](#milestones) abaixo.

## Stack

- Python 3.11
- PyTorch + torchvision
- [OpenSoundscape](https://github.com/kitzeslab/opensoundscape) (preprocessing e localização)
- pytorch-metric-learning, scikit-learn, UMAP, HDBSCAN
- Weights & Biases para tracking
- `uv` para gerenciamento de dependências

## Instalação

```bash
# Clonar
git clone https://github.com/Lynnbrosa/projeto-bioacustico.git
cd projeto-bioacustico

# Setup com uv (recomendado)
uv sync --extra ml --extra dev

# Ou setup apenas dev (sem stack ML pesada) - útil pra CI / lint
uv sync --extra dev
```

## Quick start

```bash
# Clonar referência (upstream)
git clone https://github.com/sammlapp/ovenbird-individual-recognition.git external/upstream

# Smoke test (sample dataset publico, 1-2 min em GPU, 3-10 min em CPU)
uv run scripts/01_reproduce_demo.py
```

## Estrutura

```
.
├── configs/                 # YAMLs por experimento
├── src/bioacid/             # pacote Python
├── scripts/                 # entry points executaveis
├── notebooks/               # exploração (artefatos, nao importar de src)
├── tests/                   # pytest
├── data/                    # raw/interim/processed (gitignored)
├── reports/                 # resultados e figuras
└── external/                # clones de referência (gitignored)
```

## Milestones

- **M0 — Bootstrap**: estrutura de pastas, `pyproject.toml`, CI, README. *(em andamento)*
- **M1 — Reproduzir o demo**: rodar `demo.ipynb` do upstream com checkpoint pré-treinado.
- **M2 — Re-implementação mínima**: portar pipeline do upstream para `src/bioacid/`.
- **M3 — Matriz de experimentos**: variações de backbone, loss e preprocessing.
- **M4 — Extensão neotropical**: aplicar o método a uma espécie brasileira via Xeno-canto.

## Convenções

- Commits em inglês, imperativos, com escopo (`data:`, `model:`, `eval:`, `docs:`, `chore:`).
- Branches `main` sempre rodável; trabalho em `feat/...` ou `exp/...`.
- `ruff` para lint e format, `mypy` strict em `src/bioacid/`.

## Referências

- Lapp et al., "Automated identification of individual birds by song...", bioRxiv 10.1101/2025.06.26.661638, 2025.
- Repo de referência: [sammlapp/ovenbird-individual-recognition](https://github.com/sammlapp/ovenbird-individual-recognition).

## Licença

MIT. Veja [LICENSE](LICENSE).
