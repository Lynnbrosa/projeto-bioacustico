# Bioacoustic Individual Identification

Contexto persistente do projeto. Este arquivo é a fonte de verdade pro Claude Code sobre o que estamos construindo, o que já existe e como decidir trade-offs ao longo do caminho.

## Missão

Construir um pipeline de identificação individual de aves por som (AIID) em dados de monitoramento acústico passivo (PAM), com dataset 100% público, replicando o estado da arte (Lapp et al., bioRxiv 2025) e estendendo-o para pelo menos uma espécie neotropical.

A contribuição alvo do projeto não é "mais um classificador". É demonstrar que o approach de location-pseudo-labels do paper original generaliza para outra espécie usando dados de Xeno-canto, e fornecer um pipeline open-source reproduzível que outros grupos brasileiros possam adaptar.

## Estado atual

O repositório de referência `sammlapp/ovenbird-individual-recognition` (preprint Lapp et al. 2025, bioRxiv 2025.06.26.661638) já foi clonado e inspecionado. Resumo do que ele entrega:

- Sample dataset público: 100 clipes MP3 de 10 indivíduos de Ovenbird, com metadados completos (timestamp, posição XY do pássaro estimada por localização acústica, distância ao microfone, split treino/val/test) em `sample_data/labeled_clips_sample.csv`. Banda de frequência relevante: 2 a 10 kHz.
- Modelo pré-treinado: ResNet18 PyTorch em `checkpoints/full_2025-04-10T11:02:36.028451_best.pth` (44 MB), pronto pra inferência.
- Pipeline em 6 fases: detecção de espécie (HawkEars), treino do extrator (location-pseudo-labels), embeddings, descoberta de indivíduos via HDBSCAN, histórias de recapture, modelagem ecológica CJS.
- Código modular: `src/dataset.py`, `src/model.py`, `src/loss.py`, `src/preprocessor.py`, `src/evaluation.py`.
- Demo notebook: `demo.ipynb` reproduz o pipeline na sample em 1-2 min de GPU ou 3-10 min de CPU.
- Dataset PAM completo (126 sites, 4 anos, 405 indivíduos): não público. Mediante pedido aos autores. Fora do nosso escopo.

Stack do repo upstream:

- Python 3.11, PyTorch
- OpenSoundscape (Kitzes Lab, licença MIT) para preprocessing e localization
- pytorch-metric-learning, UMAP, HDBSCAN
- W&B para tracking
- HawkEars como classificador de espécie

## Reference Implementation

- Paper: Lapp et al. 2025, "Automated identification of individual birds by song enables multi-year recapture from passive acoustic monitoring data", bioRxiv 10.1101/2025.06.26.661638.
- Repo: https://github.com/sammlapp/ovenbird-individual-recognition
- Achievement reportado: 96% de acurácia em re-identificação, 405 indivíduos rastreados em 4 anos, sobrevivência aparente de 0.70.
- Insight central: usar localização espacial como pseudo-label de identidade. Clipes vindos da mesma posição num intervalo curto são tratados como mesmo indivíduo. Treina um extrator de features com contrastive ou supervised classification e depois clusteriza embeddings com HDBSCAN.
- Achados secundários do paper: ResNet18 + supervised classification venceu ArcFace e contrastive learning. Vale considerar ao escolher loss.

## Arquitetura do nosso projeto

Estrutura proposta (criar no início):

```
.
├── CLAUDE.md                    # este arquivo
├── README.md                    # apresentação pública (sem segredos)
├── pyproject.toml               # uv / hatch / pdm, à escolha
├── configs/                     # YAMLs por experimento
│   ├── ovenbird_baseline.yml
│   ├── ovenbird_arcface.yml
│   └── neotropical_extension.yml
├── src/bioacid/                 # pacote Python
│   ├── data.py                  # datasets e samplers
│   ├── models.py                # arquiteturas
│   ├── losses.py                # contrastive, supervised, ArcFace
│   ├── preprocessor.py
│   ├── train.py
│   ├── evaluate.py
│   └── cli.py
├── scripts/
│   ├── 01_reproduce_demo.py     # smoke-test em sample_data
│   ├── 02_train_baseline.py
│   ├── 03_evaluate_clusters.py
│   ├── 04_xeno_canto_pull.py    # baixar candidata neotropical
│   └── 05_extend_to_species.py
├── notebooks/
│   ├── 01_eda_sample.ipynb
│   ├── 02_embedding_explore.ipynb
│   └── 03_cluster_diagnostics.ipynb
├── tests/
├── data/
│   ├── raw/                     # gitignored
│   ├── interim/
│   └── processed/
└── reports/
    └── figures/
```

Decisões já tomadas:

- Linguagem: Python 3.11.
- Package manager: `uv` (rápido, lockfile reproduzível). Se houver dor de integração com OpenSoundscape, fallback pra pip clássico.
- Tracking: W&B em modo local primeiro, com possibilidade de subir runs públicos no final.
- Versionamento de dados: por enquanto sem DVC. Avaliar quando o dataset neotropical estiver mais consolidado.
- Configs: YAMLs simples, compatíveis com o estilo do upstream. Hydra fica como possibilidade futura se a matriz de experimentos explodir.
- Loss padrão: supervised classification cross-entropy. ArcFace e contrastive como variações secundárias (paper original mostrou que contrastive performou pior, replicar isso é parte da contribuição honesta).
- Backbone padrão: ResNet18. Comparar com HawkEars (transfer learning) e backbones do `timm` (efficientnet_b0, convnext_tiny) em fase de experimentação.
- Preprocessing: bandpass 2-10 kHz, mel ou linear-freq spectrogram, janelas de 1.5 s, log-amplitude em dBFS de -55 a -10. Replicar exatamente o do upstream antes de experimentar variações.

## Milestones

Cada milestone deve render commit limpo, ao menos um teste e atualização do README quando relevante.

### M0: Bootstrap (1-2 sessões)

- Inicializar repo com `uv init`, `.gitignore`, MIT license.
- Estrutura de pastas conforme acima.
- README mínimo (visão, instalação, como rodar o demo).
- CI simples (lint + smoke test) em GitHub Actions.

### M1: Reproduzir o demo (1 sessão)

- Submódulo ou clone separado do repo upstream em `external/ovenbird-individual-recognition`.
- Rodar `demo.ipynb` com checkpoint pré-treinado.
- Documentar tempo de execução, resultados, e quaisquer ajustes de ambiente.
- Salvar embeddings da sample como artefato em `data/processed/sample_embeddings.parquet`.

### M2: Re-implementar o pipeline mínimo no nosso pacote (2-3 sessões)

- Portar `dataset.py`, `model.py`, `preprocessor.py`, `evaluation.py` para `src/bioacid/`, simplificando onde fizer sentido (remover dependências que não usaremos).
- Script `02_train_baseline.py` que treina ResNet18 supervised na sample e reproduz métricas declaradas.
- Métricas obrigatórias: Adjusted Rand Index (ARI), Normalized Mutual Information (NMI), Fowlkes-Mallows (FMI), Hungarian matching accuracy.
- Comparar com checkpoint upstream pra validar a porta.

### M3: Matriz de experimentos (3-5 sessões)

- Variações de backbone: ResNet18, ResNet50, EfficientNet-B0, ConvNeXt-Tiny.
- Variações de loss: cross-entropy supervisionado, ArcFace, supervised contrastive (SupCon).
- Variações de preprocessing: mel vs linear-freq, PCEN vs log-mel.
- Pelo menos uma run com SpecAugment ativo vs desativado.
- Resultado: tabela em `reports/experiment_table.md` e figura comparativa em `reports/figures/`.

### M4: Extensão neotropical (5+ sessões)

- Escolher espécie alvo. Candidata default: sabiá-laranjeira (Turdus rufiventris). Critérios: ave nacional, canto estereotipado e individualmente distintivo, alto volume em Xeno-canto, presença em ambientes urbanos e florestais.
- Script `04_xeno_canto_pull.py` para baixar gravações via API do Xeno-canto, com filtros de qualidade (quality A/B), duração mínima, e país (Brasil).
- Como não temos localização espacial pra usar como pseudo-label, a estratégia é diferente: usar identidade do recordista + data + local de gravação como pseudo-label provisório (gravações do mesmo lote são provavelmente do mesmo indivíduo). Documentar essa heurística como contribuição metodológica, com suas limitações.
- Avaliação contra subset manualmente curado (Lynn escolhe e rotula 30-50 clipes).
- Resultado: paper-style report em `reports/neotropical_extension.md`.

## Convenções de trabalho

- Branches: `main` sempre rodável. Trabalho em `feat/...` ou `exp/...`.
- Commits: imperativos, em inglês, escopo no início (`data:`, `model:`, `eval:`, `docs:`, `chore:`).
- Estilo: `ruff` pra lint e format. `mypy` strict no módulo `bioacid`. Scripts em `scripts/` mais permissivos.
- Testes: pelo menos um smoke test por módulo. Sample dataset existe pra isso, usar.
- Notebooks: tratados como artefatos exploratórios. Não importar de notebooks no código de produção. Limpar outputs antes de commitar.
- Idioma: docstrings e código em inglês. Documentos de pesquisa (`reports/`) podem ser em português ou inglês conforme finalidade.

## Out of scope (importante manter)

- Treinar do zero no dataset PAM completo do paper (não temos acesso).
- Deploy em produção, API, Railway. Projeto é pesquisa, não produto.
- Modelagem ecológica em R/JAGS (CJS survival models do paper). Fora do escopo a menos que o projeto evolua pra colaboração com ecólogo.
- Coleta de dados próprios via gravadores. Considerável só depois de M4.
- Hardware embarcado, edge inference, ONNX export. Considerar só se o projeto virar produto.

## Quick start

```bash
# Clonar repo upstream pra referência
git clone https://github.com/sammlapp/ovenbird-individual-recognition.git external/upstream

# Setup do nosso projeto
uv init
uv add torch torchvision opensoundscape pytorch-metric-learning \
       scikit-learn pandas numpy tqdm umap-learn hdbscan \
       wandb seaborn matplotlib bioacoustics_model_zoo

# Smoke test
uv run scripts/01_reproduce_demo.py
```

## Princípios pra decisões em runtime

Quando precisar decidir algo não explicitamente coberto aqui:

1. Privilegiar reprodutibilidade sobre cleverness. Se o upstream fez de um jeito, replicar primeiro e só divergir depois de medir.
2. Sample dataset é a fronteira inicial. Tudo deve rodar nela em minutos. Nada de pedir paciência pra horas de treino antes do M3.
3. Honestidade científica. Se um experimento der ruim, documenta. Resultados negativos importam, especialmente em re-id onde a literatura é otimista demais.
4. Não inflar o escopo. Cada milestone tem fim. Resistir à tentação de "só mais uma coisinha". Se aparecer ideia legal fora do milestone atual, registrar em `IDEAS.md`.
5. Citações no commit message quando porta de código upstream (`port: ssl_location_loss from upstream/src/loss.py`).

## Estado do diálogo até aqui (contexto humano)

O dono do projeto é desenvolvedor experiente, vem de SOC e atualmente trabalha com automação de vendas + IA. Esse projeto é pesquisa séria pessoal, sem viés acadêmico institucional. Critérios não-negociáveis durante o brainstorm de escopo: dataset 100% público, pipeline completo coleta-treino-inferência, problema com cara de paper. Identificação individual foi o ângulo metodológico que fisgou (vindo de uma discussão prévia sobre baleias jubarte que foi descartada por exigir parceria pra dados brasileiros).
