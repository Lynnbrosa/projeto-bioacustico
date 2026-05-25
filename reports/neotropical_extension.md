# M4 — Extensão neotropical (sabiá-laranjeira)

## Objetivo

Demonstrar que o pipeline de identificação individual de Lapp et al. 2025 generaliza para uma espécie brasileira usando dados 100% públicos do Xeno-canto, contornando o requisito original de arrays de localização acústica.

Espécie alvo: **sabiá-laranjeira** (*Turdus rufiventris*). Critérios: ave nacional, canto estereotipado e individualmente distintivo, alto volume de gravações no Xeno-canto (centenas em quality A/B no Brasil), presença em ambientes urbanos e florestais.

## Contribuição metodológica

O dataset PAM de Lapp et al. inclui posição XY do pássaro estimada por localização acústica — não temos isso no Xeno-canto. **Substituição proposta**: heurística *recordist + date + locality* como pseudo-label.

### Regra

Duas gravações são tratadas como o mesmo indivíduo quando:

1. Mesmo **recordist** (string exata, descarta gravações anônimas).
2. **Datas** dentro de `time_window_days` (default: 1 dia).
3. **Localidade** idêntica como string, OU coordenadas a menos de `location_radius_m` (default: 500 m).

Implementação: [`bioacid/xeno_canto.py:assign_pseudo_labels`](../src/bioacid/xeno_canto.py).

### Por que essa heurística

- **Recordist**: um observador que sai a campo num dia normalmente foca em poucos indivíduos. Mudar de recordista quase sempre significa expedição diferente (e provavelmente ave diferente).
- **Date + locality**: aves territoriais como o sabiá-laranjeira tendem a ficar no mesmo território por dias. Gravações do mesmo lote (mesmo dia, mesmo lugar) têm alta probabilidade de capturar o mesmo macho cantando.
- **Threshold de 500 m**: empírico. O território típico de sabiá em ambiente urbano é ~50-150 m de raio; 500 m absorve imprecisão do GPS sem fundir territórios vizinhos significativamente.

### Limitações reconhecidas

| Caso | Efeito esperado |
| --- | --- |
| Um recordist grava 2 machos num parque no mesmo dia | False positive: pseudo-label funde dois indivíduos. |
| Mesma ave gravada por dois recordistas em dias diferentes | False negative: pseudo-label trata como dois indivíduos. |
| Localidade preenchida só em texto livre (sem lat/lng) | Threshold de raio não aplicável; depende de igualdade exata. |
| Recordist anônimo (`""`) | Excluído do clustering por construção. |

Mitigação: filtrar para `len_gt:5` (clipes longos têm mais conteúdo identificável) e quality A/B (reduz ruído).

## Pipeline

### 1. Pulling (`scripts/04_xeno_canto_pull.py`)

```bash
uv run --extra ml --extra dev scripts/04_xeno_canto_pull.py \
    --species "Turdus rufiventris" \
    --country brazil \
    --min-length-s 5 \
    --max-recordings 100
```

Salva em `data/raw/xenocanto/turdus_rufiventris/`:

- `metadata.json` — lista de gravações com pseudo-label atribuído.
- `<id>.mp3` — arquivos de áudio brutos do Xeno-canto.

### 2. Segmentação + Embedding + Clustering (`scripts/05_extend_to_species.py`)

```bash
uv run --extra ml --extra dev scripts/05_extend_to_species.py \
    --metadata data/raw/xenocanto/turdus_rufiventris/metadata.json \
    --pretrained-checkpoint external/upstream/checkpoints/full_2025-04-10T11:02:36.028451_best.pth
```

Cada gravação é fatiada em janelas de 2 s (stride 2 s), embedada pelo backbone (ResNet18 do checkpoint Ovenbird, *transfer learning*), reduzida via UMAP e clusterizada com HDBSCAN. Métricas (ARI, NMI, FMI, Hungarian) comparam o cluster predito contra a pseudo-label.

### Notas sobre transfer

O checkpoint Ovenbird foi treinado em outro passarinho (Ovenbird, *Seiurus aurocapilla*), num habitat e banda de frequência (2-10 kHz) que casa razoavelmente com a faixa do sabiá. Usar como extrator de features sem fine-tuning é a abordagem mais barata; se as métricas ficarem fracas, o próximo passo é fine-tunar o backbone usando as próprias pseudo-labels como supervisão (mesma lógica do paper original).

## Estado da execução

**Não executado neste ambiente**: o sandbox onde M0-M3 rodaram não tem `xeno-canto.org` na allowlist (`curl https://xeno-canto.org` retorna `Host not in allowlist`). Os dois scripts foram escritos de forma defensiva (falham com mensagem clara em vez de exceção crua) e os módulos pertinentes têm cobertura de testes offline:

```
$ uv run pytest tests/test_xeno_canto.py -q
.............                                                            [100%]
13 passed in 0.05s
```

Para rodar end-to-end:

1. Em qualquer máquina com acesso aberto à internet:
   ```bash
   uv sync --extra ml --extra dev
   uv run scripts/04_xeno_canto_pull.py
   uv run scripts/05_extend_to_species.py --metadata data/raw/xenocanto/turdus_rufiventris/metadata.json
   ```
2. Ou habilitar `xeno-canto.org` na política de rede do sandbox.

## Próximos passos

- **Subset manualmente curado**: escolher 30-50 gravações onde a identidade individual seja certa (mesma gravação multipart ou gravações sequenciais de um único macho), pra calibrar quão otimista/pessimista a pseudo-label está.
- **Fine-tuning**: treinar o backbone do upstream nas pseudo-labels do sabiá com cross-entropy supervisionado; comparar embeddings antes/depois.
- **Outras espécies**: a heurística é genérica; rodar para uirapuru, jaó, sanhaço-do-coqueiro, etc. compara robustez metodológica entre clades.
- **Crítica honesta**: comparar pseudo-label vs subset manualmente rotulado pra reportar precision/recall da heurística como contribuição em si.

## Referências

- Lapp et al. 2025, "Automated identification of individual birds by song...", bioRxiv 10.1101/2025.06.26.661638.
- [Xeno-canto API docs](https://xeno-canto.org/explore/api).
- Sick, H. (1997). *Ornitologia Brasileira* — descrição do canto do sabiá-laranjeira como elemento territorial individualmente distintivo.
