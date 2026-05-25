# data/

Conteúdo desta pasta é **gitignored**. Apenas este README é versionado.

## Layout

- `raw/` — gravações brutas baixadas (Xeno-canto, upstream sample) e metadados originais.
- `interim/` — artefatos intermediários (clipes recortados, espectrogramas em cache, splits).
- `processed/` — produtos finais consumidos por treino e avaliação (embeddings em parquet, etc.).

## Convenções

- Não commitar arquivos grandes (audio, modelos, embeddings) aqui. Use storage externo se necessário (S3, Hugging Face datasets, Zenodo).
- Cada subpasta deve poder ser regenerada por um script em `scripts/`.
