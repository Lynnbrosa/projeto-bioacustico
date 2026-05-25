# notebooks/

Notebooks são tratados como artefatos exploratórios. Regras:

1. Não importar de notebooks no código de produção (`src/bioacid/`).
2. Limpar outputs antes de commitar (`jupyter nbconvert --clear-output --inplace`).
3. Numerar por ordem (`01_eda_sample.ipynb`, `02_embedding_explore.ipynb`, ...).
