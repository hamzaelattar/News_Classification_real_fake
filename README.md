# Classification Fake / Real News

Ce projet compare plusieurs modèles de classification de nouvelles politiques. Le notebook a été
réorganisé pour utiliser un seul pipeline de données, éviter les cellules dupliquées et séparer
clairement exploration, entraînement, évaluation et interprétation.

## Structure

```text
.
├── LLM.ipynb                         # notebook principal
├── Fake.csv/Fake.csv                 # articles étiquetés Fake (0)
├── True.csv/True.csv                 # articles étiquetés Real (1)
├── src/news_classification/
│   ├── data.py                       # chargement, nettoyage et split
│   ├── modeling.py                   # TF-IDF, métriques, interprétation
│   └── transformer_features.py       # DeBERTa gelé, optionnel
├── tests/
├── pyproject.toml
├── Dockerfile
└── compose.yaml
```

## Exécution locale avec uv

Après avoir [installé uv](https://docs.astral.sh/uv/getting-started/installation/) :

```bash
uv sync
uv run jupyter lab
```

Pour activer la section DeBERTa :

```bash
uv sync --extra transformers
RUN_FROZEN_DEBERTA=1 uv run jupyter lab
```

Les expériences GPU sont activées séparément avec les variables suivantes :

- `RUN_FROZEN_DEBERTA=1` : embeddings DeBERTa gelés ;
- `RUN_DEBERTA_FINETUNING=1` : fine-tuning complet de DeBERTa ;
- `RUN_QWEN_ZERO_SHOT=1` : classification zero-shot avec Qwen2.5-1.5B-Instruct.

Sous PowerShell, utilisez par exemple `$env:RUN_QWEN_ZERO_SHOT = "1"` avant de lancer Jupyter.

Exécution non interactive du notebook principal :

```bash
uv run jupyter nbconvert --to notebook --execute LLM.ipynb \
  --output artifacts/LLM.executed.ipynb --ExecutePreprocessor.timeout=-1
```

## Exécution avec Docker

```bash
docker compose up --build
```

Ouvrez ensuite <http://localhost:8888>. Le conteneur exécute par défaut la partie principale sans
les dépendances DeBERTa.

Pour construire également les dépendances DeBERTa, remplacez `UV_SYNC_EXTRAS: ""` par
`UV_SYNC_EXTRAS: "--extra transformers"`, puis activez l'expérience voulue dans `compose.yaml`.
Sans GPU NVIDIA configuré pour Docker, ces sections seront beaucoup plus lentes.

## Qualité du code

```bash
uv run pytest
uv run ruff check .
```

Le modèle ne doit pas être interprété comme un détecteur universel de désinformation. Le dataset
contient des signatures de sources très fortes (`Reuters`, `featured image`, etc.) susceptibles de
gonfler les résultats sur une séparation aléatoire.
