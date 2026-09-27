#!/usr/bin/env sh
set -eu

echo "Construction de l'image Docker..."
docker compose build

echo "Détection de CUDA dans le conteneur..."
if docker compose -f compose.yaml -f compose.gpu.yaml run --rm --no-deps notebook \
    python -c 'import torch; raise SystemExit(0 if torch.cuda.is_available() else 1)'; then
    echo "GPU NVIDIA détecté : lancement avec CUDA."
    docker compose -f compose.yaml -f compose.gpu.yaml up
else
    echo "GPU Docker indisponible : poursuite automatique sur CPU."
    docker compose up
fi
