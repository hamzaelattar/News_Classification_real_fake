$ErrorActionPreference = "Stop"

Write-Host "Construction de l'image Docker..."
docker compose build
if ($LASTEXITCODE -ne 0) {
    throw "La construction Docker a échoué."
}

Write-Host "Détection de CUDA dans le conteneur..."
$gpuAvailable = $false
try {
    docker compose -f compose.yaml -f compose.gpu.yaml run --rm --no-deps notebook `
        python -c "import torch; raise SystemExit(0 if torch.cuda.is_available() else 1)"
    $gpuAvailable = ($LASTEXITCODE -eq 0)
}
catch {
    $gpuAvailable = $false
}

if ($gpuAvailable) {
    Write-Host "GPU NVIDIA détecté : lancement avec CUDA."
    docker compose -f compose.yaml -f compose.gpu.yaml up
} else {
    Write-Host "GPU Docker indisponible : poursuite automatique sur CPU."
    docker compose up
}
