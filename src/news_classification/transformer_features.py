"""Extraction optionnelle d'embeddings avec un DeBERTa gelé."""

from __future__ import annotations

from collections.abc import Sequence
from time import perf_counter

import numpy as np


class FrozenDebertaEncoder:
    """Encode des textes par mean pooling sans entraîner DeBERTa."""

    def __init__(
        self,
        model_name: str = "microsoft/deberta-v3-base",
        *,
        max_length: int = 512,
    ) -> None:
        try:
            import torch
            from transformers import AutoModel, AutoTokenizer
        except ImportError as exc:
            raise RuntimeError(
                "Installez les dépendances avec `uv sync --extra transformers`."
            ) from exc

        self.torch = torch
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.max_length = max_length
        self.tokenizer = AutoTokenizer.from_pretrained(model_name)
        self.model = AutoModel.from_pretrained(model_name).to(self.device)
        self.model.eval()
        for parameter in self.model.parameters():
            parameter.requires_grad = False

    def encode(self, texts: Sequence[str], *, batch_size: int = 16) -> tuple[np.ndarray, float]:
        embeddings: list[np.ndarray] = []
        started = perf_counter()
        for start in range(0, len(texts), batch_size):
            batch = list(texts[start : start + batch_size])
            tokens = self.tokenizer(
                batch,
                padding=True,
                truncation=True,
                max_length=self.max_length,
                return_tensors="pt",
            )
            tokens = {name: value.to(self.device) for name, value in tokens.items()}
            with self.torch.no_grad():
                hidden = self.model(**tokens).last_hidden_state
                mask = tokens["attention_mask"].unsqueeze(-1).expand(hidden.size()).float()
                pooled = (hidden * mask).sum(1) / mask.sum(1).clamp(min=1e-9)
            embeddings.append(pooled.cpu().numpy())
        return np.vstack(embeddings), perf_counter() - started
