"""Sentence embedding (sentence-transformers/all-MiniLM-L6-v2)."""
from __future__ import annotations

from typing import Sequence

import numpy as np
from sentence_transformers import SentenceTransformer

from utils import EMBEDDING_MODEL_NAME

_EMBED_DIM = 384


class EmbeddingModel:
    """Lazily-loaded MiniLM encoder producing L2-normalized vectors."""

    def __init__(self, model_name: str = EMBEDDING_MODEL_NAME) -> None:
        self._model_name = model_name
        self._model: SentenceTransformer | None = None

    @property
    def model(self) -> SentenceTransformer:
        if self._model is None:
            self._model = SentenceTransformer(self._model_name)
        return self._model

    def encode(self, texts: Sequence[str], *, batch_size: int = 64) -> np.ndarray:
        """Return L2-normalized embeddings, shape ``(n, dim)``."""
        if not texts:
            return np.zeros((0, _EMBED_DIM), dtype=np.float32)
        vectors = self.model.encode(
            list(texts),
            batch_size=batch_size,
            show_progress_bar=False,
            convert_to_numpy=True,
            normalize_embeddings=True,
        )
        return np.asarray(vectors, dtype=np.float32)


# Shared instance + function facades (back-compat for offline/dev callers).
_DEFAULT_MODEL = EmbeddingModel()


def get_model() -> SentenceTransformer:
    return _DEFAULT_MODEL.model


def embed_texts(texts: Sequence[str], *, batch_size: int = 64) -> np.ndarray:
    return _DEFAULT_MODEL.encode(texts, batch_size=batch_size)


def embed_queries(queries: Sequence[str], *, batch_size: int = 64) -> np.ndarray:
    return _DEFAULT_MODEL.encode(queries, batch_size=batch_size)
