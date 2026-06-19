"""Cross-encoder reranker (query-time second stage).

The hybrid dense+BM25 stage has high recall but imperfect ordering (recall@100
~0.93 vs NDCG@10 ~0.45 on the public queries). A cross-encoder reads each
(query, page) pair jointly and rescores the top fused candidates. We use
``cross-encoder/ms-marco-MiniLM-L-6-v2``: small (~80MB, loads within the
query-time budget) and, on this corpus, a stronger reranker than larger
general-purpose models that overfit to natural-QA passages.

The reranker is *blended* with, not substituted for, the hybrid score: it
sharpens clean single-answer queries but can mis-rank the templated multi-entity
queries, so keeping a hybrid prior is more robust than pure rerank.
"""
from __future__ import annotations

from typing import List, Tuple

import numpy as np

from .interfaces import Reranker


class CrossEncoderReranker(Reranker):
    """Lazily-loaded cross-encoder that scores (query, passage) pairs."""

    def __init__(self, model_name: str, *, max_length: int = 512, batch_size: int = 64) -> None:
        self.model_name = model_name
        self.max_length = max_length
        self.batch_size = batch_size
        self._encoder = None

    @property
    def encoder(self):
        if self._encoder is None:
            from sentence_transformers import CrossEncoder

            self._encoder = CrossEncoder(self.model_name, max_length=self.max_length)
        return self._encoder

    def score_pairs(self, pairs: List[Tuple[str, str]]) -> np.ndarray:
        """Return one relevance score per ``(query, passage)`` pair."""
        if not pairs:
            return np.zeros(0, dtype=np.float32)
        scores = self.encoder.predict(
            pairs, batch_size=self.batch_size, show_progress_bar=False
        )
        return np.asarray(scores, dtype=np.float32)


# Cache one reranker per model name + function facade (back-compat for dev).
_RERANKERS: dict[str, CrossEncoderReranker] = {}


def get_reranker(model_name: str, max_length: int = 512) -> CrossEncoderReranker:
    reranker = _RERANKERS.get(model_name)
    if reranker is None:
        reranker = CrossEncoderReranker(model_name, max_length=max_length)
        _RERANKERS[model_name] = reranker
    return reranker


def score_pairs(
    model_name: str,
    pairs: List[Tuple[str, str]],
    *,
    max_length: int = 512,
    batch_size: int = 64,
) -> np.ndarray:
    reranker = get_reranker(model_name, max_length=max_length)
    reranker.batch_size = batch_size
    return reranker.score_pairs(pairs)
