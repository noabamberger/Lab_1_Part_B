"""Dense retrieval signal (MiniLM embeddings via FAISS)."""
from __future__ import annotations

from typing import Optional, Sequence

import faiss
import numpy as np

from ..embed import EmbeddingModel
from ..interfaces import PageScorer


class DenseRetriever(PageScorer):
    """Dense signal: MiniLM query embeddings vs. page vectors via FAISS."""

    def __init__(self, page_vectors: np.ndarray, embedder: EmbeddingModel) -> None:
        self._page_vectors = page_vectors
        self._embedder = embedder
        self._index: Optional["faiss.Index"] = None

    @property
    def index(self) -> "faiss.Index":
        if self._index is None:
            self._index = faiss.IndexFlatIP(self._page_vectors.shape[1])
            self._index.add(self._page_vectors)
        return self._index

    def score(self, queries: Sequence[str]) -> np.ndarray:
        """Full per-page inner-product scores, aligned to page (column) order."""
        query_vectors = self._embedder.encode(list(queries))
        n_pages = self._page_vectors.shape[0]
        sims, idxs = self.index.search(query_vectors, n_pages)  # exact, all pages
        scores = np.empty((query_vectors.shape[0], n_pages), dtype=np.float32)
        rows = np.arange(query_vectors.shape[0])[:, None]
        scores[rows, idxs] = sims  # scatter back to page order
        return scores
