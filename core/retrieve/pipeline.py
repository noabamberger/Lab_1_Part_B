"""Three-stage retrieval pipeline: dense + lexical fusion, then rerank.

Per query batch:
  1. Dense   — FAISS inner-product search over MiniLM page vectors (exact).
  2. Lexical — BM25 page scores (stemmed; decade + bigram features).
  3. Fuse    — per-query min-max normalize each signal, combine with weight alpha.
  4. Rerank  — a cross-encoder rescores the top fused candidates per query; its
               score is blended with the hybrid score before taking the top-k.
"""
from __future__ import annotations

from typing import List, Optional, Tuple

import numpy as np

from utils import K_EVAL

from ..embed import EmbeddingModel
from ..index import LoadedIndex
from ..interfaces import PageScorer, Reranker
from ..reranker import CrossEncoderReranker
from .dense import DenseRetriever
from .fusion import HybridFuser
from .normalizer import Normalizer


class RetrievalPipeline:
    """Dense + lexical fusion with an optional cross-encoder rerank stage."""

    def __init__(
        self,
        *,
        dense: PageScorer,
        lexical: PageScorer,
        fuser: HybridFuser,
        page_ids: np.ndarray,
        reranker: Optional[Reranker] = None,
        page_texts: Optional[np.ndarray] = None,
        pool: int = 100,
        rerank_weight: float = 0.5,
    ) -> None:
        self._dense = dense
        self._lexical = lexical
        self._fuser = fuser
        self._page_ids = page_ids
        self._reranker = reranker
        self._page_texts = page_texts
        self._pool = pool
        self._rerank_weight = rerank_weight

    @classmethod
    def from_index(cls, idx: LoadedIndex) -> "RetrievalPipeline":
        """Assemble the pipeline from loaded artifacts (composition root)."""
        dense = DenseRetriever(idx.page_vectors, EmbeddingModel(idx.embedding_model))
        rr = idx.rerank
        reranker: Optional[Reranker] = None
        if rr.get("enabled", True):
            reranker = CrossEncoderReranker(
                rr["model"], max_length=int(rr.get("max_length", 512))
            )
        return cls(
            dense=dense,
            lexical=idx.bm25,
            fuser=HybridFuser(idx.alpha),
            page_ids=idx.page_ids,
            reranker=reranker,
            page_texts=idx.page_texts,
            pool=int(rr.get("pool", 100)),
            rerank_weight=float(rr.get("weight", 0.5)),
        )

    def search(self, queries: List[str], *, top_k: int = K_EVAL) -> List[List[int]]:
        """Return ranked page_id lists (best first) for each query."""
        if not queries:
            return []

        fused = self._fuser.fuse(self._dense.score(queries), self._lexical.score(queries))
        n_pages = self._page_ids.shape[0]
        k = int(min(top_k, n_pages))

        if self._reranker is None:
            return self._hybrid_topk(fused, k)
        return self._rerank_topk(queries, fused, k, n_pages)

    def _hybrid_topk(self, fused: np.ndarray, k: int) -> List[List[int]]:
        ranked: List[List[int]] = []
        for row in fused:
            top = np.argpartition(-row, k - 1)[:k]
            top = top[np.argsort(-row[top])]
            ranked.append([int(self._page_ids[i]) for i in top])
        return ranked

    def _rerank_topk(
        self, queries: List[str], fused: np.ndarray, k: int, n_pages: int
    ) -> List[List[int]]:
        pool = int(min(self._pool, n_pages))
        # Top-`pool` fused candidates per query (unordered slice, then re-sort).
        pool_idx = np.argpartition(-fused, pool - 1, axis=1)[:, :pool]

        pairs: List[Tuple[str, str]] = [
            (query, str(self._page_texts[col]))
            for qi, query in enumerate(queries)
            for col in pool_idx[qi]
        ]
        ce = self._reranker.score_pairs(pairs).reshape(len(queries), pool)

        weight = self._rerank_weight
        ranked: List[List[int]] = []
        for qi in range(len(queries)):
            cols = pool_idx[qi]
            blended = (
                weight * Normalizer.vector(ce[qi])
                + (1.0 - weight) * Normalizer.vector(fused[qi, cols])
            )
            order = np.argsort(-blended)
            ranked.append([int(self._page_ids[cols[i]]) for i in order[:k]])
        return ranked
