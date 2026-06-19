"""Cross-encoder reranker (query-time second stage).

The hybrid dense+BM25 stage has high recall but imperfect ordering (recall@100
~0.93 while NDCG@10 ~0.45 on the public queries — lots of ordering headroom). A
cross-encoder reads each (query, page) pair jointly and rescores the top fused
candidates. We use `cross-encoder/ms-marco-MiniLM-L-6-v2`: small (~80MB, so it
downloads and loads well within the query-time budget) and, on this corpus, a
better reranker than larger general-purpose models (BGE/L-12 overfit to natural
QA passages and scored worse here — see dev/rerank_blend.py).

The reranker is blended with, not substituted for, the hybrid score: it sharply
improves clean single-answer factual queries but can mis-rank the templated
multi-entity queries, so keeping a hybrid prior is more robust than pure rerank.
"""
from __future__ import annotations

from typing import List, Optional

import numpy as np

_reranker = None
_reranker_name: Optional[str] = None


def get_reranker(model_name: str, max_length: int = 512):
    """Lazily load and cache one CrossEncoder per process."""
    global _reranker, _reranker_name
    if _reranker is None or _reranker_name != model_name:
        from sentence_transformers import CrossEncoder

        _reranker = CrossEncoder(model_name, max_length=max_length)
        _reranker_name = model_name
    return _reranker


def score_pairs(
    model_name: str,
    pairs: List[tuple],
    *,
    max_length: int = 512,
    batch_size: int = 64,
) -> np.ndarray:
    """Return a relevance score per (query, passage) pair."""
    if not pairs:
        return np.zeros(0, dtype=np.float32)
    ce = get_reranker(model_name, max_length=max_length)
    scores = ce.predict(pairs, batch_size=batch_size, show_progress_bar=False)
    return np.asarray(scores, dtype=np.float32)
