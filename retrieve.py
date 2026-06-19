"""Query-time retrieval (timed portion includes query embedding + reranking).

Per query batch:
  1. Embed queries with MiniLM (L2-normalized).
  2. Dense: FAISS inner-product search over page vectors (exact, full ranking).
  3. Lexical: BM25 page-level scores (stemmed; decade + bigram features).
  4. Fuse: per-query min-max normalize each signal, combine with weight alpha.
  5. Rerank: a cross-encoder rescores the top fused candidates per query; blend
     its score with the hybrid score and return the top-k page_ids.
"""
from __future__ import annotations

from pathlib import Path
from typing import List, Optional

import faiss
import numpy as np

from embed import embed_queries
from index import LoadedIndex, load_index
from reranker import score_pairs
from utils import K_EVAL

# Cache the loaded index and FAISS structure across calls within a process.
_INDEX: Optional[LoadedIndex] = None
_FAISS: Optional["faiss.Index"] = None
_ARTIFACTS_KEY: Optional[Path] = None


def _ensure_loaded(artifacts_dir: Optional[Path]) -> None:
    global _INDEX, _FAISS, _ARTIFACTS_KEY
    if _INDEX is not None and _ARTIFACTS_KEY == artifacts_dir:
        return
    idx = load_index(artifacts_dir)
    faiss_index = faiss.IndexFlatIP(idx.page_vectors.shape[1])
    faiss_index.add(idx.page_vectors)
    _INDEX, _FAISS, _ARTIFACTS_KEY = idx, faiss_index, artifacts_dir


def _minmax_rows(m: np.ndarray) -> np.ndarray:
    lo = m.min(axis=1, keepdims=True)
    hi = m.max(axis=1, keepdims=True)
    rng = np.where(hi - lo > 1e-12, hi - lo, 1.0)
    return (m - lo) / rng


def _minmax_1d(v: np.ndarray) -> np.ndarray:
    lo = float(v.min())
    rng = float(v.max()) - lo
    return (v - lo) / (rng if rng > 1e-12 else 1.0)


def _dense_scores(query_vectors: np.ndarray, n_pages: int) -> np.ndarray:
    """Full per-page inner-product scores, aligned to page_ids column order."""
    assert _FAISS is not None
    sims, idxs = _FAISS.search(query_vectors, n_pages)  # exact, all pages
    scores = np.empty((query_vectors.shape[0], n_pages), dtype=np.float32)
    rows = np.arange(query_vectors.shape[0])[:, None]
    scores[rows, idxs] = sims  # scatter back to page (column) order
    return scores


def search_batch(
    queries: List[str],
    *,
    top_k: int = K_EVAL,
    artifacts_dir: Optional[Path] = None,
) -> List[List[int]]:
    """Return ranked page_id lists (best first) for each query."""
    if not queries:
        return []
    _ensure_loaded(artifacts_dir)
    assert _INDEX is not None

    query_vectors = embed_queries(queries)
    if query_vectors.size == 0:
        return [[] for _ in queries]

    n_pages = _INDEX.page_ids.shape[0]
    dense = _dense_scores(query_vectors, n_pages)
    lex = _INDEX.bm25.score_batch(queries)

    alpha = _INDEX.alpha
    fused = alpha * _minmax_rows(dense) + (1.0 - alpha) * _minmax_rows(lex)

    page_ids = _INDEX.page_ids
    k = int(min(top_k, n_pages))

    rr = _INDEX.rerank
    if not rr.get("enabled", True):
        return _hybrid_topk(fused, page_ids, k)

    # --- Reranking stage ------------------------------------------------------
    pool = int(min(rr.get("pool", 100), n_pages))
    # Top-`pool` fused candidates per query (unordered slice, then sort).
    pool_idx = np.argpartition(-fused, pool - 1, axis=1)[:, :pool]
    texts = _INDEX.page_texts

    # One batched cross-encoder call over all (query, candidate) pairs.
    pairs: List[tuple] = []
    for qi, q in enumerate(queries):
        for col in pool_idx[qi]:
            pairs.append((q, str(texts[col])))
    ce = score_pairs(
        rr["model"], pairs,
        max_length=int(rr.get("max_length", 512)),
    ).reshape(len(queries), pool)

    w = float(rr.get("weight", 0.5))
    ranked: List[List[int]] = []
    for qi in range(len(queries)):
        cols = pool_idx[qi]
        blended = w * _minmax_1d(ce[qi]) + (1.0 - w) * _minmax_1d(fused[qi, cols])
        local = np.argsort(-blended)
        ordered = [int(page_ids[cols[i]]) for i in local[:k]]
        ranked.append(ordered)
    return ranked


def _hybrid_topk(fused: np.ndarray, page_ids: np.ndarray, k: int) -> List[List[int]]:
    """Pure hybrid fallback when the reranker is disabled."""
    ranked: List[List[int]] = []
    for row in fused:
        top = np.argpartition(-row, k - 1)[:k]
        top = top[np.argsort(-row[top])]
        ranked.append([int(page_ids[i]) for i in top])
    return ranked
