"""Artifact filenames and default retrieval configuration.

These constants define the on-disk layout written by the builder and read by the
loader, plus the tuned fusion/reranker defaults (overridable in
``retrieval_config.json``).
"""
from __future__ import annotations

PAGE_VECTORS_NAME = "page_vectors.npy"
PAGE_IDS_NAME = "page_ids.npy"
PAGE_TEXTS_NAME = "page_texts.npy"
BM25_NAME = "bm25_index.npz"
CONFIG_NAME = "retrieval_config.json"

# final = ALPHA * dense_norm + (1 - ALPHA) * bm25_norm (each per-query normalized).
# Tuned with Optuna over the public queries; picks the robust region center.
DEFAULT_ALPHA = 0.7

# Per-page reranker text is truncated to bound artifact size and tokenization
# cost (answer pages are short and fit comfortably under this).
RERANK_MAX_CHARS = 2000

# Reranker stage settings (overridable in retrieval_config.json).
DEFAULT_RERANK = {
    "enabled": True,
    "model": "cross-encoder/ms-marco-MiniLM-L-6-v2",
    "pool": 120,        # fused candidates rescored per query (Optuna-tuned)
    "weight": 0.6,      # blend: weight*ce_norm + (1-weight)*hybrid_norm
    "max_length": 512,  # reranker token cap
}
