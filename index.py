"""Offline index build and load (build is not timed at grading).

The pipeline fuses two page-level signals:
  * Dense  — whole-page MiniLM embeddings, searched with a FAISS inner-product
             index (vectors are L2-normalized, so inner product = cosine).
  * Lexical — a custom BM25 index with decade and word-bigram features that
             discriminate the corpus's specific facts (years, named phrases).

A cross-encoder reranker then reorders the top fused candidates at query time
(see retrieve.py / reranker.py); the page texts it needs are persisted too.

Artifacts written to artifacts/:
  page_vectors.npy       float32 (n_pages, dim), L2-normalized page embeddings
  page_ids.npy           int64   (n_pages,) page_id for each row / BM25 column
  page_texts.npy         object  (n_pages,) truncated page text for the reranker
  bm25_index.npz         BM25 inverted index (term_indptr, doc_indices, weights)
  bm25_index.meta.json   BM25 vocabulary + params
  retrieval_config.json  fusion weight (alpha), model name, reranker settings
"""
from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path
from typing import List, Optional

import numpy as np

from embed import embed_texts
from lexical import BM25Index
from utils import ARTIFACTS_DIR, ensure_artifacts_dir, entry_text, iter_entries

PAGE_VECTORS_NAME = "page_vectors.npy"
PAGE_IDS_NAME = "page_ids.npy"
PAGE_TEXTS_NAME = "page_texts.npy"
BM25_NAME = "bm25_index.npz"
CONFIG_NAME = "retrieval_config.json"

# Fusion: final = ALPHA * dense_norm + (1 - ALPHA) * bm25_norm, each per-query
# min-max normalized. Tuned with Optuna over the public queries (dev/optuna_tune.py),
# picking the robust region center rather than the noisy argmax. Persisted in
# retrieval_config.json and overridable there.
DEFAULT_ALPHA = 0.7

# Per-page text fed to the reranker is truncated to bound artifact size and
# tokenization cost (the reranker also caps at max_length tokens). Answer pages
# are short and fit comfortably under this.
RERANK_MAX_CHARS = 2000

# Reranker stage (cross-encoder over the top fused candidates). Small, fast, and
# downloads quickly so it fits the query-time budget; settings chosen on the
# public queries (see dev/rerank_blend.py). Overridable in retrieval_config.json.
DEFAULT_RERANK = {
    "enabled": True,
    "model": "cross-encoder/ms-marco-MiniLM-L-6-v2",
    "pool": 120,        # fused candidates rescored per query (Optuna-tuned)
    "weight": 0.6,      # blend: weight*ce_norm + (1-weight)*hybrid_norm
    "max_length": 512,  # reranker token cap
}


@dataclass
class LoadedIndex:
    page_vectors: np.ndarray
    page_ids: np.ndarray
    bm25: BM25Index
    alpha: float
    page_texts: np.ndarray
    rerank: dict


def build_index(
    *,
    entries_dir: Optional[Path] = None,
    artifacts_dir: Optional[Path] = None,
    alpha: float = DEFAULT_ALPHA,
) -> None:
    """Embed pages, build the BM25 index, and persist all artifacts."""
    out_dir = artifacts_dir or ensure_artifacts_dir()
    records = list(iter_entries(entries_dir))

    page_ids = np.asarray([int(r["page_id"]) for r in records], dtype=np.int64)
    page_texts: List[str] = [entry_text(r) for r in records]

    page_vectors = embed_texts(page_texts, batch_size=128)
    bm25 = BM25Index.build(page_texts)

    rerank_texts = np.asarray(
        [t[:RERANK_MAX_CHARS] for t in page_texts], dtype=object
    )

    np.save(out_dir / PAGE_VECTORS_NAME, page_vectors)
    np.save(out_dir / PAGE_IDS_NAME, page_ids)
    np.save(out_dir / PAGE_TEXTS_NAME, rerank_texts)
    bm25.save(out_dir / BM25_NAME)
    (out_dir / CONFIG_NAME).write_text(
        json.dumps(
            {
                "alpha": alpha,
                "model": "sentence-transformers/all-MiniLM-L6-v2",
                "num_pages": int(page_ids.shape[0]),
                "reranker": DEFAULT_RERANK,
            },
            indent=2,
        ),
        encoding="utf-8",
    )


def load_index(artifacts_dir: Optional[Path] = None) -> LoadedIndex:
    """Load all artifacts needed by retrieve.search_batch."""
    root = artifacts_dir or ARTIFACTS_DIR
    page_vectors = np.ascontiguousarray(
        np.load(root / PAGE_VECTORS_NAME), dtype=np.float32
    )
    page_ids = np.load(root / PAGE_IDS_NAME)
    page_texts = np.load(root / PAGE_TEXTS_NAME, allow_pickle=True)
    bm25 = BM25Index.load(root / BM25_NAME)
    cfg = json.loads((root / CONFIG_NAME).read_text(encoding="utf-8"))
    rerank = {**DEFAULT_RERANK, **cfg.get("reranker", {})}
    return LoadedIndex(
        page_vectors=page_vectors,
        page_ids=page_ids,
        bm25=bm25,
        alpha=float(cfg.get("alpha", DEFAULT_ALPHA)),
        page_texts=page_texts,
        rerank=rerank,
    )
