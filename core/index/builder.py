"""Offline index build (not timed at grading).

Embeds every page with MiniLM, builds the BM25 index, and persists all artifacts
described in :mod:`core.index.config`.
"""
from __future__ import annotations

import json
from dataclasses import dataclass, field
from pathlib import Path
from typing import List, Optional

import numpy as np

from utils import (
    EMBEDDING_MODEL_NAME,
    ensure_artifacts_dir,
    entry_text,
    iter_entries,
)

from ..embed import EmbeddingModel
from ..lexical import BM25Index
from .config import (
    BM25_NAME,
    CONFIG_NAME,
    DEFAULT_ALPHA,
    DEFAULT_RERANK,
    PAGE_IDS_NAME,
    PAGE_TEXTS_NAME,
    PAGE_VECTORS_NAME,
    RERANK_MAX_CHARS,
)


@dataclass
class IndexBuilder:
    """Embeds pages, builds the BM25 index, and persists all artifacts."""

    alpha: float = DEFAULT_ALPHA
    embedding_model: str = EMBEDDING_MODEL_NAME
    rerank_max_chars: int = RERANK_MAX_CHARS
    rerank: dict = field(default_factory=lambda: dict(DEFAULT_RERANK))

    def build(
        self,
        *,
        entries_dir: Optional[Path] = None,
        artifacts_dir: Optional[Path] = None,
    ) -> None:
        out_dir = artifacts_dir or ensure_artifacts_dir()
        records = list(iter_entries(entries_dir))

        page_ids = np.asarray([int(r["page_id"]) for r in records], dtype=np.int64)
        page_texts: List[str] = [entry_text(r) for r in records]

        page_vectors = EmbeddingModel(self.embedding_model).encode(page_texts, batch_size=128)
        bm25 = BM25Index.build(page_texts)
        rerank_texts = np.asarray(
            [t[: self.rerank_max_chars] for t in page_texts], dtype=object
        )

        np.save(out_dir / PAGE_VECTORS_NAME, page_vectors)
        np.save(out_dir / PAGE_IDS_NAME, page_ids)
        np.save(out_dir / PAGE_TEXTS_NAME, rerank_texts)
        bm25.save(out_dir / BM25_NAME)
        (out_dir / CONFIG_NAME).write_text(
            json.dumps(
                {
                    "alpha": self.alpha,
                    "model": self.embedding_model,
                    "num_pages": int(page_ids.shape[0]),
                    "reranker": self.rerank,
                },
                indent=2,
            ),
            encoding="utf-8",
        )


def build_index(
    *,
    entries_dir: Optional[Path] = None,
    artifacts_dir: Optional[Path] = None,
    alpha: float = DEFAULT_ALPHA,
) -> None:
    """Facade: build and persist all artifacts."""
    IndexBuilder(alpha=alpha).build(entries_dir=entries_dir, artifacts_dir=artifacts_dir)
