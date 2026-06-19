"""Load persisted artifacts into a :class:`LoadedIndex` (query-time, light)."""
from __future__ import annotations

import json
from pathlib import Path
from typing import Optional

import numpy as np

from utils import ARTIFACTS_DIR, EMBEDDING_MODEL_NAME

from ..lexical import BM25Index
from .config import (
    BM25_NAME,
    CONFIG_NAME,
    DEFAULT_ALPHA,
    DEFAULT_RERANK,
    PAGE_IDS_NAME,
    PAGE_TEXTS_NAME,
    PAGE_VECTORS_NAME,
)
from .loaded_index import LoadedIndex


class IndexLoader:
    """Reads the artifact files written by :class:`IndexBuilder`."""

    def __init__(self, artifacts_dir: Optional[Path] = None) -> None:
        self._root = artifacts_dir or ARTIFACTS_DIR

    def load(self) -> LoadedIndex:
        root = self._root
        page_vectors = np.ascontiguousarray(
            np.load(root / PAGE_VECTORS_NAME), dtype=np.float32
        )
        page_ids = np.load(root / PAGE_IDS_NAME)
        page_texts = np.load(root / PAGE_TEXTS_NAME, allow_pickle=True)
        bm25 = BM25Index.load(root / BM25_NAME)
        cfg = json.loads((root / CONFIG_NAME).read_text(encoding="utf-8"))
        return LoadedIndex(
            page_vectors=page_vectors,
            page_ids=page_ids,
            bm25=bm25,
            alpha=float(cfg.get("alpha", DEFAULT_ALPHA)),
            page_texts=page_texts,
            rerank={**DEFAULT_RERANK, **cfg.get("reranker", {})},
            embedding_model=cfg.get("model", EMBEDDING_MODEL_NAME),
        )


def load_index(artifacts_dir: Optional[Path] = None) -> LoadedIndex:
    """Facade: load all artifacts needed by the retrieval pipeline."""
    return IndexLoader(artifacts_dir).load()
