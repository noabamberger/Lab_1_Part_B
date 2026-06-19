"""In-memory artifacts consumed by the retrieval pipeline."""
from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from utils import EMBEDDING_MODEL_NAME

from ..lexical import BM25Index


@dataclass
class LoadedIndex:
    """Everything ``RetrievalPipeline.from_index`` needs to assemble itself."""

    page_vectors: np.ndarray
    page_ids: np.ndarray
    bm25: BM25Index
    alpha: float
    page_texts: np.ndarray
    rerank: dict
    embedding_model: str = EMBEDDING_MODEL_NAME
