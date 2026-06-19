"""Process-level retrieval entry point used by ``main.run``.

Caches one assembled pipeline per artifacts directory so repeated calls within a
process reuse the loaded index and FAISS structure.
"""
from __future__ import annotations

from pathlib import Path
from typing import List, Optional

from utils import K_EVAL

from ..index import load_index
from .pipeline import RetrievalPipeline

_PIPELINE: Optional[RetrievalPipeline] = None
_ARTIFACTS_KEY: Optional[Path] = None


def _get_pipeline(artifacts_dir: Optional[Path]) -> RetrievalPipeline:
    global _PIPELINE, _ARTIFACTS_KEY
    if _PIPELINE is None or _ARTIFACTS_KEY != artifacts_dir:
        _PIPELINE = RetrievalPipeline.from_index(load_index(artifacts_dir))
        _ARTIFACTS_KEY = artifacts_dir
    return _PIPELINE


def search_batch(
    queries: List[str],
    *,
    top_k: int = K_EVAL,
    artifacts_dir: Optional[Path] = None,
) -> List[List[int]]:
    if not queries:
        return []
    return _get_pipeline(artifacts_dir).search(queries, top_k=top_k)
