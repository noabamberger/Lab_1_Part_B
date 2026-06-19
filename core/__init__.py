"""Hybrid retrieval pipeline (dense + lexical BM25 -> cross-encoder rerank).

Public API is re-exported lazily so importing a light component (e.g. the BM25
index or the stemmer) does not pull in the heavy embedding/reranking stack.
"""
from __future__ import annotations

import importlib
from typing import Any

_EXPORTS = {
    # interfaces
    "PageScorer": "interfaces",
    "Reranker": "interfaces",
    # lexical
    "Tokenizer": "lexical",
    "tokenize": "lexical",
    "BM25Index": "lexical",
    # stemmer (lives in the lexical package)
    "PorterStemmer": "lexical",
    "stem": "lexical",
    # embedding / reranking
    "EmbeddingModel": "embed",
    "CrossEncoderReranker": "reranker",
    # retrieval
    "Normalizer": "retrieve",
    "DenseRetriever": "retrieve",
    "HybridFuser": "retrieve",
    "RetrievalPipeline": "retrieve",
    "search_batch": "retrieve",
    # index build/load
    "IndexBuilder": "index",
    "IndexLoader": "index",
    "LoadedIndex": "index",
    "build_index": "index",
    "load_index": "index",
    # chunking
    "Chunk": "chunk",
    "chunk_entry": "chunk",
    "chunk_corpus": "chunk",
}

__all__ = sorted(_EXPORTS)


def __getattr__(name: str) -> Any:
    module = _EXPORTS.get(name)
    if module is None:
        raise AttributeError(f"module {__name__!r} has no attribute {name!r}")
    return getattr(importlib.import_module(f".{module}", __name__), name)


def __dir__() -> list[str]:
    return __all__
