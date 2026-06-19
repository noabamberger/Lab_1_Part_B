"""Query-time retrieval package (dense + lexical fusion -> cross-encoder rerank).

Public names are re-exported lazily so the light pieces (``Normalizer`` and the
``_minmax_*`` facades) can be imported without pulling in the embedding/reranking
stack that the dense retriever and pipeline require.
"""
from __future__ import annotations

import importlib
from typing import Any

_EXPORTS = {
    # normalization (light)
    "Normalizer": "normalizer",
    "_minmax_rows": "normalizer",
    "_minmax_1d": "normalizer",
    # signals / fusion / pipeline (heavy)
    "DenseRetriever": "dense",
    "HybridFuser": "fusion",
    "RetrievalPipeline": "pipeline",
    # process entry point
    "search_batch": "service",
}

__all__ = sorted(_EXPORTS)


def __getattr__(name: str) -> Any:
    module = _EXPORTS.get(name)
    if module is None:
        raise AttributeError(f"module {__name__!r} has no attribute {name!r}")
    value = getattr(importlib.import_module(f".{module}", __name__), name)
    globals()[name] = value  # cache for subsequent lookups
    return value


def __dir__() -> list[str]:
    return __all__
