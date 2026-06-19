"""Offline index build + load.

Two page-level signals are persisted for query time: dense MiniLM page vectors
and a lexical BM25 index (plus truncated page texts for the reranker). See
:mod:`core.index.config` for the on-disk layout.

Public names are re-exported lazily so importing the light pieces (config
constants, ``LoadedIndex``, ``load_index``) does not pull in the embedding stack
that the builder needs.
"""
from __future__ import annotations

import importlib
from typing import Any

_EXPORTS = {
    # config
    "PAGE_VECTORS_NAME": "config",
    "PAGE_IDS_NAME": "config",
    "PAGE_TEXTS_NAME": "config",
    "BM25_NAME": "config",
    "CONFIG_NAME": "config",
    "DEFAULT_ALPHA": "config",
    "RERANK_MAX_CHARS": "config",
    "DEFAULT_RERANK": "config",
    # data class
    "LoadedIndex": "loaded_index",
    # build (heavy: needs the embedding model)
    "IndexBuilder": "builder",
    "build_index": "builder",
    # load (light)
    "IndexLoader": "loader",
    "load_index": "loader",
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
