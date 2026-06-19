"""Lexical retrieval package: tokenization, Porter stemming, and BM25.

Re-exports the public names so callers (and dev sweeps that monkeypatch
``core.lexical.tokenize``) keep working after the split into submodules.
"""
from __future__ import annotations

from .bm25 import BM25_B, BM25_K1, MAX_DOC_TOKENS, BM25Index
from .stemmer import PorterStemmer, stem
from .tokenizer import (
    ADD_BIGRAMS,
    STEM,
    Tokenize,
    Tokenizer,
    _TOKEN_RE,
    _YEAR_RE,
    tokenize,
)

__all__ = [
    "BM25Index",
    "BM25_K1",
    "BM25_B",
    "MAX_DOC_TOKENS",
    "Tokenizer",
    "Tokenize",
    "tokenize",
    "STEM",
    "ADD_BIGRAMS",
    "PorterStemmer",
    "stem",
]
