"""Tokenization for the lexical index.

Beyond plain lowercasing, two corpus-specific features make BM25 discriminate
this corpus's facts: shared *decade* tokens (so a "1820s" query matches an
exact-year page) and stemmed word *bigrams* (so phrases like "point guard" match
as units). Alphabetic words are Porter-stemmed so a query word matches its
morphological variants.
"""
from __future__ import annotations

import re
from typing import Callable, List

from .stemmer import PorterStemmer

# Keep alphabetic words and numbers; preserve digit-group separators so a token
# like "1,456,779" stays intact and matches between query and document.
_TOKEN_RE = re.compile(r"[a-z]+|\d[\d,\.]*\d|\d")
# A 4-digit year (1000-2099); emits the shared decade token (e.g. "1826" -> "182x").
_YEAR_RE = re.compile(r"(1\d{3}|20\d{2})")

# Default tokenizer features (module-level so dev sweeps can read/toggle them).
STEM = True
ADD_BIGRAMS = True

Tokenize = Callable[[str], List[str]]


class Tokenizer:
    """Lowercasing tokenizer with stemming, decade tokens, and word bigrams."""

    def __init__(
        self,
        *,
        stemmer: PorterStemmer | None = None,
        stem: bool = STEM,
        add_bigrams: bool = ADD_BIGRAMS,
    ) -> None:
        self._stemmer = stemmer or PorterStemmer()
        self._stem = stem
        self._add_bigrams = add_bigrams

    def __call__(self, text: str) -> List[str]:
        return self.tokenize(text)

    def tokenize(self, text: str) -> List[str]:
        raw = _TOKEN_RE.findall(text.lower())
        tokens: List[str] = []
        words: List[str] = []  # stemmed alphabetic words, for bigrams
        for token in raw:
            if token.isalpha():
                stemmed = self._stemmer.stem(token) if self._stem else token
                tokens.append(stemmed)
                words.append(stemmed)
            else:
                tokens.append(token)
        extra = [f"{t[:3]}x" for t in raw if _YEAR_RE.fullmatch(t)]
        if self._add_bigrams:
            extra += [f"{a}_{b}" for a, b in zip(words, words[1:])]
        return tokens + extra


_DEFAULT_TOKENIZER = Tokenizer()


def tokenize(text: str) -> List[str]:
    """Module-level facade over the default tokenizer (back-compat)."""
    return _DEFAULT_TOKENIZER.tokenize(text)
