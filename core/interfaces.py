"""Abstractions shared by the retrieval pipeline.

These small interfaces let the pipeline depend on *roles* (a page scorer, a
reranker) rather than concrete classes, so a new retrieval signal or reranker
can be swapped in without touching the orchestration (Open/Closed + Dependency
Inversion).
"""
from __future__ import annotations

from abc import ABC, abstractmethod
from typing import List, Sequence, Tuple

import numpy as np


class PageScorer(ABC):
    """A first-stage retrieval signal that scores every page against a query.

    Implementations return a dense ``(n_queries, n_pages)`` matrix whose columns
    are aligned to the corpus page-id order, so different signals can be fused
    column-wise.
    """

    @abstractmethod
    def score(self, queries: Sequence[str]) -> np.ndarray:
        """Return the per-page score matrix for ``queries``."""


class Reranker(ABC):
    """A second-stage scorer over a short candidate list per query."""

    @abstractmethod
    def score_pairs(self, pairs: List[Tuple[str, str]]) -> np.ndarray:
        """Return one relevance score per ``(query, passage)`` pair."""
