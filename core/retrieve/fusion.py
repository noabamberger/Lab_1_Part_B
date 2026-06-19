"""Linear fusion of the dense and lexical signals."""
from __future__ import annotations

import numpy as np

from .normalizer import Normalizer


class HybridFuser:
    """Fuse two normalized signals: ``alpha*dense + (1 - alpha)*lexical``."""

    def __init__(self, alpha: float) -> None:
        self.alpha = alpha

    def fuse(self, dense: np.ndarray, lexical: np.ndarray) -> np.ndarray:
        return (
            self.alpha * Normalizer.rows(dense)
            + (1.0 - self.alpha) * Normalizer.rows(lexical)
        )
