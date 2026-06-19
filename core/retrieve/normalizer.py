"""Per-query score normalization."""
from __future__ import annotations

import numpy as np


class Normalizer:
    """Per-query min-max normalization, putting every signal on a [0, 1] scale."""

    @staticmethod
    def rows(matrix: np.ndarray) -> np.ndarray:
        lo = matrix.min(axis=1, keepdims=True)
        hi = matrix.max(axis=1, keepdims=True)
        rng = np.where(hi - lo > 1e-12, hi - lo, 1.0)
        return (matrix - lo) / rng

    @staticmethod
    def vector(values: np.ndarray) -> np.ndarray:
        lo = float(values.min())
        rng = float(values.max()) - lo
        return (values - lo) / (rng if rng > 1e-12 else 1.0)


# Function facades (back-compat for dev sweeps importing these names).
_minmax_rows = Normalizer.rows
_minmax_1d = Normalizer.vector
