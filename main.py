"""Section B entry point.

The autograder calls ``run(queries)`` once with all evaluation queries; query
embedding + retrieval + reranking must complete within the time limit (GPU
available at grading). Running this module as a script builds the offline index.
"""
from __future__ import annotations

from typing import List

from core.index import build_index
from core.retrieve import search_batch


def run(queries: List[str]) -> List[List[int]]:
    """Rank corpus pages for each query.

    Returns one ranked list of ``page_id`` per query, most relevant first; only
    the first 10 IDs per list are scored (mean NDCG@10).
    """
    return search_batch(queries)


def build_offline_index() -> None:
    """Run once locally to create artifacts/ (not timed at grading)."""
    build_index()


if __name__ == "__main__":
    build_offline_index()
    print("Index built under artifacts/. Run: python scripts/eval_public.py")
