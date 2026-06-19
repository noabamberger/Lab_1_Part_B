"""Preprocessing and chunking.

Empirical finding (see dev/ sweeps): the corpus answer pages are short synthetic
entries whose key fact fits inside the embedding model's context window, so
splitting them into windows only dilutes the page signal and lets long
distractor articles win on a single stray window. Whole-page units scored higher
than windowed max-pooling on the public queries, so the production pipeline uses
one chunk per page. The windowed chunker is retained as ``window_chunks`` for
reproducing that comparison.
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Dict, List

from utils import entry_text


@dataclass
class Chunk:
    page_id: int
    chunk_id: int
    text: str


def chunk_entry(record: Dict[str, Any]) -> List[Chunk]:
    """Default retrieval unit: the whole page (title + content)."""
    page_id = int(record["page_id"])
    return [Chunk(page_id=page_id, chunk_id=0, text=entry_text(record))]


def chunk_corpus(records: List[Dict[str, Any]]) -> List[Chunk]:
    chunks: List[Chunk] = []
    for record in records:
        chunks.extend(chunk_entry(record))
    return chunks
