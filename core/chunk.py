"""Retrieval units.

The corpus answer pages are short synthetic entries whose key fact fits inside
the embedding context window, so splitting them into windows only dilutes the
page signal and lets long distractors win on a stray window. Whole-page units
scored higher on the public queries, so the production unit is one chunk per page.
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Dict, Iterable, List

from utils import entry_text


@dataclass
class Chunk:
    page_id: int
    chunk_id: int
    text: str


def chunk_entry(record: Dict[str, Any]) -> List[Chunk]:
    """Default retrieval unit: the whole page (title + content)."""
    return [Chunk(page_id=int(record["page_id"]), chunk_id=0, text=entry_text(record))]


def chunk_corpus(records: Iterable[Dict[str, Any]]) -> List[Chunk]:
    chunks: List[Chunk] = []
    for record in records:
        chunks.extend(chunk_entry(record))
    return chunks
