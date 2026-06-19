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

import re
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


# --------------------------------------------------------------------------- #
# Alternative (explored, not used): overlapping word-window chunking.
# --------------------------------------------------------------------------- #
_WS_RE = re.compile(r"\s+")


def window_chunks(
    record: Dict[str, Any],
    *,
    window: int = 80,
    stride: int = 50,
    max_words: int = 512,
) -> List[Chunk]:
    """Overlapping, title-prefixed word windows (dev comparison only)."""
    page_id = int(record["page_id"])
    title = str(record.get("title", "")).strip()
    content = str(record.get("content", "")).strip()
    words = _WS_RE.sub(" ", content).strip().split(" ")[:max_words]
    prefix = f"{title}. " if title else ""
    if not words or words == [""]:
        return [Chunk(page_id=page_id, chunk_id=0, text=title or entry_text(record))]
    chunks: List[Chunk] = []
    cid = 0
    for start in range(0, len(words), stride):
        win = words[start : start + window]
        if not win:
            break
        chunks.append(Chunk(page_id=page_id, chunk_id=cid, text=prefix + " ".join(win)))
        cid += 1
        if start + window >= len(words):
            break
    return chunks
