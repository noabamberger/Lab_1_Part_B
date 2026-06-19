"""Dev-only: rebuild BM25 (now stemmed) + page_texts, reusing cached embeddings.

Page embeddings don't depend on the BM25 tokenizer, so we keep the committed
page_vectors.npy and only regenerate the lexical index, the reranker page texts,
and the config. Verifies page order against the existing page_ids.
"""
from __future__ import annotations

import json
import sys
import time
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from core.index import (  # noqa: E402
    BM25_NAME, CONFIG_NAME, DEFAULT_ALPHA, DEFAULT_RERANK, PAGE_IDS_NAME,
    PAGE_TEXTS_NAME, RERANK_MAX_CHARS,
)
from core.lexical import BM25Index  # noqa: E402
from utils import ARTIFACTS_DIR, entry_text, iter_entries  # noqa: E402


def main() -> None:
    out = ARTIFACTS_DIR
    page_ids = np.load(out / PAGE_IDS_NAME)

    records = list(iter_entries())
    ids_check = np.asarray([int(r["page_id"]) for r in records], dtype=np.int64)
    assert np.array_equal(ids_check, page_ids), "corpus/page order mismatch"
    page_texts = [entry_text(r) for r in records]

    t0 = time.time()
    bm25 = BM25Index.build(page_texts)
    print(f"built stemmed BM25 in {time.time()-t0:.1f}s, vocab={len(bm25.vocab)}")

    rerank_texts = np.asarray(
        [t[:RERANK_MAX_CHARS] for t in page_texts], dtype=object
    )

    bm25.save(out / BM25_NAME)
    np.save(out / PAGE_TEXTS_NAME, rerank_texts)
    (out / CONFIG_NAME).write_text(
        json.dumps(
            {
                "alpha": DEFAULT_ALPHA,
                "model": "sentence-transformers/all-MiniLM-L6-v2",
                "num_pages": int(page_ids.shape[0]),
                "reranker": DEFAULT_RERANK,
            },
            indent=2,
        ),
        encoding="utf-8",
    )
    print(f"rewrote {BM25_NAME}, {PAGE_TEXTS_NAME}, {CONFIG_NAME}")


if __name__ == "__main__":
    main()
