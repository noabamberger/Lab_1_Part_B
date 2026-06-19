"""Dev-only: assemble artifacts/ from the cached page embeddings.

The cached page vectors are produced by the same embed_texts() call and the same
iter_entries() ordering as index.build_index(), so the assembled artifacts are
identical to a full offline build, without re-embedding 27k pages.
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from index import (  # noqa: E402
    BM25_NAME, CONFIG_NAME, DEFAULT_ALPHA, PAGE_IDS_NAME, PAGE_VECTORS_NAME,
)
from lexical import BM25Index  # noqa: E402
from utils import ensure_artifacts_dir, entry_text, iter_entries  # noqa: E402

CACHE = ROOT / "dev" / "cache"


def main() -> None:
    out = ensure_artifacts_dir()
    page_vectors = np.load(CACHE / "page_vectors.npy")
    page_ids = np.load(CACHE / "page_ids.npy")

    # Rebuild page texts in the same order to guarantee BM25 column alignment.
    records = list(iter_entries())
    ids_check = np.asarray([int(r["page_id"]) for r in records], dtype=np.int64)
    assert np.array_equal(ids_check, page_ids), "cache/page order mismatch"
    page_texts = [entry_text(r) for r in records]

    bm25 = BM25Index.build(page_texts)

    np.save(out / PAGE_VECTORS_NAME, page_vectors)
    np.save(out / PAGE_IDS_NAME, page_ids)
    bm25.save(out / BM25_NAME)
    (out / CONFIG_NAME).write_text(
        json.dumps(
            {
                "alpha": DEFAULT_ALPHA,
                "model": "sentence-transformers/all-MiniLM-L6-v2",
                "num_pages": int(page_ids.shape[0]),
            },
            indent=2,
        ),
        encoding="utf-8",
    )
    print(f"assembled artifacts/ for {len(page_ids)} pages, vocab {len(bm25.vocab)}")


if __name__ == "__main__":
    main()
