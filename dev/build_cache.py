"""Dev-only: embed the full corpus once and cache to dev/cache/ for fast sweeps.

Not part of the submission pipeline; used during development to iterate on
retrieval configurations without re-embedding 27k pages each time.
"""
from __future__ import annotations

import json
import sys
import time
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from core.embed import embed_texts  # noqa: E402
from utils import entry_text, iter_entries  # noqa: E402

CACHE = ROOT / "dev" / "cache"


def main() -> None:
    CACHE.mkdir(parents=True, exist_ok=True)
    t0 = time.perf_counter()
    records = list(iter_entries())
    page_ids = [int(r["page_id"]) for r in records]
    texts = [entry_text(r) for r in records]
    print(f"loaded {len(records)} pages in {time.perf_counter()-t0:.1f}s")

    t1 = time.perf_counter()
    vecs = embed_texts(texts, batch_size=128)
    print(f"embedded in {time.perf_counter()-t1:.1f}s -> {vecs.shape}")

    np.save(CACHE / "page_vectors.npy", vecs)
    np.save(CACHE / "page_ids.npy", np.asarray(page_ids, dtype=np.int64))
    (CACHE / "texts.json").write_text(json.dumps(texts), encoding="utf-8")
    print("cached page_vectors.npy, page_ids.npy, texts.json")


if __name__ == "__main__":
    main()
