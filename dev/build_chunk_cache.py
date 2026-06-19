"""Dev-only: embed all chunks once and cache for fast fusion sweeps.

Encodes in blocks with flushed progress logging so the long CPU run can be
monitored, and writes the result atomically at the end.
"""
from __future__ import annotations

import sys
import time
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from core.chunk import chunk_corpus  # noqa: E402
from core.embed import get_model  # noqa: E402
from utils import iter_entries  # noqa: E402

CACHE = ROOT / "dev" / "cache"
BLOCK = 20000


def log(msg: str) -> None:
    print(f"[{time.strftime('%H:%M:%S')}] {msg}", flush=True)


def main() -> None:
    CACHE.mkdir(parents=True, exist_ok=True)
    records = list(iter_entries())
    t0 = time.perf_counter()
    chunks = chunk_corpus(records)
    log(f"chunked {len(records)} pages -> {len(chunks)} chunks in {time.perf_counter()-t0:.1f}s")

    texts = [c.text for c in chunks]
    cpid = np.asarray([c.page_id for c in chunks], dtype=np.int64)

    model = get_model()
    parts = []
    t1 = time.perf_counter()
    for start in range(0, len(texts), BLOCK):
        block = texts[start : start + BLOCK]
        vecs = model.encode(
            block,
            batch_size=256,
            show_progress_bar=False,
            convert_to_numpy=True,
            normalize_embeddings=True,
        )
        parts.append(np.asarray(vecs, dtype=np.float32))
        done = start + len(block)
        rate = done / (time.perf_counter() - t1)
        eta = (len(texts) - done) / rate if rate else 0
        log(f"embedded {done}/{len(texts)} chunks ({rate:.0f}/s, eta {eta/60:.1f}m)")

    vectors = np.vstack(parts)
    log(f"embedded all in {time.perf_counter()-t1:.1f}s -> {vectors.shape}")

    tmp = CACHE / "chunk_vectors.tmp.npy"
    np.save(tmp, vectors)
    tmp.replace(CACHE / "chunk_vectors.npy")
    np.save(CACHE / "chunk_page_ids.npy", cpid)
    log("cached chunk_vectors.npy, chunk_page_ids.npy")


if __name__ == "__main__":
    main()
