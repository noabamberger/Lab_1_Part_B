"""Dev-only: score the hybrid top-POOL candidates with one reranker, cache to disk.

Usage: python dev/ce_cache.py <hf-model-name> [max_length]
"""
from __future__ import annotations

import sys
import time
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

import faiss  # noqa: E402
from sentence_transformers import CrossEncoder  # noqa: E402
from index import load_index  # noqa: E402
from embed import embed_queries  # noqa: E402
from retrieve import _minmax_rows  # noqa: E402
from utils import load_public_queries, iter_entries, entry_text  # noqa: E402

POOL = 100
MAX_CHARS = 2000


def main() -> None:
    model_name = sys.argv[1]
    max_len = int(sys.argv[2]) if len(sys.argv) > 2 else 512

    rows = load_public_queries()
    queries = [r["query"] for r in rows]
    idx = load_index()
    fa = faiss.IndexFlatIP(idx.page_vectors.shape[1])
    fa.add(idx.page_vectors)
    qv = embed_queries(queries)
    n = idx.page_ids.shape[0]
    sims, ids = fa.search(qv, n)
    dense = np.empty((len(queries), n), dtype=np.float32)
    r = np.arange(len(queries))[:, None]
    dense[r, ids] = sims
    lex = idx.bm25.score_batch(queries)
    fused = idx.alpha * _minmax_rows(dense) + (1 - idx.alpha) * _minmax_rows(lex)
    order = np.argsort(-fused, axis=1)[:, :POOL]
    cand_ids = idx.page_ids[order]

    needed = {int(p) for row in cand_ids for p in row}
    texts = {}
    for rec in iter_entries():
        pid = int(rec["page_id"])
        if pid in needed:
            texts[pid] = entry_text(rec)[:MAX_CHARS]

    print(f"[{time.strftime('%H:%M:%S')}] loading {model_name}", flush=True)
    ce = CrossEncoder(model_name, max_length=max_len)
    pairs = []
    for qi, q in enumerate(queries):
        for pid in cand_ids[qi]:
            pairs.append((q, texts.get(int(pid), "")))
    t0 = time.time()
    scores = ce.predict(pairs, batch_size=32, show_progress_bar=False)
    print(f"[{time.strftime('%H:%M:%S')}] scored {len(pairs)} pairs in "
          f"{time.time()-t0:.1f}s", flush=True)

    ce_full = np.asarray(scores, dtype=np.float32).reshape(len(queries), POOL)
    cache = ROOT / "dev" / "cache"
    cache.mkdir(parents=True, exist_ok=True)
    out = cache / f"ce_{model_name.replace('/', '__')}.npy"
    np.save(out, ce_full)
    print(f"saved {out}", flush=True)


if __name__ == "__main__":
    main()
