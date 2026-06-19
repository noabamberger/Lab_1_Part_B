"""Dev-only: recall@k of the hybrid candidate generator on the public queries.

A reranker can only fix ordering among the candidates it is given. This measures
how deep we must retrieve so the gold pages are (almost) always in the pool.
"""
from __future__ import annotations

import sys
import time
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from core.index import load_index  # noqa: E402
from core.embed import embed_queries  # noqa: E402
from core.retrieve import _minmax_rows  # noqa: E402
from utils import load_public_queries  # noqa: E402
import faiss  # noqa: E402


def main() -> None:
    rows = load_public_queries()
    queries = [r["query"] for r in rows]
    gold = [set(r["relevant_page_ids"]) for r in rows]

    idx = load_index()
    fa = faiss.IndexFlatIP(idx.page_vectors.shape[1])
    fa.add(idx.page_vectors)

    t0 = time.time()
    qv = embed_queries(queries)
    n_pages = idx.page_ids.shape[0]
    sims, ids = fa.search(qv, n_pages)
    dense = np.empty((len(queries), n_pages), dtype=np.float32)
    r = np.arange(len(queries))[:, None]
    dense[r, ids] = sims
    lex = idx.bm25.score_batch(queries)
    alpha = idx.alpha
    fused = alpha * _minmax_rows(dense) + (1 - alpha) * _minmax_rows(lex)
    print(f"scored in {time.time()-t0:.1f}s")

    page_ids = idx.page_ids
    # Full ranking per query (descending).
    order = np.argsort(-fused, axis=1)

    ks = [10, 20, 30, 50, 100, 150, 200, 300, 500, 1000]
    print(f"{'k':>6}  {'mean_recall':>11}  {'queries@100%':>12}")
    for k in ks:
        recs = []
        full = 0
        for qi in range(len(queries)):
            g = gold[qi]
            if not g:
                continue
            topk = {int(page_ids[i]) for i in order[qi, :k]}
            hit = len(g & topk)
            rec = hit / len(g)
            recs.append(rec)
            if hit == len(g):
                full += 1
        print(f"{k:>6}  {np.mean(recs):>11.4f}  {full:>5}/{len(recs)}")

    # Also: for each gold doc, the best (min) rank achieved across its query.
    worst_needed = 0
    for qi in range(len(queries)):
        g = gold[qi]
        if not g:
            continue
        ranks = {int(page_ids[i]): pos for pos, i in enumerate(order[qi])}
        for d in g:
            worst_needed = max(worst_needed, ranks.get(d, n_pages))
    print(f"\ndeepest gold rank over all queries: {worst_needed} "
          f"(k must be >= {worst_needed+1} for 100% recall)")


if __name__ == "__main__":
    main()
