"""Dev-only: clean ablation ladder on the current public queries (29).

dense-only | bm25-only | hybrid | (+reranker via cached CE scores).
Stemming is on in the loaded BM25 index.
"""
from __future__ import annotations

import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

import faiss  # noqa: E402
from index import load_index  # noqa: E402
from embed import embed_queries  # noqa: E402
from retrieve import _minmax_rows, _minmax_1d  # noqa: E402
from eval import mean_ndcg_at_k  # noqa: E402
from utils import load_public_queries  # noqa: E402


def topk(scores, page_ids, gold):
    out = [[int(p) for p in page_ids[np.argsort(-scores[qi])[:10]]]
           for qi in range(scores.shape[0])]
    return mean_ndcg_at_k(out, gold)


def main():
    rows = load_public_queries()
    queries = [r["query"] for r in rows]
    gold = [set(r["relevant_page_ids"]) for r in rows]
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
    a = idx.alpha
    fused = a * _minmax_rows(dense) + (1 - a) * _minmax_rows(lex)

    print(f"dense only        {topk(dense, idx.page_ids, gold):.4f}")
    print(f"bm25 only (stem)  {topk(lex, idx.page_ids, gold):.4f}")
    print(f"hybrid (stem)     {topk(fused, idx.page_ids, gold):.4f}")

    ce = np.load(ROOT / "dev" / "cache" /
                 "ce_cross-encoder__ms-marco-MiniLM-L-6-v2.npy")
    pool = ce.shape[1]
    order = np.argsort(-fused, axis=1)[:, :pool]
    w = 0.5
    out = []
    for qi in range(len(queries)):
        cols = order[qi]
        bl = w * _minmax_1d(ce[qi]) + (1 - w) * _minmax_1d(fused[qi, cols])
        out.append([int(idx.page_ids[cols[i]]) for i in np.argsort(-bl)[:10]])
    print(f"+ reranker blend  {mean_ndcg_at_k(out, gold):.4f}")


if __name__ == "__main__":
    main()
