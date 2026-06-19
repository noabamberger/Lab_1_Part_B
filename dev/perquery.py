"""Dev-only: per-query NDCG breakdown for the best fusion config."""
from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from core.embed import embed_queries  # noqa: E402
from eval import load_query_file, ndcg_at_k  # noqa: E402
from core.lexical import BM25Index  # noqa: E402
from utils import PUBLIC_QUERIES_PATH  # noqa: E402

CACHE = ROOT / "dev" / "cache"
K = 10


def minmax(m):
    lo = m.min(1, keepdims=True); hi = m.max(1, keepdims=True)
    return (m - lo) / np.where(hi - lo > 1e-12, hi - lo, 1.0)


def main():
    pv = np.load(CACHE / "page_vectors.npy")
    page_ids = np.load(CACHE / "page_ids.npy")
    texts = json.loads((CACHE / "texts.json").read_text(encoding="utf-8"))
    rows = load_query_file(PUBLIC_QUERIES_PATH)
    queries = [r["query"] for r in rows]
    gt = [r["relevant_page_ids"] for r in rows]

    qv = embed_queries(queries)
    pd = minmax((qv @ pv.T).astype(np.float32))
    bm = BM25Index.build(texts)
    lex = minmax(bm.score_batch(queries))
    fused = 0.6 * pd + 0.4 * lex

    def topk(row):
        t = np.argpartition(-row, K - 1)[:K]
        return [int(page_ids[i]) for i in t[np.argsort(-row[t])]]

    res = []
    for i, q in enumerate(queries):
        nd = ndcg_at_k(topk(fused[i]), gt[i], K)
        res.append((nd, i, q))
    res.sort()
    print("=== per-query NDCG (worst first) ===")
    for nd, i, q in res:
        print(f"  {nd:.3f}  [{i:2d}] {q[:78]}")
    print(f"\nmean = {sum(r[0] for r in res)/len(res):.4f}")
    # bucket: specific (has a digit) vs generic
    spec = [r[0] for r in res if any(ch.isdigit() for ch in r[2])]
    gen = [r[0] for r in res if not any(ch.isdigit() for ch in r[2])]
    print(f"queries w/ number: n={len(spec)} mean={np.mean(spec):.4f}")
    print(f"queries no number: n={len(gen)} mean={np.mean(gen):.4f}")


if __name__ == "__main__":
    main()
