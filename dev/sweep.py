"""Dev-only: sweep retrieval configurations against the 50 public queries.

Loads cached corpus embeddings (dev/build_cache.py) and the BM25 index, then
reports mean NDCG@10 for dense-only, BM25-only, and several hybrid fusions so we
can justify the final design empirically.
"""
from __future__ import annotations

import json
import sys
import time
from pathlib import Path
from typing import List, Sequence, Set

import numpy as np

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from embed import embed_queries  # noqa: E402
from eval import load_query_file, mean_ndcg_at_k  # noqa: E402
from lexical import BM25Index  # noqa: E402
from utils import PUBLIC_QUERIES_PATH  # noqa: E402

CACHE = ROOT / "dev" / "cache"
K = 10


def rank_pages(scores: np.ndarray, page_ids: np.ndarray, k: int = K) -> List[List[int]]:
    out: List[List[int]] = []
    for row in scores:
        top = np.argpartition(-row, min(k, len(row) - 1))[:k]
        top = top[np.argsort(-row[top])]
        out.append([int(page_ids[i]) for i in top])
    return out


def evaluate(scores: np.ndarray, page_ids: np.ndarray, gt: Sequence[Set[int]]) -> float:
    ranked = rank_pages(scores, page_ids)
    return mean_ndcg_at_k(ranked, gt, k=K)


def minmax_rows(m: np.ndarray) -> np.ndarray:
    lo = m.min(axis=1, keepdims=True)
    hi = m.max(axis=1, keepdims=True)
    rng = np.where(hi - lo > 1e-12, hi - lo, 1.0)
    return (m - lo) / rng


def rrf(dense: np.ndarray, lex: np.ndarray, k: int = 60) -> np.ndarray:
    """Reciprocal rank fusion of two score matrices."""
    def ranks(m: np.ndarray) -> np.ndarray:
        order = np.argsort(-m, axis=1)
        r = np.empty_like(order)
        rows = np.arange(m.shape[0])[:, None]
        r[rows, order] = np.arange(m.shape[1])[None, :]
        return r
    rd = ranks(dense)
    rl = ranks(lex)
    return 1.0 / (k + rd) + 1.0 / (k + rl)


def main() -> None:
    page_vectors = np.load(CACHE / "page_vectors.npy")
    page_ids = np.load(CACHE / "page_ids.npy")
    texts = json.loads((CACHE / "texts.json").read_text(encoding="utf-8"))
    print(f"corpus: {page_vectors.shape[0]} pages, dim {page_vectors.shape[1]}")

    rows = load_query_file(PUBLIC_QUERIES_PATH)
    queries = [r["query"] for r in rows]
    gt = [r["relevant_page_ids"] for r in rows]

    qv = embed_queries(queries)
    dense = (qv @ page_vectors.T).astype(np.float32)

    t0 = time.perf_counter()
    bm25 = BM25Index.build(texts)
    print(f"bm25 build: {time.perf_counter()-t0:.1f}s, vocab {len(bm25.vocab)}")
    t1 = time.perf_counter()
    lex = bm25.score_batch(queries)
    print(f"bm25 score: {time.perf_counter()-t1:.2f}s")

    print("\n=== single retrievers ===")
    print(f"dense-only : {evaluate(dense, page_ids, gt):.4f}")
    print(f"bm25-only  : {evaluate(lex, page_ids, gt):.4f}")

    dn = minmax_rows(dense)
    ln = minmax_rows(lex)
    print("\n=== weighted min-max fusion: alpha*dense + (1-alpha)*bm25 ===")
    best = (-1.0, None)
    for alpha in [0.0, 0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1.0]:
        s = alpha * dn + (1 - alpha) * ln
        ndcg = evaluate(s, page_ids, gt)
        print(f"  alpha={alpha:.1f} : {ndcg:.4f}")
        if ndcg > best[0]:
            best = (ndcg, alpha)
    print(f"  best alpha={best[1]} -> {best[0]:.4f}")

    print("\n=== reciprocal rank fusion ===")
    for k in [10, 30, 60, 100]:
        s = rrf(dense, lex, k=k)
        print(f"  rrf k={k:3d} : {evaluate(s, page_ids, gt):.4f}")


if __name__ == "__main__":
    main()
