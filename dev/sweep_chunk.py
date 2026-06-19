"""Dev-only: sweep chunk-maxpool + BM25 fusion against public queries."""
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


def maxpool_to_pages(chunk_scores: np.ndarray, seg_starts: np.ndarray, n_pages: int) -> np.ndarray:
    """Max chunk score per page. chunk_page_ids must be page-contiguous."""
    # np.maximum.reduceat over contiguous segments.
    pooled = np.maximum.reduceat(chunk_scores, seg_starts, axis=1)
    return pooled  # (n_queries, n_pages)


def minmax_rows(m: np.ndarray) -> np.ndarray:
    lo = m.min(axis=1, keepdims=True)
    hi = m.max(axis=1, keepdims=True)
    rng = np.where(hi - lo > 1e-12, hi - lo, 1.0)
    return (m - lo) / rng


def rank_pages(scores: np.ndarray, page_ids: np.ndarray, k: int = K) -> List[List[int]]:
    out: List[List[int]] = []
    for row in scores:
        top = np.argpartition(-row, min(k, len(row) - 1))[:k]
        top = top[np.argsort(-row[top])]
        out.append([int(page_ids[i]) for i in top])
    return out


def evaluate(scores: np.ndarray, page_ids: np.ndarray, gt: Sequence[Set[int]]) -> float:
    return mean_ndcg_at_k(rank_pages(scores, page_ids), gt, k=K)


def main() -> None:
    cv = np.load(CACHE / "chunk_vectors.npy")
    cpid = np.load(CACHE / "chunk_page_ids.npy")
    texts = json.loads((CACHE / "texts.json").read_text(encoding="utf-8"))
    page_ids_full = np.load(CACHE / "page_ids.npy")
    print(f"chunks: {cv.shape}, pages: {len(page_ids_full)}")

    # Page segments (chunks are contiguous per page in build order).
    seg_starts = np.concatenate(([0], np.where(np.diff(cpid) != 0)[0] + 1))
    page_order = cpid[seg_starts]  # page_id per segment, in order
    assert len(page_order) == len(page_ids_full), (len(page_order), len(page_ids_full))

    rows = load_query_file(PUBLIC_QUERIES_PATH)
    queries = [r["query"] for r in rows]
    gt = [r["relevant_page_ids"] for r in rows]

    qv = embed_queries(queries)
    t0 = time.perf_counter()
    chunk_scores = (qv @ cv.T).astype(np.float32)
    dense = maxpool_to_pages(chunk_scores, seg_starts, len(page_order))
    print(f"dense chunk+maxpool: {time.perf_counter()-t0:.2f}s -> {dense.shape}")

    bm25 = BM25Index.build(texts)
    # Align BM25 columns to page_order (texts.json is in page_ids_full order).
    lex_full = bm25.score_batch(queries)  # columns = page_ids_full order
    # page_ids_full and page_order are both sorted corpus order -> identical.
    assert np.array_equal(page_order, page_ids_full)
    lex = lex_full

    print("\n=== single retrievers (chunk dense vs bm25) ===")
    print(f"dense chunk-maxpool : {evaluate(dense, page_order, gt):.4f}")
    print(f"bm25 page-level     : {evaluate(lex, page_order, gt):.4f}")

    dn = minmax_rows(dense)
    ln = minmax_rows(lex)
    print("\n=== weighted min-max fusion: alpha*dense + (1-alpha)*bm25 ===")
    best = (-1.0, None)
    for alpha in [round(0.05 * i, 2) for i in range(21)]:
        ndcg = evaluate(alpha * dn + (1 - alpha) * ln, page_order, gt)
        if ndcg > best[0]:
            best = (ndcg, alpha)
    for alpha in [0.3, 0.4, 0.5, 0.6, 0.7]:
        print(f"  alpha={alpha:.2f} : {evaluate(alpha*dn+(1-alpha)*ln, page_order, gt):.4f}")
    print(f"  >>> best alpha={best[1]} -> {best[0]:.4f}")


if __name__ == "__main__":
    main()
