"""Dev-only: comprehensive fusion sweep over page-dense, chunk-dense, BM25.

Also reports the achievable NDCG ceiling given that duplicate query strings
(twins) must receive identical rankings.
"""
from __future__ import annotations

import itertools
import json
import sys
from collections import defaultdict
from pathlib import Path
from typing import List, Sequence, Set

import numpy as np

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from core.embed import embed_queries  # noqa: E402
from eval import dcg_at_k, load_query_file, mean_ndcg_at_k, ndcg_at_k  # noqa: E402
from core.lexical import BM25Index  # noqa: E402
from utils import PUBLIC_QUERIES_PATH  # noqa: E402

CACHE = ROOT / "dev" / "cache"
K = 10


def minmax(m):
    lo = m.min(1, keepdims=True); hi = m.max(1, keepdims=True)
    return (m - lo) / np.where(hi - lo > 1e-12, hi - lo, 1.0)


def zscore(m):
    mu = m.mean(1, keepdims=True); sd = m.std(1, keepdims=True)
    return (m - mu) / np.where(sd > 1e-12, sd, 1.0)


def rank_pages(scores, page_ids, k=K):
    out = []
    for row in scores:
        top = np.argpartition(-row, k - 1)[:k]
        top = top[np.argsort(-row[top])]
        out.append([int(page_ids[i]) for i in top])
    return out


def evl(scores, page_ids, gt):
    return mean_ndcg_at_k(rank_pages(scores, page_ids), gt, k=K)


def ceiling(queries: List[str], gts: Sequence[Set[int]]) -> float:
    """Best mean NDCG@10 if identical query strings share one ranking."""
    groups = defaultdict(list)
    for i, q in enumerate(queries):
        groups[q].append(i)
    total = 0.0
    discounts = [1.0] + [1.0 / np.log2(p) for p in range(2, K + 1)]
    for q, members in groups.items():
        union = []
        owner = []
        seen = set()
        idcg = {}
        for mi in members:
            g = list(gts[mi])
            idcg[mi] = dcg_at_k([1.0] * min(len(g), K), K)
            for pid in g:
                if pid not in seen:
                    seen.add(pid); union.append(pid); owner.append(mi)
        # value per page = 1/idcg(owner); fill best positions greedily.
        vals = sorted(range(len(union)), key=lambda j: -1.0 / max(idcg[owner[j]], 1e-9))
        member_dcg = defaultdict(float)
        for pos, j in enumerate(vals[:K]):
            member_dcg[owner[j]] += discounts[pos]
        for mi in members:
            total += member_dcg[mi] / max(idcg[mi], 1e-9)
    return total / len(queries)


def main():
    pv = np.load(CACHE / "page_vectors.npy")
    cv = np.load(CACHE / "chunk_vectors.npy")
    cpid = np.load(CACHE / "chunk_page_ids.npy")
    page_ids = np.load(CACHE / "page_ids.npy")
    texts = json.loads((CACHE / "texts.json").read_text(encoding="utf-8"))

    seg = np.concatenate(([0], np.where(np.diff(cpid) != 0)[0] + 1))
    assert np.array_equal(cpid[seg], page_ids)

    rows = load_query_file(PUBLIC_QUERIES_PATH)
    queries = [r["query"] for r in rows]
    gt = [r["relevant_page_ids"] for r in rows]

    print(f"twin-constrained NDCG@10 ceiling: {ceiling(queries, gt):.4f}")

    qv = embed_queries(queries)
    pd = (qv @ pv.T).astype(np.float32)
    cd = np.maximum.reduceat((qv @ cv.T).astype(np.float32), seg, axis=1)
    bm = BM25Index.build(texts)
    lex = bm.score_batch(queries)

    print("\n=== singles ===")
    for name, m in [("page-dense", pd), ("chunk-dense", cd), ("bm25", lex)]:
        print(f"  {name:12s}: {evl(m, page_ids, gt):.4f}")

    pdn, cdn, lxn = minmax(pd), minmax(cd), minmax(lex)
    pz, cz, lz = zscore(pd), zscore(cd), zscore(lex)

    print("\n=== page-dense + bm25 (minmax) ===")
    best = (-1, None)
    for a in np.linspace(0, 1, 21):
        s = evl(a * pdn + (1 - a) * lxn, page_ids, gt)
        if s > best[0]: best = (s, round(float(a), 2))
    print(f"  best a={best[1]} -> {best[0]:.4f}")

    print("\n=== page-dense + bm25 (zscore) ===")
    best = (-1, None)
    for a in np.linspace(0, 1, 21):
        s = evl(a * pz + (1 - a) * lz, page_ids, gt)
        if s > best[0]: best = (s, round(float(a), 2))
    print(f"  best a={best[1]} -> {best[0]:.4f}")

    print("\n=== 3-way page-dense + chunk-dense + bm25 (minmax) ===")
    best = (-1, None)
    for wp in np.linspace(0, 1, 11):
        for wc in np.linspace(0, 1 - wp, int((1 - wp) * 10) + 1):
            wl = 1 - wp - wc
            if wl < -1e-9: continue
            s = evl(wp * pdn + wc * cdn + wl * lxn, page_ids, gt)
            if s > best[0]: best = (s, (round(float(wp), 2), round(float(wc), 2), round(float(wl), 2)))
    print(f"  best (wp,wc,wl)={best[1]} -> {best[0]:.4f}")

    print("\n=== 3-way zscore ===")
    best = (-1, None)
    for wp in np.linspace(0, 1, 11):
        for wc in np.linspace(0, 1 - wp, int((1 - wp) * 10) + 1):
            wl = 1 - wp - wc
            if wl < -1e-9: continue
            s = evl(wp * pz + wc * cz + wl * lz, page_ids, gt)
            if s > best[0]: best = (s, (round(float(wp), 2), round(float(wc), 2), round(float(wl), 2)))
    print(f"  best (wp,wc,wl)={best[1]} -> {best[0]:.4f}")


if __name__ == "__main__":
    main()
