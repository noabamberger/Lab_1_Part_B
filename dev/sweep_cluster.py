"""Dev-only: add entity-level (sibling) aggregation on top of fusion.

Pages of one synthetic entity share an identical first sentence ("History of X"
restates X's intro). Grouping by first sentence recovers those entities; we then
lift every page by the best-scoring sibling so a fact stated on one page of the
entity surfaces the whole entity (its gold pages) together.
"""
from __future__ import annotations

import json
import re
import sys
from collections import defaultdict
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from embed import embed_queries  # noqa: E402
from eval import load_query_file, mean_ndcg_at_k  # noqa: E402
from lexical import BM25Index  # noqa: E402
from utils import PUBLIC_QUERIES_PATH, iter_entries  # noqa: E402

CACHE = ROOT / "dev" / "cache"
K = 10
MIN_FS = 40       # min first-sentence length to form a cluster
MAX_CLUSTER = 8   # ignore oversized (junk) clusters


def minmax(m):
    lo = m.min(1, keepdims=True); hi = m.max(1, keepdims=True)
    return (m - lo) / np.where(hi - lo > 1e-12, hi - lo, 1.0)


def first_sentence(c: str) -> str:
    c = c.strip()
    return re.split(r"(?<=[.])\s", c, maxsplit=1)[0][:120].lower()


def build_clusters(page_ids):
    recs = {int(r["page_id"]): r.get("content", "") for r in iter_entries()}
    groups = defaultdict(list)
    for col, pid in enumerate(page_ids):
        fs = first_sentence(recs[int(pid)])
        if len(fs) >= MIN_FS:
            groups[fs].append(col)
    # cluster id per column; -1 if singleton/ignored.
    cluster_of = np.full(len(page_ids), -1, dtype=np.int64)
    cid = 0
    members = []
    for cols in groups.values():
        if 2 <= len(cols) <= MAX_CLUSTER:
            for c in cols:
                cluster_of[c] = cid
            members.append(cols)
            cid += 1
    return cluster_of, members


def cluster_boost(scores, members, beta):
    """page' = (1-beta)*page + beta*max(sibling). Singletons unchanged."""
    out = scores.copy()
    for cols in members:
        idx = np.asarray(cols)
        cmax = scores[:, idx].max(axis=1, keepdims=True)  # (q,1)
        out[:, idx] = (1 - beta) * scores[:, idx] + beta * cmax
    return out


def rank(scores, page_ids):
    res = []
    for row in scores:
        t = np.argpartition(-row, K - 1)[:K]
        res.append([int(page_ids[i]) for i in t[np.argsort(-row[t])]])
    return res


def main():
    pv = np.load(CACHE / "page_vectors.npy")
    page_ids = np.load(CACHE / "page_ids.npy")
    texts = json.loads((CACHE / "texts.json").read_text(encoding="utf-8"))
    rows = load_query_file(PUBLIC_QUERIES_PATH)
    queries = [r["query"] for r in rows]
    gt = [r["relevant_page_ids"] for r in rows]

    cluster_of, members = build_clusters(page_ids)
    print(f"entity clusters (size 2..{MAX_CLUSTER}): {len(members)}, "
          f"clustered pages: {int((cluster_of>=0).sum())}")

    qv = embed_queries(queries)
    pd = minmax((qv @ pv.T).astype(np.float32))
    bm = BM25Index.build(texts)
    lex = minmax(bm.score_batch(queries))

    def score_at(alpha):
        return alpha * pd + (1 - alpha) * lex

    base = score_at(0.6)
    print(f"\nbaseline fused (a=0.6): {mean_ndcg_at_k(rank(base, page_ids), gt, K):.4f}")
    print("\n=== cluster boost beta sweep (on fused a=0.6) ===")
    best = (-1, None)
    for beta in [0.0, 0.2, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1.0]:
        s = cluster_boost(base, members, beta)
        nd = mean_ndcg_at_k(rank(s, page_ids), gt, K)
        print(f"  beta={beta:.1f} : {nd:.4f}")
        if nd > best[0]: best = (nd, beta)
    print(f"  best beta={best[1]} -> {best[0]:.4f}")

    print("\n=== joint alpha x beta ===")
    bestj = (-1, None)
    for alpha in [0.4, 0.5, 0.6, 0.7]:
        for beta in [0.5, 0.7, 0.85, 1.0]:
            s = cluster_boost(score_at(alpha), members, beta)
            nd = mean_ndcg_at_k(rank(s, page_ids), gt, K)
            if nd > bestj[0]: bestj = (nd, (alpha, beta))
    print(f"  best (alpha,beta)={bestj[1]} -> {bestj[0]:.4f}")


if __name__ == "__main__":
    main()
