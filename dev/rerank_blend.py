"""Dev-only: sweep blend of CE reranker with hybrid score; score-blend vs RRF."""
from __future__ import annotations

import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

import faiss  # noqa: E402
from index import load_index  # noqa: E402
from embed import embed_queries  # noqa: E402
from retrieve import _minmax_rows  # noqa: E402
from eval import mean_ndcg_at_k  # noqa: E402
from utils import load_public_queries  # noqa: E402

POOL = 100


def candidates():
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
    fused = idx.alpha * _minmax_rows(dense) + (1 - idx.alpha) * _minmax_rows(lex)
    order = np.argsort(-fused, axis=1)[:, :POOL]
    return queries, gold, idx.page_ids[order], np.take_along_axis(fused, order, axis=1)


def eval_scoreblend(cand_ids, cand_hyb, ce, gold, k, w):
    out = []
    for qi in range(len(gold)):
        ids = cand_ids[qi, :k]
        cs = ce[qi, :k]; hy = cand_hyb[qi, :k]
        cs_n = (cs - cs.min()) / (np.ptp(cs) or 1.0)
        hy_n = (hy - hy.min()) / (np.ptp(hy) or 1.0)
        bl = w * cs_n + (1 - w) * hy_n
        o = np.argsort(-bl)
        out.append([int(ids[i]) for i in o[:10]])
    return mean_ndcg_at_k(out, gold)


def eval_rrf(cand_ids, cand_hyb, ce, gold, k, c):
    out = []
    for qi in range(len(gold)):
        ids = cand_ids[qi, :k]
        cs = ce[qi, :k]; hy = cand_hyb[qi, :k]
        rank_ce = np.empty(k, int); rank_ce[np.argsort(-cs)] = np.arange(k)
        rank_hy = np.empty(k, int); rank_hy[np.argsort(-hy)] = np.arange(k)
        rrf = 1.0 / (c + rank_ce) + 1.0 / (c + rank_hy)
        o = np.argsort(-rrf)
        out.append([int(ids[i]) for i in o[:10]])
    return mean_ndcg_at_k(out, gold)


def main():
    queries, gold, cand_ids, cand_hyb = candidates()
    base = mean_ndcg_at_k([[int(x) for x in row[:10]] for row in cand_ids], gold)
    print(f"hybrid baseline = {base:.4f}\n")
    for slug in ["cross-encoder__ms-marco-MiniLM-L-6-v2"]:
        p = ROOT / "dev" / "cache" / f"ce_{slug}.npy"
        if not p.exists():
            continue
        ce = np.load(p)
        print(f"=== {slug} ===")
        print("  score-blend  w=ce weight")
        for k in [20, 30, 50, 100]:
            best = max(((w, eval_scoreblend(cand_ids, cand_hyb, ce, gold, k, w))
                        for w in np.arange(0.0, 1.01, 0.1)), key=lambda x: x[1])
            line = "  ".join(
                f"w={w:.1f}:{eval_scoreblend(cand_ids,cand_hyb,ce,gold,k,w):.4f}"
                for w in [0.0, 0.3, 0.5, 0.6, 0.7, 0.8, 1.0])
            print(f"  k={k:>3}  {line}   best w={best[0]:.1f}->{best[1]:.4f}")
        print("  RRF  c=60")
        for k in [20, 30, 50, 100]:
            vals = "  ".join(f"c={c}:{eval_rrf(cand_ids,cand_hyb,ce,gold,k,c):.4f}"
                             for c in [10, 30, 60])
            print(f"  k={k:>3}  {vals}")
        print()


if __name__ == "__main__":
    main()
