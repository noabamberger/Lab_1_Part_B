"""Dev-only: evaluate cross-encoder rerankers on top of the hybrid retriever.

Stage 1 (candidate generation): hybrid dense+BM25 fused ranking -> top-N page ids.
Stage 2 (rerank): a CrossEncoder scores (query, page_text) for those N and reorders.
We report NDCG@10 for pure-rerank and for a blend with the hybrid score, across
candidate depths and models.
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
from eval import mean_ndcg_at_k  # noqa: E402
from utils import load_public_queries, iter_entries, entry_text  # noqa: E402

MODELS = [
    "cross-encoder/ms-marco-MiniLM-L-6-v2",
    "cross-encoder/ms-marco-MiniLM-L-12-v2",
]
POOL = 100          # candidates fed to the reranker per query
KS = [20, 50, 100]  # candidate depths to evaluate
MAX_CHARS = 2000    # truncate page text fed to reranker (speed)


def build_candidates():
    rows = load_public_queries()
    queries = [r["query"] for r in rows]
    gold = [set(r["relevant_page_ids"]) for r in rows]

    idx = load_index()
    fa = faiss.IndexFlatIP(idx.page_vectors.shape[1])
    fa.add(idx.page_vectors)

    qv = embed_queries(queries)
    n_pages = idx.page_ids.shape[0]
    sims, ids = fa.search(qv, n_pages)
    dense = np.empty((len(queries), n_pages), dtype=np.float32)
    r = np.arange(len(queries))[:, None]
    dense[r, ids] = sims
    lex = idx.bm25.score_batch(queries)
    fused = idx.alpha * _minmax_rows(dense) + (1 - idx.alpha) * _minmax_rows(lex)

    page_ids = idx.page_ids
    order = np.argsort(-fused, axis=1)[:, :max(KS)]
    cand_ids = page_ids[order]                          # (q, POOL) page ids
    cand_hyb = np.take_along_axis(fused, order, axis=1)  # hybrid score for blend
    return queries, gold, cand_ids, cand_hyb


def load_texts(needed):
    texts = {}
    for rec in iter_entries():
        pid = int(rec["page_id"])
        if pid in needed:
            texts[pid] = entry_text(rec)[:MAX_CHARS]
    return texts


def main() -> None:
    queries, gold, cand_ids, cand_hyb = build_candidates()
    base = mean_ndcg_at_k([[int(x) for x in row[:10]] for row in cand_ids], gold)
    print(f"hybrid baseline NDCG@10 = {base:.4f}  ({len(queries)} queries)\n")

    needed = {int(p) for row in cand_ids[:, :max(KS)] for p in row}
    texts = load_texts(needed)

    cache_dir = ROOT / "dev" / "cache"
    cache_dir.mkdir(parents=True, exist_ok=True)

    for model_name in MODELS:
        print(f"=== {model_name} ===")
        slug = model_name.replace("/", "__")
        cpath = cache_dir / f"ce_{slug}.npy"
        if cpath.exists():
            ce_full = np.load(cpath)
            print("  loaded cached CE scores")
        else:
            t0 = time.time()
            ce = CrossEncoder(model_name, max_length=512)
            pairs, spans = [], []
            for qi, q in enumerate(queries):
                ids = cand_ids[qi, :max(KS)]
                start = len(pairs)
                for pid in ids:
                    pairs.append((q, texts.get(int(pid), "")))
                spans.append((start, len(pairs)))
            scores = ce.predict(pairs, batch_size=64, show_progress_bar=False)
            scores = np.asarray(scores, dtype=np.float32)
            print(f"  scored {len(pairs)} pairs in {time.time()-t0:.1f}s")
            ce_full = np.full((len(queries), max(KS)), -1e9, dtype=np.float32)
            for qi, (s, e) in enumerate(spans):
                ce_full[qi, : e - s] = scores[s:e]
            np.save(cpath, ce_full)

        for k in KS:
            # pure rerank within top-k pool
            pure = []
            blend = []
            for qi in range(len(queries)):
                ids = cand_ids[qi, :k]
                cs = ce_full[qi, :k]
                o = np.argsort(-cs)
                pure.append([int(ids[i]) for i in o[:10]])
                # blend: minmax CE + minmax hybrid
                cs_n = (cs - cs.min()) / (np.ptp(cs) or 1.0)
                hy = cand_hyb[qi, :k]
                hy_n = (hy - hy.min()) / (np.ptp(hy) or 1.0)
                bl = 0.7 * cs_n + 0.3 * hy_n
                ob = np.argsort(-bl)
                blend.append([int(ids[i]) for i in ob[:10]])
            npure = mean_ndcg_at_k(pure, gold)
            nblend = mean_ndcg_at_k(blend, gold)
            print(f"  k={k:>3}  pure={npure:.4f}  blend(0.7ce+0.3hyb)={nblend:.4f}")
        print()


if __name__ == "__main__":
    main()
