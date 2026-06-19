"""Dev-only: precompute everything the Optuna study needs, so the search itself
is pure-numpy and instant.

Caches to dev/cache/optuna_prep.npz:
  dense       (Q, n)      per-query dense inner-product scores
  bm25_grid   (G, Q, n)   BM25 scores for each (k1,b) grid cell (stemmed index)
  grid        (G, 2)      the (k1, b) values
  ce          (Q, n)      cross-encoder scores, NaN where not in the candidate
                          union (only top candidates are ever reranked)
  gold/page_ids/queries

BM25 is the slow part, so the corpus is tokenized ONCE into postings and each
(k1,b) cell is then scored by cheap vector math (the weight formula matches
lexical.py). The CE union spans the grid x an alpha sweep at depth 200, so any
plausible (alpha,k1,b,pool<=150) trial's pool is covered.
"""
from __future__ import annotations

import sys
import time
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

import faiss  # noqa: E402
from core.index import load_index  # noqa: E402
from core.embed import embed_queries  # noqa: E402
from core.reranker import score_pairs  # noqa: E402
from core.lexical import tokenize, MAX_DOC_TOKENS  # noqa: E402
from utils import load_public_queries, iter_entries, entry_text  # noqa: E402

K1S = [1.0, 1.5, 2.0]
BS = [0.4, 0.6, 0.75]
ALPHA_GRID = [0.2, 0.35, 0.5, 0.6, 0.75]
UNION_DEPTH = 200
RERANK_MAX_CHARS = 2000
CE_MODEL = "cross-encoder/ms-marco-MiniLM-L-6-v2"


def build_postings(docs):
    vocab = {}
    postings_docs, postings_tf = [], []  # per term: lists
    dl = np.zeros(len(docs), dtype=np.float32)
    for i, text in enumerate(docs):
        toks = tokenize(text)[:MAX_DOC_TOKENS]
        dl[i] = len(toks)
        counts = {}
        for t in toks:
            counts[t] = counts.get(t, 0) + 1
        for t, c in counts.items():
            tid = vocab.get(t)
            if tid is None:
                tid = len(vocab); vocab[t] = tid
                postings_docs.append([]); postings_tf.append([])
            postings_docs[tid].append(i)
            postings_tf[tid].append(c)
    n_docs = len(docs)
    df = np.array([len(p) for p in postings_docs], dtype=np.float32)
    idf = np.log((n_docs - df + 0.5) / (df + 0.5) + 1.0).astype(np.float32)
    idf = np.maximum(idf, 1e-6)
    pdocs = [np.asarray(p, dtype=np.int32) for p in postings_docs]
    ptf = [np.asarray(p, dtype=np.float32) for p in postings_tf]
    return vocab, pdocs, ptf, dl, idf, float(dl.mean())


def bm25_scores(queries, vocab, pdocs, ptf, dl, idf, avgdl, k1, b, n_docs):
    out = np.zeros((len(queries), n_docs), dtype=np.float32)
    for qi, q in enumerate(queries):
        seen = set()
        row = out[qi]
        for t in tokenize(q):
            tid = vocab.get(t)
            if tid is None or tid in seen:
                continue
            seen.add(tid)
            d = pdocs[tid]; tf = ptf[tid]
            denom = tf + k1 * (1.0 - b + b * dl[d] / avgdl)
            np.add.at(row, d, idf[tid] * tf * (k1 + 1.0) / denom)
    return out


def minmax_rows(m):
    lo = m.min(axis=1, keepdims=True)
    hi = m.max(axis=1, keepdims=True)
    rng = np.where(hi - lo > 1e-12, hi - lo, 1.0)
    return (m - lo) / rng


def main():
    rows = load_public_queries()
    queries = [r["query"] for r in rows]
    gold = [list(map(int, r["relevant_page_ids"])) for r in rows]

    idx = load_index()
    page_ids = idx.page_ids
    n = page_ids.shape[0]

    fa = faiss.IndexFlatIP(idx.page_vectors.shape[1])
    fa.add(idx.page_vectors)
    qv = embed_queries(queries)
    sims, ids = fa.search(qv, n)
    dense = np.empty((len(queries), n), dtype=np.float32)
    r = np.arange(len(queries))[:, None]
    dense[r, ids] = sims

    records = list(iter_entries())
    docs = [entry_text(rr) for rr in records]
    print(f"[{time.strftime('%H:%M:%S')}] tokenizing corpus once...", flush=True)
    t0 = time.time()
    vocab, pdocs, ptf, dl, idf, avgdl = build_postings(docs)
    print(f"  done in {time.time()-t0:.1f}s, vocab={len(vocab)}", flush=True)

    grid = [(k1, b) for k1 in K1S for b in BS]
    bm25_grid = np.empty((len(grid), len(queries), n), dtype=np.float32)
    for gi, (k1, b) in enumerate(grid):
        t0 = time.time()
        bm25_grid[gi] = bm25_scores(queries, vocab, pdocs, ptf, dl, idf, avgdl,
                                    k1, b, n)
        print(f"  bm25 grid {gi+1}/{len(grid)} (k1={k1},b={b}) "
              f"{time.time()-t0:.1f}s", flush=True)

    # Candidate union for CE.
    Dn = minmax_rows(dense)
    union_cols = [set() for _ in queries]
    for gi in range(len(grid)):
        Bn = minmax_rows(bm25_grid[gi])
        for a in ALPHA_GRID:
            fused = a * Dn + (1 - a) * Bn
            top = np.argpartition(-fused, UNION_DEPTH - 1, axis=1)[:, :UNION_DEPTH]
            for qi in range(len(queries)):
                union_cols[qi].update(int(c) for c in top[qi])
    total_pairs = sum(len(s) for s in union_cols)
    print(f"[{time.strftime('%H:%M:%S')}] CE union = {total_pairs} pairs "
          f"(avg {total_pairs/len(queries):.0f}/query)", flush=True)

    texts = [str(t)[:RERANK_MAX_CHARS] for t in idx.page_texts]
    ce = np.full((len(queries), n), np.nan, dtype=np.float32)
    pairs, locs = [], []
    for qi in range(len(queries)):
        for c in union_cols[qi]:
            pairs.append((queries[qi], texts[c])); locs.append((qi, c))
    t0 = time.time()
    scores = score_pairs(CE_MODEL, pairs, max_length=512, batch_size=32)
    print(f"  CE scored {len(pairs)} pairs in {time.time()-t0:.1f}s", flush=True)
    for (qi, c), s in zip(locs, scores):
        ce[qi, c] = s

    out = ROOT / "dev" / "cache" / "optuna_prep.npz"
    out.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(
        out,
        dense=dense, bm25_grid=bm25_grid, grid=np.asarray(grid, dtype=np.float32),
        ce=ce, page_ids=page_ids,
        gold=np.asarray(_pad(gold), dtype=np.int64),
        gold_len=np.asarray([len(g) for g in gold], dtype=np.int64),
    )
    print(f"saved {out}", flush=True)


def _pad(gold):
    m = max(len(g) for g in gold)
    return [g + [-1] * (m - len(g)) for g in gold]


if __name__ == "__main__":
    main()
