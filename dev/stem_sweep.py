"""Dev-only: does stemming the BM25 tokenizer help recall / hybrid NDCG?

Rebuilds BM25 with a stemming tokenizer (monkeypatched) and compares against the
current index on both hybrid NDCG@10 and recall@k.
"""
from __future__ import annotations

import sys
import time
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

import faiss  # noqa: E402
import core.lexical as lexical  # noqa: E402
from porter import stem  # noqa: E402
from core.index import load_index  # noqa: E402
from core.embed import embed_queries  # noqa: E402
from core.retrieve import _minmax_rows  # noqa: E402
from eval import mean_ndcg_at_k  # noqa: E402
from utils import load_public_queries, iter_entries, entry_text  # noqa: E402

_stem_cache: dict[str, str] = {}


def _st(t: str) -> str:
    s = _stem_cache.get(t)
    if s is None:
        s = stem(t)
        _stem_cache[t] = s
    return s


def make_stem_tokenizer(orig):
    def tok(text: str):
        base = lexical._TOKEN_RE.findall(text.lower())
        words = [t for t in base if t.isalpha()]
        stemmed = [_st(t) for t in words]
        toks = []
        wi = 0
        for t in base:
            if t.isalpha():
                toks.append(stemmed[wi]); wi += 1
            else:
                toks.append(t)
        extra = [f"{t[:3]}x" for t in base if lexical._YEAR_RE.fullmatch(t)]
        extra += [f"{a}_{b}" for a, b in zip(stemmed, stemmed[1:])]
        return toks + extra
    return tok


def hybrid_scores(queries, idx, bm25):
    fa = faiss.IndexFlatIP(idx.page_vectors.shape[1])
    fa.add(idx.page_vectors)
    qv = embed_queries(queries)
    n = idx.page_ids.shape[0]
    sims, ids = fa.search(qv, n)
    dense = np.empty((len(queries), n), dtype=np.float32)
    r = np.arange(len(queries))[:, None]
    dense[r, ids] = sims
    lex = bm25.score_batch(queries)
    return idx.alpha * _minmax_rows(dense) + (1 - idx.alpha) * _minmax_rows(lex)


def recall_table(fused, page_ids, gold, ks):
    order = np.argsort(-fused, axis=1)
    out = {}
    for k in ks:
        recs = []
        for qi in range(len(gold)):
            g = gold[qi]
            topk = {int(page_ids[i]) for i in order[qi, :k]}
            recs.append(len(g & topk) / len(g))
        out[k] = float(np.mean(recs))
    return out


def main():
    rows = load_public_queries()
    queries = [r["query"] for r in rows]
    gold = [set(r["relevant_page_ids"]) for r in rows]
    idx = load_index()
    page_texts = [entry_text(r) for r in iter_entries()]

    ks = [10, 20, 50, 100]
    # Baseline (current, already-built bm25 from artifacts).
    fused0 = hybrid_scores(queries, idx, idx.bm25)
    base = mean_ndcg_at_k([[int(page_ids) for page_ids in
                            idx.page_ids[np.argsort(-fused0[qi])[:10]]]
                           for qi in range(len(queries))], gold)
    rec0 = recall_table(fused0, idx.page_ids, gold, ks)
    print(f"current  NDCG@10={base:.4f}  recall={rec0}")

    # Stemmed BM25.
    orig = lexical.tokenize
    lexical.tokenize = make_stem_tokenizer(orig)
    t0 = time.time()
    bm25s = lexical.BM25Index.build(page_texts)
    print(f"built stemmed BM25 in {time.time()-t0:.1f}s, vocab={len(bm25s.vocab)}")
    fused1 = hybrid_scores(queries, idx, bm25s)
    stemmed = mean_ndcg_at_k([[int(p) for p in
                               idx.page_ids[np.argsort(-fused1[qi])[:10]]]
                              for qi in range(len(queries))], gold)
    rec1 = recall_table(fused1, idx.page_ids, gold, ks)
    print(f"stemmed  NDCG@10={stemmed:.4f}  recall={rec1}")
    lexical.tokenize = orig


if __name__ == "__main__":
    main()
