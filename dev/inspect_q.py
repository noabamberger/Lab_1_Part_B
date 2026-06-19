"""Dev-only: show top-10 retrieved pages vs gold for selected queries."""
from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from core.embed import embed_queries  # noqa: E402
from eval import load_query_file  # noqa: E402
from core.lexical import BM25Index  # noqa: E402
from utils import PUBLIC_QUERIES_PATH  # noqa: E402

CACHE = ROOT / "dev" / "cache"
K = 10
SHOW = [0, 19, 16, 13, 48]


def minmax(m):
    lo = m.min(1, keepdims=True); hi = m.max(1, keepdims=True)
    return (m - lo) / np.where(hi - lo > 1e-12, hi - lo, 1.0)


def main():
    pv = np.load(CACHE / "page_vectors.npy")
    page_ids = np.load(CACHE / "page_ids.npy")
    texts = json.loads((CACHE / "texts.json").read_text(encoding="utf-8"))
    id2idx = {int(p): i for i, p in enumerate(page_ids)}
    rows = load_query_file(PUBLIC_QUERIES_PATH)
    queries = [r["query"] for r in rows]
    gt = [r["relevant_page_ids"] for r in rows]

    qv = embed_queries(queries)
    pd = minmax((qv @ pv.T).astype(np.float32))
    bm = BM25Index.build(texts)
    lexraw = bm.score_batch(queries)
    lex = minmax(lexraw)
    fused = 0.6 * pd + 0.4 * lex

    def show_rank(name, scores, i):
        row = scores[i]
        order = np.argsort(-row)[:K]
        print(f"  -- {name} top{K}:")
        for r, idx in enumerate(order):
            pid = int(page_ids[idx])
            star = "*GOLD*" if pid in gt[i] else "      "
            t = texts[id2idx[pid]][:60].replace("\n", " ")
            print(f"     {r+1:2d} {star} {pid:6d} {row[idx]:.3f}  {t}")
        # gold ranks
        gr = {}
        full = np.argsort(-row)
        pos = {int(page_ids[j]): k for k, j in enumerate(full[:500])}
        for g in gt[i]:
            gr[g] = pos.get(g, 999)
        print(f"     gold ranks: {gr}")

    for i in SHOW:
        print(f"\n==== [{i}] {queries[i]}")
        print(f"     gold={sorted(gt[i])}")
        show_rank("fused", fused, i)
        show_rank("bm25", lex, i)


if __name__ == "__main__":
    main()
