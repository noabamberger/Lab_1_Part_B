"""Dev-only: per-query rank diagnostics for the gold pages."""
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


def rank_of(scores_row, page_ids, gold):
    order = np.argsort(-scores_row)
    pos = {int(page_ids[i]): r for r, i in enumerate(order[:200])}
    return {g: pos.get(g, 999) for g in gold}


def main() -> None:
    pv = np.load(CACHE / "page_vectors.npy")
    page_ids = np.load(CACHE / "page_ids.npy")
    texts = json.loads((CACHE / "texts.json").read_text(encoding="utf-8"))
    id2idx = {int(p): i for i, p in enumerate(page_ids)}

    rows = load_query_file(PUBLIC_QUERIES_PATH)
    queries = [r["query"] for r in rows]
    gts = [r["relevant_page_ids"] for r in rows]

    qv = embed_queries(queries)
    dense = qv @ pv.T
    bm25 = BM25Index.build(texts)
    lex = bm25.score_batch(queries)

    worst = []
    for i, (q, gt) in enumerate(zip(queries, gts)):
        rd = rank_of(dense[i], page_ids, gt)
        rl = rank_of(lex[i], page_ids, gt)
        best_d = min(rd.values())
        worst.append((best_d, i, q, gt, rd, rl))
    worst.sort(reverse=True)
    print("=== hardest queries (by best dense rank of any gold) ===")
    for best_d, i, q, gt, rd, rl in worst[:12]:
        print(f"\n[{i}] dense_best_rank={best_d}  Q: {q}")
        for g in gt:
            print(f"    gold {g}: dense_rank={rd[g]} bm25_rank={rl[g]}")
            t = texts[id2idx[g]][:200].replace("\n", " ")
            print(f"        text: {t}")


if __name__ == "__main__":
    main()
