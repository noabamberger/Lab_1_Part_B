# Section B — Hybrid Retrieval + Reranking

> A three-stage retrieval system over **~27,000** Wikipedia-style pages.
> `run(queries)` returns, per query, a ranked list of `page_id`s (best first);
> only the top 10 are scored, by **mean NDCG@10**.

<table>
<tr><td><b>Final score</b></td><td>NDCG@10 = <b>0.485</b> (29 public queries)</td></tr>
<tr><td><b>Query latency</b></td><td>seconds on GPU · well under the 60 s budget</td></tr>
<tr><td><b>Stack</b></td><td>MiniLM embeddings · FAISS · custom NumPy BM25 · MS-MARCO cross-encoder</td></tr>
</table>

🎬 **Video presentation:** <!-- TODO: paste public video link here -->

---

## 🚀 Quickstart

```bash
cd "Lab_1_Part_B"
pip install -r requirements.txt      # numpy · sentence-transformers · faiss-cpu
```

The prebuilt index is committed under `artifacts/`, so a fresh clone evaluates
directly — **the index is not rebuilt at grading time:**

```bash
python scripts/eval_public.py
# public_queries=29
# mean_ndcg@10=0.4854
# query_phase_time ≈ 130 s CPU (this machine) / a few seconds on the grading GPU
```

Two models download from the HuggingFace hub on first use (not shipped):
`all-MiniLM-L6-v2` (embeddings, ~80 MB) and `ms-marco-MiniLM-L-6-v2` (reranker,
~80 MB). The reranker is deliberately small so its download + load + inference
fit the query-time budget; on the grading GPU it runs ~20–30× faster than the CPU
number above.

> **Rebuild (optional, offline only):** `python scripts/build_index.py` —
> ≈13 min CPU / ~1–2 min GPU. Only the page-embedding stage is GPU-accelerated;
> the BM25 build and disk I/O are CPU-bound.

---

## 🧭 Pipeline

The corpus mixes long real articles with short **synthetic** entries that restate
a queried fact (a specific year, population, or named phrase). Queries are
natural-language paraphrases — the whole design targets that structure.

```
          ┌──────────────────────── candidate generation ────────────────────────┐
query ──▶ │  Dense  (MiniLM + FAISS) ──┐                                           │
          │                            ├─▶ 0.7·dense + 0.3·bm25 ─▶ top-120 pool ── │ ─▶ Rerank ─▶ top-10
          │  Lexical (BM25, stemmed) ──┘     (per-query min-max)                   │   (cross-encoder,
          └───────────────────────────────────────────────────────────────────────┘    blend 0.6·CE + 0.4·hybrid)
```

| Stage | Module | Method |
|-------|--------|--------|
| **Chunk** | `core/chunk.py` | One unit **per whole page** (title + content). Windowed chunking *underperformed* — answer pages are short, so windows only dilute the page signal and let a long distractor win on a stray window. |
| **Embed** | `core/embed.py` | `all-MiniLM-L6-v2`, L2-normalized (cosine = inner product). |
| **Index** | `core/index/`, `core/lexical/` | **Dense:** FAISS `IndexFlatIP` over page vectors. **Lexical:** a custom NumPy **BM25** (`k1=2.0, b=0.75`) with **Porter stemming** plus two corpus-specific features — *decade tokens* (`1826`→`182x`, so "the 1820s" matches an exact-year page) and *word bigrams* (so "point guard", "cold-water fisheries" match as phrases). |
| **Retrieve** | `core/retrieve/` | Dense and BM25 scores are each per-query **min-max normalized** and fused `0.7·dense + 0.3·bm25` into a candidate pool. |
| **Rerank** | `core/reranker.py` | A cross-encoder (`ms-marco-MiniLM-L-6-v2`) rescores the top **120** candidates; the result is blended `0.6·reranker + 0.4·hybrid` and the top-10 returned. |

---

## 💡 Design rationale

<details open>
<summary><b>Why a reranker, and how deep a pool? (recall@k)</b></summary>

The candidate generator has **high recall but imperfect ordering** — gold pages
are usually in the pool, just not in the top 10:

| depth *k* | mean recall@k | oracle NDCG@10 (perfect ordering of top-*k*) |
|:---:|:---:|:---:|
| 10  | 0.59 | 0.60 |
| 20  | 0.75 | 0.80 |
| 50  | 0.89 | 0.91 |
| 100 | 0.93 | **0.94** |

A *perfect* reranker over the top-100 could reach ~0.94 versus the hybrid's ~0.45
— **almost the entire gap is ordering**, exactly what a cross-encoder fixes. We
rerank the top **120** (tuned): deeper adds noise, shallower loses recall. 100 %
recall is unreachable at any practical *k* — one gold page sits at fused rank
~12,500, and a few templated queries match many equivalent entities.
</details>

<details>
<summary><b>Why blend the reranker instead of trusting it outright?</b></summary>

The cross-encoder sharply improves clean single-answer factual queries (one jumps
0.63 → 1.00) but can *mis-rank* the templated multi-entity queries ("What links
X, Y, and Z?"), whose gold is a whole cluster of near-equivalent pages it scores
independently. Keeping a hybrid prior (**blend, don't replace**) is more robust
than pure rerank.
</details>

<details>
<summary><b>Why this small reranker and not a bigger one?</b></summary>

Larger general-purpose rerankers (BGE-reranker, MiniLM-L-12) scored **worse**
here — they overfit to natural-QA passages, while the small MS-MARCO L-6 transfers
best to this synthetic corpus. It is also light enough (~80 MB) to download, load,
and run within the query-time budget.
</details>

---

## 📊 Results — public 29 queries, mean NDCG@10

| Configuration | NDCG@10 |
|---|:---:|
| Dense only (whole page) | 0.329 |
| BM25 only (stemmed, `k1=2.0`, +decade +bigrams) | 0.422 |
| Hybrid · `0.7·dense + 0.3·BM25` | 0.447 |
| + cross-encoder reranker (blend 0.5) | 0.469 |
| **+ Optuna-tuned · blend 0.6, pool 120, `k1=2.0`, α=0.7 — final** | **0.485** |

Stemming the BM25 tokenizer raised recall@100 from 0.92 → 0.93; the Optuna pass
added a further ~0.016.

> **On tuning honestly.** The four knobs (`alpha`, BM25 `k1/b`, blend `weight`,
> `pool`) were tuned with an **Optuna** study (`dev/optuna_tune.py`, 600 trials
> over cached score matrices; cached search and live `eval_public` agreed to four
> decimals — 0.4855 vs 0.4854). On only 29 queries the optimum is *shallow* — a
> bootstrap CI of the tuned-vs-untuned gain straddles 0 — so we adopted the robust
> **region center** (higher neighborhood-mean NDCG, comparable worst case), not a
> cherry-picked spike. Optuna is a **dev-only** dependency, never imported at
> runtime. Reproduce via `dev/{recall,stem_sweep,rerank_blend,ablation,optuna_prep,optuna_tune}.py`.

---

## 📦 Artifacts (`artifacts/`, committed — required)

| File | Format | Contents |
|------|--------|----------|
| `page_vectors.npy` | float32 `(27074, 384)` | L2-normalized whole-page MiniLM embeddings |
| `page_ids.npy` | int64 `(27074,)` | `page_id` per vector row / BM25 column |
| `page_texts.npy` | object `(27074,)` | truncated page text fed to the reranker |
| `bm25_index.npz` | NumPy npz | BM25 inverted index (`term_indptr`, `doc_indices`, `weights`) |
| `bm25_index.meta.json` | JSON | BM25 vocabulary + parameters |
| `retrieval_config.json` | JSON | fusion `alpha`, model name, reranker settings |

All files are < 100 MB → plain Git is sufficient (no Git LFS). Row order is shared
across `page_vectors` / `page_ids` / `page_texts` / BM25 columns — keep them
aligned if you rebuild.

---

## 🗂️ Layout

The production code is an OOP package (`core/`) organized around single-responsibility
modules — one class per file — with abstractions (`PageScorer`, `Reranker`) the
pipeline depends on, so a new retrieval signal or reranker can be swapped in without
touching the orchestration. The three grading-contract files (`main.py`, `utils.py`,
`eval.py`) stay at the repo root.

```
main.py                 run(queries) entry point          ← autograder calls this
utils.py                paths, corpus / query loaders
eval.py                 NDCG@10                            ← read-only

core/                   production pipeline package
  interfaces.py         PageScorer / Reranker abstractions (depend on roles, not classes)
  embed.py              MiniLM embedding model
  reranker.py           cross-encoder reranker
  chunk.py              page → retrieval unit(s)
  lexical/              tokenizer.py · bm25.py · stemmer.py   (BM25 + Porter stemming)
  index/                config.py · loaded_index.py · builder.py · loader.py   (offline build/load)
  retrieve/             normalizer.py · dense.py · fusion.py · pipeline.py · service.py   (query time)

scripts/                build_index.py, eval_public.py
artifacts/              prebuilt index (committed)
dev/                    reproducible experiments behind the design (not used at runtime)
```

Each subpackage re-exports its public API through `__init__.py` (lazily, so importing
the light pieces — BM25, the stemmer, the score normalizer — doesn't pull in the
embedding/reranking stack).
