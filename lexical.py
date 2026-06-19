"""Lexical BM25 retrieval (standard library + numpy only).

Complements the dense MiniLM retriever: BM25 rewards exact overlap on rare
tokens (specific numbers, coined names) that paraphrase-oriented sentence
embeddings can blur. The corpus answer pages are short synthetic entries that
restate the queried fact almost verbatim, so a sparse lexical signal is a strong
companion to the dense one.

Design for query-time speed: all BM25 term-document weights are precomputed
offline and stored in a term-major (CSC-like) inverted index. Scoring a query is
then a handful of postings-list lookups (one per query term) plus a scatter-add,
independent of corpus size beyond the touched postings.

Artifacts:
  bm25_index.npz        term_indptr, doc_indices, weights (inverted index)
  bm25_index.meta.json  vocabulary, BM25 params
"""
from __future__ import annotations

import json
import re
from pathlib import Path
from typing import Dict, List

import numpy as np

from stemmer import stem as _porter_stem

# Keep alphabetic words and numbers; preserve digit-group separators so a token
# like "1,456,779" stays intact and matches between query and document.
_TOKEN_RE = re.compile(r"[a-z]+|\d[\d,\.]*\d|\d")
# A 4-digit year (1000-2099). Queries often name a decade ("the 1820s") while
# the matching page names an exact year ("1826"); emitting a shared decade token
# ("182x") for both lets BM25 bridge that gap.
_YEAR_RE = re.compile(r"(1\d{3}|20\d{2})")

# Cap tokens indexed per document. Relevant (answer) pages are short (<=~600
# words); long distractor articles add millions of postings with no upside, so
# we bound the lexical footprint while still covering every answer in full.
MAX_DOC_TOKENS = 600

# k1/b tuned with Optuna over the public queries (dev/optuna_tune.py); k1=2.0 is
# independently corroborated by the earlier standalone k1/b sweep.
BM25_K1 = 2.0
BM25_B = 0.75


ADD_BIGRAMS = True
STEM = True

# Stemming is called on every token of every document at build time and every
# query token at search time; cache results since the vocabulary is small.
_stem_cache: Dict[str, str] = {}


def _stem(word: str) -> str:
    s = _stem_cache.get(word)
    if s is None:
        s = _porter_stem(word)
        _stem_cache[word] = s
    return s


def tokenize(text: str) -> List[str]:
    """Lowercase tokenizer; keeps numeric tokens intact and adds decade tokens.

    Alphabetic words are Porter-stemmed so a query word matches its morphological
    variants ("negotiator"/"negotiations" -> "negoti"). A 4-digit year token also
    yields a decade token, e.g. "1826" -> "182x" and "1820" (from "1820s") ->
    "182x", so a decade query matches an exact-year page. Adjacent (stemmed) word
    bigrams are appended so discriminative phrases ("point guard", "cold-water
    fisheries") match as units, not just as independent words.
    """
    raw = _TOKEN_RE.findall(text.lower())
    tokens: List[str] = []
    words: List[str] = []  # stemmed alphabetic words, for bigrams
    for t in raw:
        if t.isalpha():
            s = _stem(t) if STEM else t
            tokens.append(s)
            words.append(s)
        else:
            tokens.append(t)
    extra = [f"{t[:3]}x" for t in raw if _YEAR_RE.fullmatch(t)]
    if ADD_BIGRAMS:
        extra += [f"{a}_{b}" for a, b in zip(words, words[1:])]
    return tokens + extra


class BM25Index:
    """BM25 with precomputed term-document weights in a term-major layout."""

    def __init__(
        self,
        *,
        vocab: Dict[str, int],
        term_indptr: np.ndarray,
        doc_indices: np.ndarray,
        weights: np.ndarray,
        num_docs: int,
        k1: float = BM25_K1,
        b: float = BM25_B,
    ) -> None:
        self.vocab = vocab
        self.term_indptr = term_indptr
        self.doc_indices = doc_indices
        self.weights = weights
        self.num_docs = num_docs
        self.k1 = k1
        self.b = b

    # ----------------------------------------------------------------- build
    @classmethod
    def build(
        cls,
        docs: List[str],
        *,
        k1: float = BM25_K1,
        b: float = BM25_B,
        max_doc_tokens: int = MAX_DOC_TOKENS,
    ) -> "BM25Index":
        vocab: Dict[str, int] = {}
        doc_len = np.zeros(len(docs), dtype=np.float32)

        # First pass: build vocabulary, per-doc term frequencies, doc lengths.
        rows: List[int] = []   # doc index per posting
        cols: List[int] = []   # term id per posting
        tfs: List[int] = []    # term frequency per posting
        for i, text in enumerate(docs):
            tokens = tokenize(text)[:max_doc_tokens]
            doc_len[i] = len(tokens)
            counts: Dict[int, int] = {}
            for tok in tokens:
                tid = vocab.get(tok)
                if tid is None:
                    tid = len(vocab)
                    vocab[tok] = tid
                counts[tid] = counts.get(tid, 0) + 1
            for tid, c in counts.items():
                rows.append(i)
                cols.append(tid)
                tfs.append(c)

        n_docs = len(docs)
        n_terms = len(vocab)
        row_arr = np.asarray(rows, dtype=np.int32)
        col_arr = np.asarray(cols, dtype=np.int32)
        tf_arr = np.asarray(tfs, dtype=np.float32)

        # Document frequency per term -> idf.
        df = np.bincount(col_arr, minlength=n_terms).astype(np.float32)
        idf = np.log((n_docs - df + 0.5) / (df + 0.5) + 1.0).astype(np.float32)
        idf = np.maximum(idf, 1e-6)

        avgdl = float(doc_len.mean()) if n_docs else 1.0
        denom = k1 * (1.0 - b + b * doc_len / (avgdl or 1.0))  # per-doc

        # Precompute BM25 weight for every posting.
        weights = idf[col_arr] * (tf_arr * (k1 + 1.0)) / (tf_arr + denom[row_arr])

        # Sort postings by term id to form a term-major (CSC) inverted index.
        order = np.argsort(col_arr, kind="stable")
        col_sorted = col_arr[order]
        doc_indices = row_arr[order].astype(np.int32)
        weights_sorted = weights[order].astype(np.float32)
        term_indptr = np.zeros(n_terms + 1, dtype=np.int64)
        np.add.at(term_indptr, col_sorted + 1, 1)
        np.cumsum(term_indptr, out=term_indptr)

        return cls(
            vocab=vocab,
            term_indptr=term_indptr,
            doc_indices=doc_indices,
            weights=weights_sorted,
            num_docs=n_docs,
            k1=k1,
            b=b,
        )

    # ------------------------------------------------------------------ score
    def score_batch(self, queries: List[str]) -> np.ndarray:
        """Return a dense (num_queries, num_docs) BM25 score matrix."""
        scores = np.zeros((len(queries), self.num_docs), dtype=np.float32)
        for qi, q in enumerate(queries):
            row = scores[qi]
            seen: set[int] = set()
            for tok in tokenize(q):
                tid = self.vocab.get(tok)
                if tid is None or tid in seen:
                    continue
                seen.add(tid)
                s, e = self.term_indptr[tid], self.term_indptr[tid + 1]
                if s == e:
                    continue
                np.add.at(row, self.doc_indices[s:e], self.weights[s:e])
        return scores

    # ----------------------------------------------------------------- persist
    def save(self, path: Path) -> None:
        np.savez_compressed(
            path,
            term_indptr=self.term_indptr,
            doc_indices=self.doc_indices,
            weights=self.weights,
        )
        meta = {
            "vocab": self.vocab,
            "num_docs": self.num_docs,
            "k1": self.k1,
            "b": self.b,
        }
        path.with_suffix(".meta.json").write_text(
            json.dumps(meta), encoding="utf-8"
        )

    @classmethod
    def load(cls, path: Path) -> "BM25Index":
        data = np.load(path)
        meta = json.loads(
            path.with_suffix(".meta.json").read_text(encoding="utf-8")
        )
        return cls(
            vocab=meta["vocab"],
            term_indptr=data["term_indptr"],
            doc_indices=data["doc_indices"],
            weights=data["weights"],
            num_docs=int(meta["num_docs"]),
            k1=meta["k1"],
            b=meta["b"],
        )
