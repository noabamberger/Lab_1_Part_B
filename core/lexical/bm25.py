"""BM25 lexical retriever (standard library + numpy only).

BM25 complements the dense MiniLM retriever: it rewards exact overlap on rare
tokens (specific numbers, coined names) that paraphrase-oriented embeddings blur.

Query-time speed: all term-document weights are precomputed offline into a
term-major (CSC-like) inverted index, so scoring a query is a handful of
postings-list lookups plus a scatter-add, independent of corpus size.
"""
from __future__ import annotations

import json
from pathlib import Path
from typing import Dict, List, Sequence

import numpy as np

from ..interfaces import PageScorer
from .tokenizer import Tokenize

# Cap tokens indexed per document: answer pages are short, while long distractor
# articles would otherwise add millions of useless postings.
MAX_DOC_TOKENS = 600

# k1/b tuned with Optuna over the public queries (dev/optuna_tune.py).
BM25_K1 = 2.0
BM25_B = 0.75


def _default_tokenize(text: str) -> List[str]:
    """Resolve the package-level ``tokenize`` at call time so dev sweeps that
    monkeypatch ``core.lexical.tokenize`` are honored (matches the old late
    binding of the module-global tokenizer)."""
    from core.lexical import tokenize

    return tokenize(text)


class BM25Index(PageScorer):
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
        tokenizer: Tokenize | None = None,
    ) -> None:
        self.vocab = vocab
        self.term_indptr = term_indptr
        self.doc_indices = doc_indices
        self.weights = weights
        self.num_docs = num_docs
        self.k1 = k1
        self.b = b
        self._tokenize: Tokenize = tokenizer or _default_tokenize

    # --- build -----------------------------------------------------------------
    @classmethod
    def build(
        cls,
        docs: List[str],
        *,
        k1: float = BM25_K1,
        b: float = BM25_B,
        max_doc_tokens: int = MAX_DOC_TOKENS,
        tokenizer: Tokenize | None = None,
    ) -> "BM25Index":
        tok: Tokenize = tokenizer or _default_tokenize
        vocab: Dict[str, int] = {}
        doc_len = np.zeros(len(docs), dtype=np.float32)

        # First pass: vocabulary, per-doc term frequencies, doc lengths.
        rows: List[int] = []  # doc index per posting
        cols: List[int] = []  # term id per posting
        tfs: List[int] = []   # term frequency per posting
        for i, text in enumerate(docs):
            tokens = tok(text)[:max_doc_tokens]
            doc_len[i] = len(tokens)
            counts: Dict[int, int] = {}
            for token in tokens:
                tid = vocab.get(token)
                if tid is None:
                    tid = len(vocab)
                    vocab[token] = tid
                counts[tid] = counts.get(tid, 0) + 1
            for tid, count in counts.items():
                rows.append(i)
                cols.append(tid)
                tfs.append(count)

        n_docs = len(docs)
        n_terms = len(vocab)
        row_arr = np.asarray(rows, dtype=np.int32)
        col_arr = np.asarray(cols, dtype=np.int32)
        tf_arr = np.asarray(tfs, dtype=np.float32)

        df = np.bincount(col_arr, minlength=n_terms).astype(np.float32)
        idf = np.log((n_docs - df + 0.5) / (df + 0.5) + 1.0).astype(np.float32)
        idf = np.maximum(idf, 1e-6)

        avgdl = float(doc_len.mean()) if n_docs else 1.0
        denom = k1 * (1.0 - b + b * doc_len / (avgdl or 1.0))  # per-doc
        weights = idf[col_arr] * (tf_arr * (k1 + 1.0)) / (tf_arr + denom[row_arr])

        # Sort postings by term id to form the term-major (CSC) inverted index.
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
            tokenizer=tokenizer,
        )

    # --- score -----------------------------------------------------------------
    def score(self, queries: Sequence[str]) -> np.ndarray:
        """Return a dense ``(n_queries, n_docs)`` BM25 score matrix."""
        scores = np.zeros((len(queries), self.num_docs), dtype=np.float32)
        for qi, query in enumerate(queries):
            row = scores[qi]
            seen: set[int] = set()
            for token in self._tokenize(query):
                tid = self.vocab.get(token)
                if tid is None or tid in seen:
                    continue
                seen.add(tid)
                start, end = self.term_indptr[tid], self.term_indptr[tid + 1]
                if start == end:
                    continue
                np.add.at(row, self.doc_indices[start:end], self.weights[start:end])
        return scores

    # Back-compat alias for the previous public method name.
    score_batch = score

    # --- persist ---------------------------------------------------------------
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
        path.with_suffix(".meta.json").write_text(json.dumps(meta), encoding="utf-8")

    @classmethod
    def load(cls, path: Path, *, tokenizer: Tokenize | None = None) -> "BM25Index":
        data = np.load(path)
        meta = json.loads(path.with_suffix(".meta.json").read_text(encoding="utf-8"))
        return cls(
            vocab=meta["vocab"],
            term_indptr=data["term_indptr"],
            doc_indices=data["doc_indices"],
            weights=data["weights"],
            num_docs=int(meta["num_docs"]),
            k1=meta["k1"],
            b=meta["b"],
            tokenizer=tokenizer,
        )
