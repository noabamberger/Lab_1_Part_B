"""Dependency-free Porter stemmer (classic 1980 algorithm, stdlib only).

Stemming the BM25 tokenizer lets a query word match its morphological variants
in a page ("negotiator"/"negotiations" -> "negoti"), which raised both lexical
recall and the fused NDCG@10 on this corpus.
"""
from __future__ import annotations

from typing import Dict

_VOWELS = "aeiou"


class PorterStemmer:
    """Reduces a word to its Porter stem; memoizes over a small vocabulary."""

    def __init__(self) -> None:
        self._cache: Dict[str, str] = {}

    def stem(self, word: str) -> str:
        cached = self._cache.get(word)
        if cached is None:
            cached = self._stem(word)
            self._cache[word] = cached
        return cached

    # --- letter classification -------------------------------------------------
    @staticmethod
    def _is_consonant(word: str, i: int) -> bool:
        c = word[i]
        if c in _VOWELS:
            return False
        if c == "y":
            return i == 0 or not PorterStemmer._is_consonant(word, i - 1)
        return True

    @staticmethod
    def _measure(stem: str) -> int:
        n = 0
        prev_vowel = False
        for i in range(len(stem)):
            vowel = not PorterStemmer._is_consonant(stem, i)
            if prev_vowel and not vowel:
                n += 1
            prev_vowel = vowel
        return n

    @staticmethod
    def _contains_vowel(stem: str) -> bool:
        return any(not PorterStemmer._is_consonant(stem, i) for i in range(len(stem)))

    @staticmethod
    def _ends_double_consonant(word: str) -> bool:
        return (
            len(word) >= 2
            and word[-1] == word[-2]
            and PorterStemmer._is_consonant(word, len(word) - 1)
        )

    @staticmethod
    def _cvc(word: str) -> bool:
        """True if word ends consonant-vowel-consonant and the last isn't w/x/y."""
        if len(word) < 3:
            return False
        if not (
            PorterStemmer._is_consonant(word, len(word) - 3)
            and not PorterStemmer._is_consonant(word, len(word) - 2)
            and PorterStemmer._is_consonant(word, len(word) - 1)
        ):
            return False
        return word[-1] not in "wxy"

    # --- algorithm -------------------------------------------------------------
    _STEP2 = {
        "ational": "ate", "tional": "tion", "enci": "ence", "anci": "ance",
        "izer": "ize", "bli": "ble", "alli": "al", "entli": "ent",
        "eli": "e", "ousli": "ous", "ization": "ize", "ation": "ate",
        "ator": "ate", "alism": "al", "iveness": "ive", "fulness": "ful",
        "ousness": "ous", "aliti": "al", "iviti": "ive", "biliti": "ble",
        "logi": "log",
    }
    _STEP3 = {
        "icate": "ic", "ative": "", "alize": "al", "iciti": "ic",
        "ical": "ic", "ful": "", "ness": "",
    }
    _STEP4 = [
        "al", "ance", "ence", "er", "ic", "able", "ible", "ant", "ement",
        "ment", "ent", "ou", "ism", "ate", "iti", "ous", "ive", "ize",
    ]

    def _stem(self, word: str) -> str:
        if len(word) <= 2:
            return word
        w = word
        w = self._step1a(w)
        w = self._step1b(w)
        w = self._step1c(w)
        w = self._replace_suffix(w, self._STEP2)
        w = self._replace_suffix(w, self._STEP3)
        w = self._step4(w)
        w = self._step5(w)
        return w

    @staticmethod
    def _step1a(w: str) -> str:
        if w.endswith("sses"):
            return w[:-2]
        if w.endswith("ies"):
            return w[:-2]
        if w.endswith("ss"):
            return w
        if w.endswith("s"):
            return w[:-1]
        return w

    def _step1b(self, w: str) -> str:
        stripped = False
        if w.endswith("eed"):
            if self._measure(w[:-3]) > 0:
                w = w[:-1]
        elif w.endswith("ed") and self._contains_vowel(w[:-2]):
            w, stripped = w[:-2], True
        elif w.endswith("ing") and self._contains_vowel(w[:-3]):
            w, stripped = w[:-3], True
        if stripped:
            if w.endswith(("at", "bl", "iz")):
                w += "e"
            elif self._ends_double_consonant(w) and not w.endswith(("l", "s", "z")):
                w = w[:-1]
            elif self._measure(w) == 1 and self._cvc(w):
                w += "e"
        return w

    def _step1c(self, w: str) -> str:
        if w.endswith("y") and self._contains_vowel(w[:-1]):
            return w[:-1] + "i"
        return w

    def _replace_suffix(self, w: str, table: Dict[str, str]) -> str:
        for suffix, replacement in table.items():
            if w.endswith(suffix):
                if self._measure(w[: -len(suffix)]) > 0:
                    w = w[: -len(suffix)] + replacement
                break
        return w

    def _step4(self, w: str) -> str:
        for suffix in self._STEP4:
            if w.endswith(suffix):
                stem = w[: -len(suffix)]
                if suffix == "ion":  # never reached; kept for fidelity to source
                    continue
                if self._measure(stem) > 1:
                    w = stem
                break
        else:
            if w.endswith("ion") and self._measure(w[:-3]) > 1 and w[-4] in "st":
                w = w[:-3]
        return w

    def _step5(self, w: str) -> str:
        if w.endswith("e"):
            stem = w[:-1]
            if self._measure(stem) > 1 or (self._measure(stem) == 1 and not self._cvc(stem)):
                w = stem
        if self._measure(w) > 1 and self._ends_double_consonant(w) and w.endswith("l"):
            w = w[:-1]
        return w


# Module-level facade over a shared stemmer (back-compat for callers importing
# ``stem``); the cache persists across calls.
_DEFAULT_STEMMER = PorterStemmer()


def stem(word: str) -> str:
    return _DEFAULT_STEMMER.stem(word)
