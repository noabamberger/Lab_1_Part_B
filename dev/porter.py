"""Compact, dependency-free Porter stemmer (stdlib only).

Classic Porter (1980) algorithm. Used to test whether stemming the BM25
tokenizer improves lexical recall on the Section B corpus.
"""
from __future__ import annotations

VOWELS = "aeiou"


def _is_consonant(w: str, i: int) -> bool:
    c = w[i]
    if c in VOWELS:
        return False
    if c == "y":
        return i == 0 or not _is_consonant(w, i - 1)
    return True


def _measure(stem: str) -> int:
    n = 0
    prev_v = False
    for i in range(len(stem)):
        v = not _is_consonant(stem, i)
        if prev_v and not v:
            n += 1
        prev_v = v
    return n


def _contains_vowel(stem: str) -> bool:
    return any(not _is_consonant(stem, i) for i in range(len(stem)))


def _ends_double_consonant(w: str) -> bool:
    return len(w) >= 2 and w[-1] == w[-2] and _is_consonant(w, len(w) - 1)


def _cvc(w: str) -> bool:
    if len(w) < 3:
        return False
    if not (_is_consonant(w, len(w)-3) and not _is_consonant(w, len(w)-2)
            and _is_consonant(w, len(w)-1)):
        return False
    return w[-1] not in "wxy"


def stem(w: str) -> str:
    if len(w) <= 2:
        return w
    # Step 1a
    if w.endswith("sses"):
        w = w[:-2]
    elif w.endswith("ies"):
        w = w[:-2]
    elif w.endswith("ss"):
        pass
    elif w.endswith("s"):
        w = w[:-1]
    # Step 1b
    step1b_extra = False
    if w.endswith("eed"):
        if _measure(w[:-3]) > 0:
            w = w[:-1]
    elif w.endswith("ed"):
        if _contains_vowel(w[:-2]):
            w = w[:-2]
            step1b_extra = True
    elif w.endswith("ing"):
        if _contains_vowel(w[:-3]):
            w = w[:-3]
            step1b_extra = True
    if step1b_extra:
        if w.endswith(("at", "bl", "iz")):
            w += "e"
        elif _ends_double_consonant(w) and not w.endswith(("l", "s", "z")):
            w = w[:-1]
        elif _measure(w) == 1 and _cvc(w):
            w += "e"
    # Step 1c
    if w.endswith("y") and _contains_vowel(w[:-1]):
        w = w[:-1] + "i"
    # Step 2
    s2 = {
        "ational": "ate", "tional": "tion", "enci": "ence", "anci": "ance",
        "izer": "ize", "bli": "ble", "alli": "al", "entli": "ent",
        "eli": "e", "ousli": "ous", "ization": "ize", "ation": "ate",
        "ator": "ate", "alism": "al", "iveness": "ive", "fulness": "ful",
        "ousness": "ous", "aliti": "al", "iviti": "ive", "biliti": "ble",
        "logi": "log",
    }
    for suf, rep in s2.items():
        if w.endswith(suf):
            if _measure(w[:-len(suf)]) > 0:
                w = w[:-len(suf)] + rep
            break
    # Step 3
    s3 = {"icate": "ic", "ative": "", "alize": "al", "iciti": "ic",
          "ical": "ic", "ful": "", "ness": ""}
    for suf, rep in s3.items():
        if w.endswith(suf):
            if _measure(w[:-len(suf)]) > 0:
                w = w[:-len(suf)] + rep
            break
    # Step 4
    s4 = ["al", "ance", "ence", "er", "ic", "able", "ible", "ant", "ement",
          "ment", "ent", "ou", "ism", "ate", "iti", "ous", "ive", "ize"]
    for suf in s4:
        if w.endswith(suf):
            stem_ = w[:-len(suf)]
            if suf == "ion":
                continue
            if _measure(stem_) > 1:
                w = stem_
            break
    else:
        if w.endswith("ion") and _measure(w[:-3]) > 1 and w[-4] in "st":
            w = w[:-3]
    # Step 5a
    if w.endswith("e"):
        stem_ = w[:-1]
        if _measure(stem_) > 1 or (_measure(stem_) == 1 and not _cvc(stem_)):
            w = stem_
    # Step 5b
    if _measure(w) > 1 and _ends_double_consonant(w) and w.endswith("l"):
        w = w[:-1]
    return w
