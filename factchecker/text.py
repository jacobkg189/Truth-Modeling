"""Text utilities that belong in code rather than in the model.

Jev's guidance is explicit: splitting, counting, ranking and number comparison
are code's job. Jev is reserved for the judgments.
"""

from __future__ import annotations

import math
import re
from collections import Counter

_SENTENCE_BOUNDARY = re.compile(
    r"(?:(?<=[.!?])|(?<=[.!?][\"”’')\]]))\s+(?=[A-Z0-9\"“‘'(\[])"
)
_ABBREVIATIONS = {
    "mr", "mrs", "ms", "dr", "prof", "st", "jr", "sr", "vs", "etc", "inc", "ltd",
    "co", "corp", "no", "approx", "e.g", "i.e", "u.s", "u.k", "u.n", "a.m", "p.m",
    "jan", "feb", "mar", "apr", "jun", "jul", "aug", "sep", "sept", "oct", "nov",
    "dec", "mt", "ft", "gen", "gov", "sen", "rep", "rev", "lt", "col", "sgt",
}
_STOPWORDS = frozenset(
    "a an and are as at be been being but by did do does for from had has have he her "
    "his i if in into is it its of on or our she so than that the their them then there "
    "these they this those to was we were what when where which who whom why will with "
    "would you your not no".split()
)
_TOKEN = re.compile(r"[a-z0-9]+")
_NUMBER = re.compile(r"(?<![\w.])(\d{1,3}(?:,\d{3})+|\d+)(?:\.(\d+))?")
_SKIP_SECTIONS = {
    "see also", "references", "external links", "further reading", "notes",
    "bibliography", "sources", "citations", "footnotes", "works cited",
}
_HEADING = re.compile(r"^(=+)\s*(.*?)\s*\1$")


def split_sentences(text: str) -> list[str]:
    """Split prose into sentences, keeping common abbreviations and initials intact."""
    text = re.sub(r"\s+", " ", text).strip()
    if not text:
        return []
    sentences: list[str] = []
    for part in _SENTENCE_BOUNDARY.split(text):
        if sentences and _ends_with_abbreviation(sentences[-1]):
            sentences[-1] = f"{sentences[-1]} {part}"
        else:
            sentences.append(part)
    return [s.strip() for s in sentences if s.strip()]


def _ends_with_abbreviation(sentence: str) -> bool:
    last = sentence.rsplit(" ", 1)[-1].rstrip(".").lower()
    return last in _ABBREVIATIONS or (len(last) == 1 and last.isalpha())


def tokenize(text: str) -> list[str]:
    """Lowercase, stemmed content words for lexical ranking."""
    return [_stem(t) for t in _TOKEN.findall(text.lower()) if t not in _STOPWORDS]


def _stem(token: str) -> str:
    """A deliberately light suffix strip: 'headquartered' and 'headquarters' both match."""
    for suffix in ("ing", "ed"):
        if len(token) > len(suffix) + 3 and token.endswith(suffix):
            return token[: -len(suffix)]
    if len(token) > 3 and token.endswith("s") and not token.endswith("ss"):
        return token[:-1]
    return token


def bm25_scores(query: str, documents: list[str], k1: float = 1.5, b: float = 0.75) -> list[float]:
    """Okapi BM25 score of each document against the query."""
    if not documents:
        return []
    query_terms = set(tokenize(query))
    doc_tokens = [tokenize(d) for d in documents]
    n = len(documents)
    avg_len = (sum(len(t) for t in doc_tokens) / n) or 1.0
    doc_freq: Counter[str] = Counter()
    for tokens in doc_tokens:
        doc_freq.update(set(tokens))

    scores = []
    for tokens in doc_tokens:
        tf = Counter(tokens)
        score = 0.0
        for term in query_terms:
            if term not in tf:
                continue
            idf = math.log(1 + (n - doc_freq[term] + 0.5) / (doc_freq[term] + 0.5))
            norm = tf[term] + k1 * (1 - b + b * len(tokens) / avg_len)
            score += idf * tf[term] * (k1 + 1) / norm
        scores.append(score)
    return scores


def keyword_query(text: str, max_terms: int = 10) -> str:
    """The claim's content words in their original casing, for a search engine."""
    words = re.findall(r"[\w'-]+", text)
    kept = [w for w in words if w.lower() not in _STOPWORDS and len(w) > 1]
    return " ".join(kept[:max_terms])


_NAME_WORD = r"[A-Z][\w'’.-]*"
# Lowercase words that can sit inside a name: "Great Wall of China", "Cheese in the Trap".
_TIGHT_JOINER = r"(?:of|da|de|del|der|di|du|van|von|la|le|y)"
_LOOSE_JOINER = r"(?:the|and|in|on|at|with|for|to|a|an)"
_NAME_JOINER = rf"(?:{_TIGHT_JOINER}|{_LOOSE_JOINER})"
_NAME = re.compile(rf"{_NAME_WORD}(?:\s+(?:{_NAME_JOINER}\s+)*{_NAME_WORD})*(?:\s+\([^()]{{1,40}}\))?")
# Capitalized only because they start the sentence: "Was the Eiffel Tower ...", "In 1889 ...".
_NOT_NAME_START = _STOPWORDS | {
    "in", "on", "at", "by", "for", "from", "during", "after", "before", "since", "until",
    "according", "although", "because", "however", "yes", "also", "many", "some", "most",
    "every", "all", "both", "each", "only", "even", "just", "how", "did", "does", "can",
}


def entity_phrases(text: str, max_phrases: int = 6) -> list[str]:
    """Capitalized names in the text, for exact encyclopedia-title lookup.

    A full-text search for a whole claim often ranks pages that merely mention an
    entity above the entity's own page. Each name comes in its full form first
    ("Cheese in the Trap (TV series)", "Paris in France"), then the pieces it splits
    into at qualifiers and loose joiners ("Cheese in the Trap", "Paris", "France"), so
    a lookup finds the right page whether or not the joined form is a real title.
    """
    full: list[str] = []
    parts: list[str] = []
    for match in _NAME.finditer(text):
        phrase = _clean_name(match.group(0))
        if not phrase:
            continue
        full.append(phrase)
        bare = re.sub(r"\s*\([^()]*\)$", "", phrase)
        pieces = [bare] + re.split(rf"\s+{_LOOSE_JOINER}\s+", bare)
        parts.extend(_clean_name(piece) for piece in pieces)
    ordered = sorted(dict.fromkeys(full), key=len, reverse=True)
    ordered += sorted({p for p in parts if p and p not in ordered}, key=len, reverse=True)
    return ordered[:max_phrases]


def _clean_name(phrase: str) -> str:
    words = phrase.split()
    while words and (words[0].lower() in _NOT_NAME_START or re.fullmatch(_NAME_JOINER, words[0])):
        words.pop(0)
    while words and re.fullmatch(_NAME_JOINER, words[-1]):
        words.pop()
    name = re.sub(r"['’]s$", "", " ".join(words).rstrip("."))
    return "" if len(name) < 2 or name.lower() in _STOPWORDS else name


def extract_numbers(text: str) -> set[str]:
    """Numbers in the text, normalized so '1,083' and '1083' compare equal."""
    numbers = set()
    for whole, fraction in _NUMBER.findall(text):
        value = whole.replace(",", "")
        fraction = fraction.rstrip("0")
        numbers.add(f"{value}.{fraction}" if fraction else value)
    return numbers


def chunk_document(text: str, max_words: int = 120) -> list[tuple[str | None, str]]:
    """Split a plain-text document into (section, passage) pairs of about max_words.

    Recognizes MediaWiki-style ``== Heading ==`` lines, and stops at reference
    sections, which hold citations rather than prose.
    """
    passages: list[tuple[str | None, str]] = []
    section: str | None = None
    for raw in text.splitlines():
        line = raw.strip()
        if not line:
            continue
        heading = _HEADING.match(line)
        if heading:
            title = heading.group(2)
            if heading.group(1) == "==" and title.lower() in _SKIP_SECTIONS:
                break
            section = title
            continue
        words = line.split()
        if len(words) < 6:  # captions, list stubs and other fragments carry little evidence
            continue
        if len(words) <= max_words:
            passages.append((section, line))
            continue
        passages.extend((section, window) for window in sentence_windows(line, max_words))
    return passages


def sentence_windows(paragraph: str, max_words: int) -> list[str]:
    """Group sentences into windows of about max_words, overlapping by one sentence."""
    sentences = split_sentences(paragraph)
    windows: list[str] = []
    current: list[str] = []
    count = 0
    for sentence in sentences:
        length = len(sentence.split())
        if current and count + length > max_words:
            windows.append(" ".join(current))
            current = current[-1:]
            count = len(current[0].split())
        current.append(sentence)
        count += length
    if current and (not windows or len(current) > 1):
        windows.append(" ".join(current))
    return windows
