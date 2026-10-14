"""Evidence sources: where the passages Jev judges come from.

Jev decides whether evidence supports a claim; it is not a knowledge base. The
quality of the verdict is bounded by the quality of the evidence retrieved here.
"""

from __future__ import annotations

import asyncio
import hashlib
import html
import json
import os
import random
import re
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Protocol

import httpx

from .text import bm25_scores, chunk_document, entity_phrases, keyword_query


@dataclass(frozen=True)
class Candidate:
    """A document that might hold evidence, before its full text is fetched."""

    id: str
    title: str
    snippet: str
    url: str | None = None


@dataclass(frozen=True)
class Passage:
    """A short span of a document, small enough for Jev to judge precisely."""

    source: str
    title: str
    text: str
    url: str | None = None
    section: str | None = None
    position: int = 0  # index of the passage within its document; 0 is the lead


class EvidenceSource(Protocol):
    name: str

    async def search(self, query: str, limit: int) -> list[Candidate]: ...

    async def passages(self, candidate: Candidate, max_words: int) -> list[Passage]: ...

    async def aclose(self) -> None: ...


class WikipediaSource:
    """English (or any language) Wikipedia via the public MediaWiki API."""

    name = "wikipedia"

    def __init__(
        self,
        lang: str = "en",
        *,
        client: httpx.AsyncClient | None = None,
        user_agent: str | None = None,
        timeout: float = 10.0,
        max_concurrency: int = 4,
        requests_per_second: float | None = None,
        burst: int = 30,
        max_retries: int = 3,
        max_retry_wait: float = 4.0,
        cache_dir: str | Path | None = None,
        cache_ttl: float = 7 * 24 * 3600,
    ) -> None:
        self.lang = lang
        self._endpoint = f"https://{lang}.wikipedia.org/w/api.php"
        # Wikimedia asks API clients for a descriptive User-Agent with contact details;
        # set FACTCHECK_USER_AGENT to include yours.
        agent = user_agent or os.environ.get(
            "FACTCHECK_USER_AGENT", "jev-factchecker/0.1 (evidence retrieval for a fact-checking tool)"
        )
        self._client = client or httpx.AsyncClient(
            timeout=timeout, headers={"User-Agent": agent}, follow_redirects=True
        )
        self._memory: dict[str, dict] = {}
        self._disk = _DiskCache(cache_dir, cache_ttl) if cache_dir else None
        # Wikimedia throttles bursts from one client; stay polite and retry on 429/5xx,
        # but give up rather than stall a fact check behind a long Retry-After.
        self._limit = asyncio.Semaphore(max_concurrency)
        # Bursts keep a single fact check fast; the sustained rate keeps bulk runs (like an
        # eval over hundreds of claims) under Wikimedia's bot-traffic limits.
        rate = requests_per_second or float(os.environ.get("FACTCHECK_WIKI_RPS", "8"))
        self._rate = _TokenBucket(rate, burst)
        self._max_retries = max_retries
        self._max_retry_wait = max_retry_wait

    async def _get(self, **params: str | int) -> dict:
        params = {"action": "query", "format": "json", "formatversion": 2, **params}
        key = json.dumps([self.lang, sorted(params.items())])
        if key in self._memory:
            return self._memory[key]
        data = self._disk.get(key) if self._disk else None
        if data is None:
            data = await self._fetch(params)
            if self._disk:
                self._disk.set(key, data)
        self._memory[key] = data
        return data

    async def _fetch(self, params: dict) -> dict:
        for attempt in range(self._max_retries + 1):
            await self._rate.acquire()
            async with self._limit:
                response = await self._client.get(self._endpoint, params=params)
            retryable = response.status_code == 429 or response.status_code >= 500
            delay = _retry_delay(response, attempt) if retryable else 0.0
            if not retryable or attempt == self._max_retries or delay > self._max_retry_wait:
                response.raise_for_status()
                return response.json()
            await asyncio.sleep(delay)
        raise AssertionError("unreachable")

    async def search(self, query: str, limit: int) -> list[Candidate]:
        # Search with the claim as written and with its content words: CirrusSearch
        # ranks them differently, and the union recovers more of the right articles.
        # Named entities are also looked up by exact title, since their own article is
        # usually the best evidence and full-text search does not always rank it first.
        queries = list(dict.fromkeys([query, keyword_query(query)]))
        entities, *results = await asyncio.gather(
            self._entity_pages(entity_phrases(query)),
            *(self._get(list="search", srsearch=q, srlimit=limit, srprop="snippet") for q in queries if q),
        )
        seen: dict[str, Candidate] = {c.title: c for c in entities}
        for rank in range(limit):
            for result in results:
                hits = result.get("query", {}).get("search", [])
                if rank < len(hits) and hits[rank]["title"] not in seen:
                    title = hits[rank]["title"]
                    seen[title] = Candidate(
                        id=title,
                        title=title,
                        snippet=_strip_html(hits[rank].get("snippet", "")),
                        url=self._url(title),
                    )
        return list(seen.values())[: limit + len(entities)]

    async def _entity_pages(self, names: list[str]) -> list[Candidate]:
        if not names:
            return []
        data = await self._get(
            prop="extracts|pageprops",
            ppprop="disambiguation",
            exintro=1,
            explaintext=1,
            exsentences=2,
            exlimit=20,
            redirects=1,
            titles="|".join(names),
        )
        pages = {
            page["title"]: page
            for page in data.get("query", {}).get("pages", [])
            if not page.get("missing") and "disambiguation" not in page.get("pageprops", {})
        }
        # Report pages in the order the names were found (longest, most specific first).
        resolved = {name: name for name in names}
        for mapping in data.get("query", {}).get("normalized", []) + data.get("query", {}).get("redirects", []):
            for name, target in resolved.items():
                if target == mapping["from"]:
                    resolved[name] = mapping["to"]
        ordered = dict.fromkeys(resolved[name] for name in names if resolved[name] in pages)
        return [
            Candidate(id=title, title=title, snippet=pages[title].get("extract", "")[:300], url=self._url(title))
            for title in ordered
        ]

    async def passages(self, candidate: Candidate, max_words: int) -> list[Passage]:
        data = await self._get(prop="extracts", explaintext=1, redirects=1, titles=candidate.title)
        pages = data.get("query", {}).get("pages", [])
        if not pages or "extract" not in pages[0]:
            return []
        page = pages[0]
        title = page.get("title", candidate.title)
        return [
            Passage(self.name, title, text, self._url(title), section, i)
            for i, (section, text) in enumerate(chunk_document(page["extract"], max_words))
        ]

    def _url(self, title: str) -> str:
        return f"https://{self.lang}.wikipedia.org/wiki/{title.replace(' ', '_')}"

    async def aclose(self) -> None:
        await self._client.aclose()


class LocalCorpusSource:
    """A directory of .txt / .md files you trust, such as policies, reports or notes."""

    name = "local"

    def __init__(self, directory: str | Path, *, extensions: tuple[str, ...] = (".txt", ".md")) -> None:
        self.directory = Path(directory)
        if not self.directory.is_dir():
            raise FileNotFoundError(f"Corpus directory not found: {self.directory}")
        self._documents = {
            str(path.relative_to(self.directory)): path.read_text(errors="replace")
            for path in sorted(self.directory.rglob("*"))
            if path.is_file() and path.suffix.lower() in extensions
        }

    async def search(self, query: str, limit: int) -> list[Candidate]:
        names = list(self._documents)
        scores = bm25_scores(query, [f"{n} {self._documents[n]}" for n in names])
        ranked = sorted(zip(scores, names), reverse=True)
        return [
            Candidate(id=name, title=name, snippet=self._documents[name][:300], url=str(self.directory / name))
            for score, name in ranked[:limit]
            if score > 0
        ]

    async def passages(self, candidate: Candidate, max_words: int) -> list[Passage]:
        text = self._documents.get(candidate.id, "")
        return [
            Passage(self.name, candidate.title, chunk, candidate.url, section, i)
            for i, (section, chunk) in enumerate(chunk_document(text, max_words))
        ]

    async def aclose(self) -> None:
        return None


class _TokenBucket:
    """Allow up to ``burst`` requests at once, refilling at ``rate`` per second."""

    def __init__(self, rate: float, burst: int) -> None:
        self.rate = rate
        self.capacity = float(burst)
        self.tokens = float(burst)
        self.updated = time.monotonic()
        self._lock = asyncio.Lock()

    async def acquire(self) -> None:
        async with self._lock:
            while True:
                now = time.monotonic()
                self.tokens = min(self.capacity, self.tokens + (now - self.updated) * self.rate)
                self.updated = now
                if self.tokens >= 1:
                    self.tokens -= 1
                    return
                await asyncio.sleep((1 - self.tokens) / self.rate)


class _DiskCache:
    """One JSON file per request, written atomically, so parallel runs can share it."""

    def __init__(self, directory: str | Path, ttl: float) -> None:
        self.directory = Path(directory).expanduser()
        self.directory.mkdir(parents=True, exist_ok=True)
        self.ttl = ttl

    def _path(self, key: str) -> Path:
        return self.directory / f"{hashlib.sha256(key.encode()).hexdigest()}.json"

    def get(self, key: str) -> dict | None:
        path = self._path(key)
        try:
            if time.time() - path.stat().st_mtime > self.ttl:
                return None
            return json.loads(path.read_text())
        except (OSError, ValueError):
            return None

    def set(self, key: str, value: dict) -> None:
        path = self._path(key)
        temporary = path.with_suffix(f".{os.getpid()}.tmp")
        try:
            temporary.write_text(json.dumps(value))
            temporary.replace(path)
        except OSError:
            temporary.unlink(missing_ok=True)


def default_cache_dir() -> Path:
    return Path(os.environ.get("FACTCHECK_CACHE_DIR", "~/.cache/jev-factchecker")).expanduser()


def _retry_delay(response: httpx.Response, attempt: int) -> float:
    """Honor Retry-After when the server sends seconds; otherwise back off exponentially."""
    header = response.headers.get("retry-after", "")
    if header.isdigit():
        return min(float(header), 30.0)
    return min(0.5 * 2**attempt, 8.0) + random.uniform(0, 0.25)


def _strip_html(text: str) -> str:
    return html.unescape(re.sub(r"<[^>]+>", "", text))
