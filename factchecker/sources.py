"""Evidence sources: where the passages Jev judges come from.

Jev decides whether evidence supports a claim; it is not a knowledge base. The
quality of the verdict is bounded by the quality of the evidence retrieved here,
which is why several sources can be combined and each website carries a learned
reliability weight (see reliability.py).
"""

from __future__ import annotations

import asyncio
import dataclasses
import hashlib
import html
import json
import logging
import os
import random
import re
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Protocol

import httpx

from .text import bm25_scores, chunk_document, entity_phrases, keyword_query, sentence_windows

log = logging.getLogger(__name__)


@dataclass(frozen=True)
class Candidate:
    """A document that might hold evidence, before its full text is fetched."""

    id: str
    title: str
    snippet: str
    url: str | None = None
    source: str = ""  # name of the EvidenceSource that found it


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


class _JsonHttp:
    """JSON requests with a request-rate limit, bounded concurrency, retries and caching.

    Public APIs throttle bursts; this stays polite, retries 429/5xx with backoff, and
    gives up rather than stall a fact check behind a long Retry-After.
    """

    def __init__(
        self,
        client: httpx.AsyncClient,
        *,
        requests_per_second: float,
        burst: int,
        max_concurrency: int,
        max_retries: int,
        max_retry_wait: float,
        cache_dir: str | Path | None,
        cache_ttl: float,
    ) -> None:
        self.client = client
        self._rate = _TokenBucket(requests_per_second, burst)
        self._limit = asyncio.Semaphore(max_concurrency)
        self._max_retries = max_retries
        self._max_retry_wait = max_retry_wait
        self._memory: dict[str, Any] = {}
        self._disk = _DiskCache(cache_dir, cache_ttl) if cache_dir else None

    async def request(self, cache_key: str, method: str, url: str, **kwargs: Any) -> Any:
        if cache_key in self._memory:
            return self._memory[cache_key]
        data = self._disk.get(cache_key) if self._disk else None
        if data is None:
            data = await self._send(method, url, **kwargs)
            if self._disk:
                self._disk.set(cache_key, data)
        self._memory[cache_key] = data
        return data

    async def _send(self, method: str, url: str, **kwargs: Any) -> Any:
        for attempt in range(self._max_retries + 1):
            await self._rate.acquire()
            async with self._limit:
                response = await self.client.request(method, url, **kwargs)
            retryable = response.status_code == 429 or response.status_code >= 500
            delay = _retry_delay(response, attempt) if retryable else 0.0
            if not retryable or attempt == self._max_retries or delay > self._max_retry_wait:
                response.raise_for_status()
                return response.json()
            await asyncio.sleep(delay)
        raise AssertionError("unreachable")

    async def aclose(self) -> None:
        await self.client.aclose()


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
        # Wikimedia requires a User-Agent with contact details and throttles anonymous
        # clients hard; set FACTCHECK_USER_AGENT to include yours.
        agent = user_agent or os.environ.get(
            "FACTCHECK_USER_AGENT", "jev-factchecker/0.1 (evidence retrieval for a fact-checking tool)"
        )
        self._http = _JsonHttp(
            client or httpx.AsyncClient(timeout=timeout, headers={"User-Agent": agent}, follow_redirects=True),
            # Bursts keep a single fact check fast; the sustained rate keeps bulk runs
            # (an eval over hundreds of claims) under Wikimedia's bot-traffic limits.
            requests_per_second=requests_per_second or float(os.environ.get("FACTCHECK_WIKI_RPS", "8")),
            burst=burst,
            max_concurrency=max_concurrency,
            max_retries=max_retries,
            max_retry_wait=max_retry_wait,
            cache_dir=cache_dir,
            cache_ttl=cache_ttl,
        )

    async def _get(self, **params: str | int) -> dict:
        params = {"action": "query", "format": "json", "formatversion": 2, **params}
        key = json.dumps([self.lang, sorted(params.items())])
        return await self._http.request(key, "GET", self._endpoint, params=params)

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
                        source=self.name,
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
            Candidate(
                id=title,
                title=title,
                snippet=pages[title].get("extract", "")[:300],
                url=self._url(title),
                source=self.name,
            )
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
        await self._http.aclose()


class WebSearchSource:
    """The open web, via the Tavily search API.

    Tavily returns the most query-relevant excerpts of each page in the search
    response itself, so no separate page download or HTML parsing is needed. With
    no TAVILY_API_KEY it uses Tavily's free, rate-limited keyless mode; a free key
    (1,000 searches/month) lifts the limit.
    """

    name = "web"
    endpoint = "https://api.tavily.com/search"

    def __init__(
        self,
        *,
        api_key: str | None = None,
        client: httpx.AsyncClient | None = None,
        search_depth: str = "basic",
        chunks_per_source: int = 3,
        exclude_domains: tuple[str, ...] = (),
        timeout: float = 15.0,
        requests_per_second: float | None = None,
        burst: int = 10,
        max_concurrency: int = 8,
        max_retries: int = 2,
        max_retry_wait: float = 4.0,
        cache_dir: str | Path | None = None,
        cache_ttl: float = 24 * 3600,  # the web changes faster than an encyclopedia
    ) -> None:
        api_key = api_key if api_key is not None else os.environ.get("TAVILY_API_KEY", "")
        headers = {"Authorization": f"Bearer {api_key}"} if api_key else {"X-Tavily-Access-Mode": "keyless"}
        self.keyless = not api_key
        self.search_depth = search_depth
        self.chunks_per_source = chunks_per_source
        self.exclude_domains = list(exclude_domains)
        self._http = _JsonHttp(
            client or httpx.AsyncClient(timeout=timeout, headers=headers),
            requests_per_second=requests_per_second or float(os.environ.get("FACTCHECK_WEB_RPS", "5")),
            burst=burst,
            max_concurrency=max_concurrency,
            max_retries=max_retries,
            max_retry_wait=max_retry_wait,
            cache_dir=cache_dir,
            cache_ttl=cache_ttl,
        )
        self._content: dict[str, str] = {}

    async def search(self, query: str, limit: int) -> list[Candidate]:
        body = {
            "query": query[:400],
            "max_results": limit,
            "search_depth": self.search_depth,
            "chunks_per_source": self.chunks_per_source,
            "include_answer": False,
        }
        if self.exclude_domains:
            body["exclude_domains"] = self.exclude_domains
        key = json.dumps(["tavily", sorted(body.items())])
        data = await self._http.request(key, "POST", self.endpoint, json=body)
        candidates = []
        for result in data.get("results", []):
            url, content = result.get("url"), result.get("content") or ""
            if not url or not content:
                continue
            self._content[url] = content
            title = result.get("title") or url
            candidates.append(Candidate(url, title, content[:300], url, self.name))
        return candidates

    async def passages(self, candidate: Candidate, max_words: int) -> list[Passage]:
        # Tavily separates the excerpts it picked from a page with "[...]".
        passages = []
        for chunk in self._content.get(candidate.id, "").split("[...]"):
            text = re.sub(r"\s+", " ", chunk).strip()
            if len(text.split()) < 6:
                continue
            windows = [text] if len(text.split()) <= max_words * 1.5 else sentence_windows(text, max_words)
            passages.extend(
                Passage(self.name, candidate.title, window, candidate.url, None, len(passages) + i)
                for i, window in enumerate(windows)
            )
        return passages

    async def aclose(self) -> None:
        await self._http.aclose()


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
            Candidate(name, name, self._documents[name][:300], str(self.directory / name), self.name)
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


class MultiSource:
    """Query several sources in parallel; one failing source doesn't fail the check."""

    name = "multi"

    def __init__(self, sources: list[EvidenceSource]) -> None:
        if not sources:
            raise ValueError("MultiSource needs at least one source.")
        self.sources = {source.name: source for source in sources}
        self._warned: set[str] = set()

    async def search(self, query: str, limit: int) -> list[Candidate]:
        candidates, _ = await self.search_with_failures(query, limit)
        return candidates

    async def search_with_failures(self, query: str, limit: int) -> tuple[list[Candidate], dict[str, str]]:
        """Candidates from every source that answered, and the error of each that didn't."""
        results = await asyncio.gather(
            *(source.search(query, limit) for source in self.sources.values()), return_exceptions=True
        )
        found: list[list[Candidate]] = []
        failures: dict[str, Exception] = {}
        for name, result in zip(self.sources, results):
            if isinstance(result, BaseException):
                if not isinstance(result, Exception):
                    raise result  # cancellation and other non-errors propagate
                if name not in self._warned:  # once per source, not once per claim
                    log.warning("Evidence source %r failed, continuing without it: %s", name, result)
                    self._warned.add(name)
                failures[name] = result
                continue
            found.append([c if c.source else dataclasses.replace(c, source=name) for c in result])
        if failures and not found:
            raise next(iter(failures.values()))
        return _round_robin(found), {name: _short_error(e) for name, e in failures.items()}

    async def passages(self, candidate: Candidate, max_words: int) -> list[Passage]:
        return await self.sources[candidate.source].passages(candidate, max_words)

    async def aclose(self) -> None:
        await asyncio.gather(*(source.aclose() for source in self.sources.values()))


def _short_error(error: Exception) -> str:
    if isinstance(error, httpx.HTTPStatusError):
        return f"HTTP {error.response.status_code}"
    return type(error).__name__


def _round_robin(groups: list[list[Candidate]]) -> list[Candidate]:
    merged = []
    for rank in range(max((len(g) for g in groups), default=0)):
        merged.extend(g[rank] for g in groups if rank < len(g))
    return merged


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

    def get(self, key: str) -> Any:
        path = self._path(key)
        try:
            if time.time() - path.stat().st_mtime > self.ttl:
                return None
            return json.loads(path.read_text())
        except (OSError, ValueError):
            return None

    def set(self, key: str, value: Any) -> None:
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
