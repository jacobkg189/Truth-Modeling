import json

import httpx
import pytest

from factchecker import sources
from factchecker.sources import LocalCorpusSource, WikipediaSource

EXTRACT = (
    "The Eiffel Tower is a wrought-iron lattice tower on the Champ de Mars in Paris, France.\n"
    "== History ==\n"
    "The tower was built from 1887 to 1889 as the entrance arch to the World's Fair.\n"
    "== References ==\n"
    "A citation that must not become evidence for anything at all.\n"
)


def wiki_handler(request: httpx.Request) -> httpx.Response:
    params = request.url.params
    if params.get("list") == "search":
        hits = [
            {"title": "Eiffel Tower", "snippet": 'The <span class="searchmatch">Eiffel</span> Tower &amp; Paris'},
            {"title": "Paris", "snippet": "Capital of France"},
        ]
        if params["srsearch"] == "Eiffel Tower built 1889":  # keyword variant adds a new title
            hits = [hits[0], {"title": "Gustave Eiffel", "snippet": "Engineer"}]
        return httpx.Response(200, json={"query": {"search": hits}})
    if params["prop"] == "extracts|pageprops":  # exact-title entity lookup
        known = {
            "Eiffel Tower": {"title": "Eiffel Tower", "extract": "A lattice tower in Paris."},
            "Gustave Eiffel": {"title": "Gustave Eiffel", "extract": "French engineer."},
            "Mercury": {"title": "Mercury", "extract": "Mercury may refer to:", "pageprops": {"disambiguation": ""}},
        }
        pages = [known.get(t, {"title": t, "missing": True}) for t in params["titles"].split("|")]
        return httpx.Response(200, json={"query": {"pages": pages}})
    assert params["prop"] == "extracts"
    return httpx.Response(
        200, json={"query": {"pages": [{"title": params["titles"], "extract": EXTRACT}]}}
    )


async def test_wikipedia_search_merges_queries_and_strips_html():
    source = WikipediaSource(client=httpx.AsyncClient(transport=httpx.MockTransport(wiki_handler)))
    candidates = await source.search("Was the Eiffel Tower built in 1889?", limit=6)
    assert [c.title for c in candidates] == ["Eiffel Tower", "Paris", "Gustave Eiffel"]
    assert candidates[0].snippet == "A lattice tower in Paris."  # from the exact-title lookup
    assert candidates[1].snippet == "Capital of France"
    assert candidates[0].url == "https://en.wikipedia.org/wiki/Eiffel_Tower"


async def test_wikipedia_passages_skip_references_and_keep_sections():
    source = WikipediaSource(client=httpx.AsyncClient(transport=httpx.MockTransport(wiki_handler)))
    [candidate] = (await source.search("Eiffel Tower", limit=1))
    passages = await source.passages(candidate, max_words=120)
    assert [(p.section, p.position) for p in passages] == [(None, 0), ("History", 1)]
    assert all("citation" not in p.text for p in passages)


async def test_wikipedia_caches_and_retries_rate_limits(monkeypatch):
    calls = {"n": 0}

    def flaky(request: httpx.Request) -> httpx.Response:
        calls["n"] += 1
        if calls["n"] == 1:
            return httpx.Response(429, headers={"retry-after": "0"})
        return wiki_handler(request)

    monkeypatch.setattr(sources, "_retry_delay", lambda response, attempt: 0)
    source = WikipediaSource(client=httpx.AsyncClient(transport=httpx.MockTransport(flaky)))
    candidate = sources.Candidate("Eiffel Tower", "Eiffel Tower", "")
    first = await source.passages(candidate, 120)
    second = await source.passages(candidate, 120)
    assert first == second and calls["n"] == 2  # one 429, one success, then cached


async def test_wikipedia_gives_up_after_max_retries(monkeypatch):
    monkeypatch.setattr(sources, "_retry_delay", lambda response, attempt: 0)
    source = WikipediaSource(
        client=httpx.AsyncClient(transport=httpx.MockTransport(lambda r: httpx.Response(503))),
        max_retries=2,
    )
    with pytest.raises(httpx.HTTPStatusError):
        await source.search("anything", limit=3)


async def test_local_corpus(tmp_path):
    (tmp_path / "policy.md").write_text(
        "Refunds are available within 30 days of purchase for every customer.\n"
        "Shipping is free on orders over fifty dollars within the country.\n"
    )
    (tmp_path / "other.txt").write_text("Unrelated notes about the office plants and their watering.\n")
    (tmp_path / "image.png").write_bytes(b"\x89PNG")
    source = LocalCorpusSource(tmp_path)
    candidates = await source.search("refund window days", limit=5)
    assert [c.title for c in candidates] == ["policy.md"]
    passages = await source.passages(candidates[0], 120)
    assert passages[0].text.startswith("Refunds are available")

    with pytest.raises(FileNotFoundError):
        LocalCorpusSource(tmp_path / "missing")


async def test_long_retry_after_fails_fast_instead_of_stalling():
    calls = {"n": 0}

    def throttled(request: httpx.Request) -> httpx.Response:
        calls["n"] += 1
        return httpx.Response(429, headers={"retry-after": "44"})

    source = WikipediaSource(client=httpx.AsyncClient(transport=httpx.MockTransport(throttled)))
    with pytest.raises(httpx.HTTPStatusError):
        await source.search("anything", limit=3)
    assert calls["n"] == 1  # gave up at once instead of sleeping 44 seconds


async def test_disk_cache_is_shared_across_instances(tmp_path):
    calls = {"n": 0}

    def counting(request: httpx.Request) -> httpx.Response:
        calls["n"] += 1
        return wiki_handler(request)

    def new_source() -> WikipediaSource:
        transport = httpx.MockTransport(counting)
        return WikipediaSource(client=httpx.AsyncClient(transport=transport), cache_dir=tmp_path)

    candidate = sources.Candidate("Eiffel Tower", "Eiffel Tower", "")
    first = await new_source().passages(candidate, 120)
    second = await new_source().passages(candidate, 120)
    assert first == second and calls["n"] == 1
    assert len(list(tmp_path.glob("*.json"))) == 1


async def test_entity_lookup_puts_named_pages_first_and_skips_disambiguation():
    source = WikipediaSource(client=httpx.AsyncClient(transport=httpx.MockTransport(wiki_handler)))
    candidates = await source.search("Gustave Eiffel named Mercury and Atlantis", limit=6)
    titles = [c.title for c in candidates]
    assert titles[0] == "Gustave Eiffel"
    assert "Mercury" not in titles and "Atlantis" not in titles


async def test_token_bucket_allows_burst_then_paces():
    import time

    bucket = sources._TokenBucket(rate=50, burst=5)
    started = time.monotonic()
    for _ in range(5):
        await bucket.acquire()
    assert time.monotonic() - started < 0.05  # the burst is immediate
    for _ in range(5):
        await bucket.acquire()
    assert time.monotonic() - started >= 0.09  # then 5 more at 50/s take ~0.1s


TAVILY_RESPONSE = {
    "results": [
        {
            "url": "https://www.nobelprize.org/curie",
            "title": "Marie Curie – Biographical",
            "content": "Marie Curie was born in Warsaw on November 7, 1867, the daughter of a teacher. "
            "[...] She went to Paris in 1891 to continue her studies at the Sorbonne in physics. [...] ok",
            "score": 0.8,
        },
        {"url": "https://empty.example/", "title": "Empty", "content": ""},
    ]
}


async def test_web_search_keyless_parses_excerpts_into_passages():
    seen = {}

    def handler(request: httpx.Request) -> httpx.Response:
        seen["headers"] = request.headers
        seen["body"] = json.loads(request.content)
        return httpx.Response(200, json=TAVILY_RESPONSE)

    source = sources.WebSearchSource(
        api_key="", client=httpx.AsyncClient(transport=httpx.MockTransport(handler), headers={"X-Tavily-Access-Mode": "keyless"})
    )
    assert source.keyless
    candidates = await source.search("Marie Curie was born in Germany", limit=5)
    assert seen["body"]["max_results"] == 5 and seen["body"]["include_answer"] is False
    assert [c.url for c in candidates] == ["https://www.nobelprize.org/curie"]  # empty result dropped
    assert candidates[0].source == "web"
    passages = await source.passages(candidates[0], 120)
    assert [p.text[:20] for p in passages] == ["Marie Curie was born", "She went to Paris in"]
    assert all(p.url == "https://www.nobelprize.org/curie" for p in passages)


def test_web_search_auth_headers(monkeypatch):
    monkeypatch.setenv("TAVILY_API_KEY", "tvly-test")
    keyed = sources.WebSearchSource()
    assert not keyed.keyless
    assert keyed._http.client.headers["Authorization"] == "Bearer tvly-test"
    monkeypatch.delenv("TAVILY_API_KEY")
    keyless = sources.WebSearchSource()
    assert keyless.keyless and keyless._http.client.headers["X-Tavily-Access-Mode"] == "keyless"


class StaticSource:
    def __init__(self, name, titles=(), error=None):
        self.name, self.titles, self.error = name, titles, error

    async def search(self, query, limit):
        if self.error:
            raise self.error
        return [sources.Candidate(t, t, "", f"https://{self.name}.example/{t}") for t in self.titles]

    async def passages(self, candidate, max_words):
        return [sources.Passage(self.name, candidate.title, f"text from {self.name}")]

    async def aclose(self):
        return None


async def test_multi_source_interleaves_tags_and_dispatches():
    multi = sources.MultiSource([StaticSource("a", ["a1", "a2"]), StaticSource("b", ["b1"])])
    candidates = await multi.search("q", 5)
    assert [(c.source, c.id) for c in candidates] == [("a", "a1"), ("b", "b1"), ("a", "a2")]
    passages = await multi.passages(candidates[1], 120)
    assert passages[0].text == "text from b"


async def test_multi_source_survives_one_failure_but_not_all():
    failing = StaticSource("web", error=httpx.ConnectError("down"))
    multi = sources.MultiSource([failing, StaticSource("wiki", ["w1"])])
    assert [c.id for c in await multi.search("q", 5)] == ["w1"]
    candidates, failures = await multi.search_with_failures("q", 5)
    assert failures == {"web": "ConnectError"}
    with pytest.raises(httpx.ConnectError):
        await sources.MultiSource([failing]).search("q", 5)
