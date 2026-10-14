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
