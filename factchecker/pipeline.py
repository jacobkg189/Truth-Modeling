"""The fact-checking pipeline.

    text ──split──▶ sentences ──Jev: check-worthy?──▶ claims
    claim ──search Wikipedia + the web in parallel──▶ candidate documents
          ──Jev: is this document about what the claim names?──▶ documents (per source)
    documents ──chunk + BM25, one best passage per site first──▶ passages
          ──Jev: stance (×2 option orders)──▶ judgments
    judgments ──pool per site, weighted by site reliability──▶ verdict + review flag
          ──leave-one-out agreement──▶ updated site reliability

Claims are checked concurrently, and each claim's passages are judged
concurrently, so latency is a few sequential round trips regardless of length.
"""

from __future__ import annotations

import asyncio
import re
import time
from collections.abc import Iterable
from dataclasses import dataclass, field, replace

import httpx
from typesafe_sdk import (
    TypeSafeAPIConnectionError,
    TypeSafeAPIError,
    TypeSafeAuthenticationError,
    TypeSafePermissionDeniedError,
)

from .config import Config
from .jev import JevJudge, Stance, Usage
from .reliability import SourceReliability, default_reliability_path, site_of
from .sources import (
    Candidate,
    EvidenceSource,
    LocalCorpusSource,
    MultiSource,
    Passage,
    WebSearchSource,
    WikipediaSource,
    default_cache_dir,
)
from .text import bm25_scores, split_sentences
from .verdict import Judgment, Label, Verdict, aggregate, consensus_observations, site_stances

_LEADING_REFERENCE = re.compile(
    r"^(it|its|he|his|she|her|they|their|this|these|that|those|the company|the city|the country)\b",
    re.IGNORECASE,
)
SOURCE_NAMES = ("wikipedia", "web", "local")


def unresolved_reference(claim: str) -> str | None:
    """The leading "it" / "the company" a claim uses instead of naming its subject, if any.

    "The company Apple was founded in 1976" names its subject right after the reference,
    so it is not counted.
    """
    match = _LEADING_REFERENCE.match(claim.strip())
    if not match:
        return None
    rest = claim.strip()[match.end() :].lstrip(" ,")
    if rest[:1].isupper():
        return None
    return match.group(0)


@dataclass
class ClaimResult:
    claim: str
    verdict: Verdict
    evidence: list[Judgment]  # most decisive first
    documents: list[Candidate]
    seconds: float
    check_worthiness: float | None = None
    error: str | None = None


@dataclass
class Report:
    text: str
    claims: list[ClaimResult]
    skipped: list[tuple[str, float]] = field(default_factory=list)  # (sentence, check-worthiness)
    seconds: float = 0.0
    usage: Usage = field(default_factory=Usage)


class FactChecker:
    def __init__(
        self,
        judge: JevJudge,
        source: EvidenceSource,
        config: Config | None = None,
        *,
        reliability: SourceReliability | None = None,
        learn: bool = False,
    ) -> None:
        self.judge = judge
        self.source = source
        self.config = config or Config()
        self.reliability = reliability
        self.learn = learn and reliability is not None

    @classmethod
    def create(
        cls,
        *,
        sources: Iterable[str] = ("wikipedia", "web"),
        corpus_dir: str | None = None,
        lang: str = "en",
        model: str | None = None,
        config: Config | None = None,
        cache: bool = True,
        reliability_path: str | None = None,
        learn: bool = True,
        web_exclude_domains: tuple[str, ...] = (),
    ) -> "FactChecker":
        config = config or Config()
        cache_dir = default_cache_dir() if cache else None
        built: list[EvidenceSource] = []
        for name in dict.fromkeys(sources):
            if name == "wikipedia":
                built.append(WikipediaSource(lang, cache_dir=cache_dir))
            elif name == "web":
                built.append(
                    WebSearchSource(
                        exclude_domains=web_exclude_domains,
                        cache_dir=cache_dir / "web" if cache_dir else None,
                    )
                )
            elif name == "local":
                if not corpus_dir:
                    raise ValueError("The local source needs a corpus directory.")
                built.append(LocalCorpusSource(corpus_dir))
            else:
                raise ValueError(f"Unknown evidence source {name!r}; choose from {', '.join(SOURCE_NAMES)}.")
        evidence = built[0] if len(built) == 1 else MultiSource(built)
        reliability = SourceReliability(reliability_path or default_reliability_path())
        judge = JevJudge(model=model, max_concurrency=config.max_concurrency)
        return cls(judge, evidence, config, reliability=reliability, learn=learn)

    async def __aenter__(self) -> "FactChecker":
        return self

    async def __aexit__(self, *exc: object) -> None:
        await self.aclose()

    async def aclose(self) -> None:
        if self.reliability:
            self.reliability.save()
        await asyncio.gather(self.judge.aclose(), self.source.aclose())

    async def check_text(self, text: str, *, detect_claims: bool = True) -> Report:
        """Find the checkable claims in a text and check each one."""
        started = time.perf_counter()
        usage_before = replace(self.judge.usage, models=set(self.judge.usage.models))
        sentences = split_sentences(text)
        if detect_claims:
            scores = await self.judge.detect_claims(sentences)
        else:
            scores = [1.0] * len(sentences)

        tasks, skipped = [], []
        for i, (sentence, score) in enumerate(zip(sentences, scores)):
            if score < self.config.claim_threshold:
                skipped.append((sentence, score))
                continue
            # Two sentences back: the referent is often one sentence further than the last.
            context = " ".join(sentences[max(0, i - 2) : i]) or None
            tasks.append(self.check_claim(sentence, context=context, check_worthiness=score))
        claims = await asyncio.gather(*tasks)
        after = self.judge.usage
        usage = Usage(
            after.requests - usage_before.requests,
            after.input_tokens - usage_before.input_tokens,
            after.output_tokens - usage_before.output_tokens,
            set(after.models),
        )
        return Report(text, list(claims), skipped, time.perf_counter() - started, usage)

    async def check_claim(
        self, claim: str, *, context: str | None = None, check_worthiness: float | None = None
    ) -> ClaimResult:
        """Retrieve evidence for one claim and have Jev judge it."""
        started = time.perf_counter()
        # A claim that opens with a pronoun ("It was built in 1889") cannot be searched
        # on its own; borrow the preceding sentences for retrieval and for the judgment.
        reference = unresolved_reference(claim)
        if reference and not context:
            # Without earlier text there is no way to know who "it" or "the company" is, and
            # searching anyway "verifies" the claim against whichever subject turns up.
            verdict = Verdict(
                Label.NOT_ENOUGH_EVIDENCE,
                1.0,
                0.0,
                0.0,
                True,
                [f"The claim refers to its subject only as “{reference}”, so it can't be checked on its own."],
            )
            return ClaimResult(claim, verdict, [], [], time.perf_counter() - started, check_worthiness)
        needs_context = bool(context and reference)
        query = f"{context} {claim}" if needs_context else claim
        unavailable: dict[str, str] = {}
        try:
            documents = await self._find_documents(
                claim, query, context if needs_context else None, unavailable
            )
            # Context picks the documents; the claim alone picks passages within them,
            # otherwise the context's words outweigh what the claim actually asserts.
            passages = await self._top_passages(claim, documents)
            stances = await asyncio.gather(
                *(self.judge.stance(claim, p, context if needs_context else None) for p in passages)
            )
        except (TypeSafeAuthenticationError, TypeSafePermissionDeniedError):
            raise  # a configuration problem, not a property of this claim
        except (TypeSafeAPIError, TypeSafeAPIConnectionError, httpx.HTTPError) as error:
            verdict = Verdict(
                Label.NOT_ENOUGH_EVIDENCE, 0.0, 0.0, 0.0, True, [f"Check failed: {error}"]
            )
            return ClaimResult(
                claim, verdict, [], [], time.perf_counter() - started, check_worthiness, str(error)
            )

        judgments = [self._judgment(p, s) for p, s in zip(passages, stances)]
        verdict = aggregate(claim, judgments, self.config)
        for name, error in sorted(unavailable.items()):
            verdict.reasons.append(f"The {name} source was unavailable ({error}); checked without it.")
        if self.learn:
            for site, agreed, strength in consensus_observations(claim, judgments, self.config):
                self.reliability.record(site, agreed, strength)  # type: ignore[union-attr]
        judgments.sort(key=lambda j: j.weight * j.reliability, reverse=True)
        return ClaimResult(
            claim, verdict, judgments, documents, time.perf_counter() - started, check_worthiness
        )

    def learn_from_label(self, result: ClaimResult, label: Label) -> int:
        """Update site weights from a claim whose true answer is known."""
        if self.reliability is None or label not in (Label.SUPPORTED, Label.REFUTED):
            return 0
        return self.reliability.learn_from_label(
            site_stances(result.evidence), label is Label.SUPPORTED, self.config.decide_threshold
        )

    def _judgment(self, passage: Passage, stance: Stance) -> Judgment:
        site = site_of(passage.url, fallback=passage.source)
        weight = self.reliability.weight(site) if self.reliability else 1.0
        return Judgment(passage, stance, site, weight)

    async def _find_documents(
        self, claim: str, query: str, context: str | None, unavailable: dict[str, str]
    ) -> list[Candidate]:
        # A claim that leans on earlier sentences gets a second search for those
        # sentences alone, which name the subject; long combined queries often miss it.
        queries = [query, context] if context else [query]
        results = await asyncio.gather(*(self._search(q, unavailable) for q in queries))
        candidates = list({(c.source, c.id): c for c in _interleave(results)}.values())
        if len(candidates) <= 1:
            return candidates
        relevance = await self.judge.document_relevance(claim, candidates, context)
        ranked = sorted(zip(relevance, range(len(candidates))), reverse=True)
        # Keep the best documents from each source, so a source with many weak hits
        # can't crowd out another source's few strong ones.
        kept: list[Candidate] = []
        per_source: dict[str, int] = {}
        for score, i in ranked:
            source = candidates[i].source
            if score >= self.config.min_document_relevance and per_source.get(source, 0) < self.config.max_documents:
                kept.append(candidates[i])
                per_source[source] = per_source.get(source, 0) + 1
        # If nothing clears the bar, still judge the best candidate: the stance step
        # will report "says nothing" if it is truly irrelevant.
        return kept or [candidates[ranked[0][1]]]

    async def _search(self, query: str, unavailable: dict[str, str]) -> list[Candidate]:
        if isinstance(self.source, MultiSource):
            candidates, failures = await self.source.search_with_failures(query, self.config.search_limit)
            unavailable.update(failures)
            return candidates
        return await self.source.search(query, self.config.search_limit)

    async def _top_passages(self, query: str, documents: list[Candidate]) -> list[Passage]:
        per_document = await asyncio.gather(
            *(self.source.passages(d, self.config.passage_words) for d in documents)
        )
        passages = [p for doc in per_document for p in doc]
        if not passages:
            return []
        scores = bm25_scores(query, [f"{p.title} {p.text}" for p in passages])
        ranked = [p for _, p in sorted(zip(scores, passages), key=lambda x: x[0], reverse=True)]
        # Independent sites are what make a verdict trustworthy, so take each site's best
        # passage first, then fill the remaining slots by relevance.
        top: list[Passage] = []
        seen_sites: set[str] = set()
        for p in ranked:
            site = site_of(p.url, fallback=p.source)
            if site not in seen_sites and len(top) < self.config.max_passages:
                top.append(p)
                seen_sites.add(site)
        for p in ranked:
            if len(top) >= self.config.max_passages:
                break
            if p not in top:
                top.append(p)
        # The lead paragraph of the most relevant document summarizes its key facts and
        # is often the best evidence even when it shares few words with the claim.
        lead = next((p for p in per_document[0] if p.position == 0), None) if per_document else None
        if lead is not None and lead not in top:
            top.append(lead)
        return top


def _interleave(lists: list[list[Candidate]]) -> list[Candidate]:
    """Round-robin merge, so every query's best hits come before any query's weaker ones."""
    merged = []
    for rank in range(max((len(items) for items in lists), default=0)):
        merged.extend(items[rank] for items in lists if rank < len(items))
    return merged
