"""The fact-checking pipeline.

    text ──split──▶ sentences ──Jev: check-worthy?──▶ claims
    claim ──search──▶ candidate documents ──Jev: relevant?──▶ documents
    documents ──chunk + BM25──▶ top passages ──Jev: stance (×2 orders)──▶ judgments
    judgments ──aggregate in code──▶ verdict + confidence + review flag

Claims are checked concurrently, and each claim's passages are judged
concurrently, so latency is a few sequential round trips regardless of length.
"""

from __future__ import annotations

import asyncio
import re
import time
from dataclasses import dataclass, field, replace

import httpx
from typesafe_sdk import (
    TypeSafeAPIConnectionError,
    TypeSafeAPIError,
    TypeSafeAuthenticationError,
    TypeSafePermissionDeniedError,
)

from .config import Config
from .jev import JevJudge, Usage
from .sources import (
    Candidate,
    EvidenceSource,
    LocalCorpusSource,
    Passage,
    WikipediaSource,
    default_cache_dir,
)
from .text import bm25_scores, split_sentences
from .verdict import Judgment, Label, Verdict, aggregate

_LEADING_REFERENCE = re.compile(
    r"^(it|its|he|his|she|her|they|their|this|these|that|those|the company|the city|the country)\b",
    re.IGNORECASE,
)


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
    def __init__(self, judge: JevJudge, source: EvidenceSource, config: Config | None = None) -> None:
        self.judge = judge
        self.source = source
        self.config = config or Config()

    @classmethod
    def create(
        cls,
        *,
        source: str = "wikipedia",
        corpus_dir: str | None = None,
        lang: str = "en",
        model: str | None = None,
        config: Config | None = None,
        cache: bool = True,
    ) -> "FactChecker":
        config = config or Config()
        if source == "wikipedia":
            evidence: EvidenceSource = WikipediaSource(lang, cache_dir=default_cache_dir() if cache else None)
        elif source == "local":
            if not corpus_dir:
                raise ValueError("The local source needs a corpus directory.")
            evidence = LocalCorpusSource(corpus_dir)
        else:
            raise ValueError(f"Unknown evidence source: {source!r}")
        return cls(JevJudge(model=model, max_concurrency=config.max_concurrency), evidence, config)

    async def __aenter__(self) -> "FactChecker":
        return self

    async def __aexit__(self, *exc: object) -> None:
        await self.aclose()

    async def aclose(self) -> None:
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
        needs_context = bool(context and _LEADING_REFERENCE.match(claim.strip()))
        query = f"{context} {claim}" if needs_context else claim
        try:
            documents = await self._find_documents(claim, query, context if needs_context else None)
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

        judgments = [Judgment(p, s) for p, s in zip(passages, stances)]
        verdict = aggregate(claim, judgments, self.config)
        judgments.sort(key=lambda j: j.weight, reverse=True)
        return ClaimResult(
            claim, verdict, judgments, documents, time.perf_counter() - started, check_worthiness
        )

    async def _find_documents(self, claim: str, query: str, context: str | None) -> list[Candidate]:
        # A claim that leans on earlier sentences gets a second search for those
        # sentences alone, which name the subject; long combined queries often miss it.
        queries = [query, context] if context else [query]
        results = await asyncio.gather(
            *(self.source.search(q, self.config.search_limit) for q in queries)
        )
        candidates = list({c.id: c for c in _interleave(results)}.values())
        if len(candidates) <= 1:
            return candidates
        relevance = await self.judge.document_relevance(claim, candidates, context)
        ranked = sorted(zip(relevance, range(len(candidates))), reverse=True)
        kept = [candidates[i] for score, i in ranked if score >= self.config.min_document_relevance]
        # If nothing clears the bar, still judge the best candidate: the stance step
        # will report "says nothing" if it is truly irrelevant.
        return kept[: self.config.max_documents] or [candidates[ranked[0][1]]]

    async def _top_passages(self, query: str, documents: list[Candidate]) -> list[Passage]:
        per_document = await asyncio.gather(
            *(self.source.passages(d, self.config.passage_words) for d in documents)
        )
        passages = [p for doc in per_document for p in doc]
        if not passages:
            return []
        scores = bm25_scores(query, [f"{p.title} {p.text}" for p in passages])
        ranked = [p for _, p in sorted(zip(scores, passages), key=lambda x: x[0], reverse=True)]
        top = ranked[: self.config.max_passages]
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
