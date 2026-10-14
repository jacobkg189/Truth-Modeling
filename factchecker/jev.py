"""Every judgment the fact checker asks of Jev.

Jev answers typed questions about a ``state`` with calibrated probabilities. The
fact checker uses it for three narrow decisions, following TypeSafe's guidance
to keep each question atomic and the state small:

1. Claim detection: is this sentence a checkable factual assertion? (Noul)
2. Document relevance: is this candidate document about the claim's subject? (Noul)
3. Stance: does this passage support, contradict, or say nothing about the claim? (Choice)

Related questions go out together in one request (speculative fan-out), and
independent requests run concurrently, so a whole text is checked in a handful
of round trips.
"""

from __future__ import annotations

import asyncio
from dataclasses import dataclass, field
from typing import Any, Protocol

from typesafe_sdk import AsyncTypeSafeClient, Choice, Noul, SystemOneResponse

from .sources import Candidate, Passage

# Jev reads literally (see "Jev 1.13 jaggedness"), so each option spells out its
# boundary cases, and "supports" requires every part of the claim to be covered.
STANCE_OPTIONS = {
    "supports": (
        "The evidence is about the same subject as the claim and states the claim, or directly "
        "implies that every part of the claim is true"
    ),
    "contradicts": (
        "The evidence is about the same subject as the claim and states something that cannot be "
        "true at the same time as the claim: a different date, number, place, person or outcome "
        "for the same fact, or a different kind of thing altogether (the evidence says the "
        "subject is a film, while the claim says it is a food)"
    ),
    "says_nothing": (
        "The evidence is about a different subject than the claim (including a different thing "
        "with a similar name), or it does not settle what the claim asserts either way: it covers "
        "only part of the claim, or it is merely related background"
    ),
}
STANCE_INSTRUCTIONS = "How does `evidence` relate to `claim`?"
# When a claim leans on the sentence before it ("The company is based in Seattle"),
# say so explicitly; Jev reads literally and will not infer the referent on its own.
CONTEXT_NOTE = (
    "`claim` continues `text_before_claim`: words like \"it\", \"he\" or \"the company\" "
    "in `claim` refer to what `text_before_claim` is about."
)

CLAIM_INSTRUCTIONS = (
    "Does `sentence` make a specific factual assertion about the real world that could be "
    "checked against a reliable reference source? Opinions, predictions, questions, "
    "commands, jokes and vague statements do not count."
)

RELEVANCE_INSTRUCTIONS = (
    "Is `document` about the specific person, place, organization, work, event or thing that "
    "`claim` names or makes an assertion about, so that it could confirm or refute the claim? "
    "A document about a different thing that merely shares a name (a replica, namesake, or "
    "another place with the same name) does not count."
)

# Keep each detection request's state comfortably inside Jev's 32k-token budget.
_DETECTION_BATCH = 40


class SystemOneClient(Protocol):
    async def system_one(self, state: Any, questions: dict[str, Any]) -> SystemOneResponse: ...

    async def aclose(self) -> None: ...


@dataclass(frozen=True)
class Stance:
    supports: float
    contradicts: float
    says_nothing: float
    order_agreement: bool  # did both option orderings pick the same answer?

    @property
    def label(self) -> str:
        return max(
            ("supports", self.supports),
            ("contradicts", self.contradicts),
            ("says_nothing", self.says_nothing),
            key=lambda item: item[1],
        )[0]

    @property
    def confidence(self) -> float:
        """TypeSafe's Choice confidence formula, applied to the order-averaged distribution."""
        top = max(self.supports, self.contradicts, self.says_nothing)
        return max(0.0, (top - 1 / 3) / (1 - 1 / 3))


@dataclass
class Usage:
    requests: int = 0
    input_tokens: int = 0
    output_tokens: int = 0
    models: set[str] = field(default_factory=set)

    def add(self, response: SystemOneResponse) -> None:
        self.requests += 1
        self.input_tokens += response.usage.input_tokens or 0
        self.output_tokens += response.usage.output_tokens or 0
        self.models.add(response.model)

    @property
    def cost_usd(self) -> float:
        """Jev 1.13 pricing: $0.042 per million input tokens; output tokens are free."""
        return self.input_tokens * 0.042 / 1_000_000


class JevJudge:
    def __init__(
        self,
        client: SystemOneClient | None = None,
        *,
        model: str | None = None,
        max_concurrency: int = 32,
    ) -> None:
        self._client = client or AsyncTypeSafeClient(model=model)
        self._limit = asyncio.Semaphore(max_concurrency)
        self.usage = Usage()

    async def _ask(self, state: Any, questions: dict[str, Any]) -> SystemOneResponse:
        async with self._limit:
            response = await self._client.system_one(state, questions)
        self.usage.add(response)
        return response

    async def detect_claims(self, sentences: list[str]) -> list[float]:
        """Probability that each sentence is a checkable factual claim."""
        if not sentences:
            return []
        batches = [
            sentences[i : i + _DETECTION_BATCH] for i in range(0, len(sentences), _DETECTION_BATCH)
        ]
        responses = await asyncio.gather(*(self._detect_batch(batch) for batch in batches))
        return [p for batch in responses for p in batch]

    async def _detect_batch(self, sentences: list[str]) -> list[float]:
        # The surrounding text is the state so pronouns and references resolve; each
        # question carries its own sentence, so the model never has to index into state.
        questions = {
            f"s{i}": Noul(instructions={"question": CLAIM_INSTRUCTIONS, "sentence": sentence})
            for i, sentence in enumerate(sentences)
        }
        response = await self._ask({"text": " ".join(sentences)}, questions)
        return [response.nouls[f"s{i}"].noul for i in range(len(sentences))]

    async def document_relevance(
        self, claim: str, candidates: list[Candidate], context: str | None = None
    ) -> list[float]:
        """Probability that each candidate document is about what the claim discusses."""
        if not candidates:
            return []
        state, note = _claim_state(claim, context)
        questions = {
            f"d{i}": Noul(
                instructions={
                    "question": RELEVANCE_INSTRUCTIONS,
                    **note,
                    "document": {"title": c.title, "snippet": c.snippet},
                }
            )
            for i, c in enumerate(candidates)
        }
        response = await self._ask(state, questions)
        return [response.nouls[f"d{i}"].noul for i in range(len(candidates))]

    async def stance(self, claim: str, passage: Passage, context: str | None = None) -> Stance:
        """How the passage bears on the claim, averaged over two option orderings.

        Jev 1.13 can lean toward the first Choice option. Asking the same question
        with the options reversed, in the same request, costs no extra latency and
        cancels that bias; disagreement between the two is itself a review signal.
        """
        state, note = _claim_state(claim, context)
        state["evidence"] = {"source": passage.title, "text": passage.text}
        if passage.section:
            state["evidence"]["section"] = passage.section
        instructions: Any = {"question": STANCE_INSTRUCTIONS, **note} if note else STANCE_INSTRUCTIONS
        reversed_options = dict(reversed(list(STANCE_OPTIONS.items())))
        response = await self._ask(
            state,
            {
                "forward": Choice(instructions=instructions, criteria=STANCE_OPTIONS),
                "reverse": Choice(instructions=instructions, criteria=reversed_options),
            },
        )
        forward = response.choices["forward"]
        reverse = response.choices["reverse"]
        averaged = {
            option: (forward.probabilities.get(option, 0.0) + reverse.probabilities.get(option, 0.0)) / 2
            for option in STANCE_OPTIONS
        }
        return Stance(
            supports=averaged["supports"],
            contradicts=averaged["contradicts"],
            says_nothing=averaged["says_nothing"],
            order_agreement=forward.choice == reverse.choice,
        )

    async def aclose(self) -> None:
        await self._client.aclose()


def _claim_state(claim: str, context: str | None) -> tuple[dict[str, Any], dict[str, str]]:
    """The claim as state, plus an instruction note when it depends on the prior sentence."""
    if not context:
        return {"claim": claim}, {}
    return {"text_before_claim": context, "claim": claim}, {"note": CONTEXT_NOTE}
