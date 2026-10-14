from __future__ import annotations

from typing import Any

import pytest
from typesafe_sdk import Choice, Noul, SystemOneResponse

from factchecker.config import Config
from factchecker.jev import JevJudge
from factchecker.pipeline import FactChecker
from factchecker.sources import Candidate, Passage


class FakeJev:
    """Scripted stand-in for AsyncTypeSafeClient, returning real SDK response objects.

    * Claim detection: a sentence is a claim unless it is a question or an opinion.
    * Document relevance: high when the document title appears in the claim.
    * Stance: decided by a marker in the evidence text: [SUPPORTS] or [CONTRADICTS].
    * ``first_option_bias`` shifts probability onto whichever option is listed first,
      imitating the ordering bias the stance check averages away.
    """

    def __init__(self, first_option_bias: float = 0.0) -> None:
        self.first_option_bias = first_option_bias
        self.calls: list[tuple[Any, dict[str, Any]]] = []
        self.closed = False

    async def system_one(self, state: Any, questions: dict[str, Any]) -> SystemOneResponse:
        self.calls.append((state, questions))
        answers = {}
        for qid, question in questions.items():
            if isinstance(question, Noul):
                answers[qid] = {"type": "noul", "noul": self._noul(state, question.instructions)}
            elif isinstance(question, Choice):
                answers[qid] = self._choice(state, list(question.criteria))
        return SystemOneResponse.model_validate(
            {"model": "jev-fake", "usage": {"input_tokens": 100, "output_tokens": 10}, "answers": answers}
        )

    def _noul(self, state: Any, instructions: dict) -> float:
        if "sentence" in instructions:
            sentence = instructions["sentence"]
            return 0.05 if sentence.endswith("?") or sentence.lower().startswith("i think") else 0.95
        title = instructions["document"]["title"].lower()
        return 0.9 if title in state["claim"].lower() else 0.1

    def _choice(self, state: Any, options: list[str]) -> dict:
        text = state["evidence"]["text"]
        if "[SUPPORTS]" in text:
            probs = {"supports": 0.9, "contradicts": 0.02, "says_nothing": 0.08}
        elif "[CONTRADICTS]" in text:
            probs = {"supports": 0.02, "contradicts": 0.9, "says_nothing": 0.08}
        else:
            probs = {"supports": 0.05, "contradicts": 0.05, "says_nothing": 0.9}
        if self.first_option_bias:
            first = options[0]
            probs = {k: v * (1 - self.first_option_bias) for k, v in probs.items()}
            probs[first] += self.first_option_bias
        choice = max(probs, key=probs.get)
        n = len(probs)
        return {
            "type": "choice",
            "choice": choice,
            "confidence": (probs[choice] - 1 / n) / (1 - 1 / n),
            "probabilities": probs,
        }

    async def aclose(self) -> None:
        self.closed = True


class FakeSource:
    name = "fake"

    def __init__(self, documents: dict[str, list[str]]) -> None:
        self.documents = documents
        self.queries: list[str] = []

    async def search(self, query: str, limit: int) -> list[Candidate]:
        self.queries.append(query)
        return [Candidate(id=t, title=t, snippet=texts[0][:80]) for t, texts in self.documents.items()][:limit]

    async def passages(self, candidate: Candidate, max_words: int) -> list[Passage]:
        return [
            Passage("fake", candidate.title, text, f"https://example.org/{candidate.title}", None, i)
            for i, text in enumerate(self.documents[candidate.title])
        ]

    async def aclose(self) -> None:
        return None


@pytest.fixture
def fake_jev() -> FakeJev:
    return FakeJev()


def make_checker(documents: dict[str, list[str]], jev: FakeJev | None = None, **config: Any) -> FactChecker:
    return FactChecker(JevJudge(jev or FakeJev()), FakeSource(documents), Config(**config))
