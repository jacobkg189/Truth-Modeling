import json

import httpx2
import pytest
from typesafe_sdk import Choice, Noul, TypeSafeAuthenticationError, TypeSafeInternalServerError

from factchecker.cli import best_excerpt, render_text, report_to_dict
from factchecker.jev import STANCE_OPTIONS, JevJudge
from factchecker.reliability import SourceReliability
from factchecker.sources import Passage
from factchecker.verdict import Label

from factchecker.pipeline import FactChecker

from .conftest import FakeJev, FakeSource, make_checker

DOCUMENTS = {
    "Eiffel Tower": [
        "The Eiffel Tower is a lattice tower in Paris, France. [SUPPORTS]",
        "The tower has three levels for visitors, with restaurants on the first two.",
    ],
    "Rome": ["Rome is the capital city of Italy and has no Eiffel Tower. [CONTRADICTS]"],
}


async def test_check_text_detects_claims_and_skips_the_rest():
    checker = make_checker({"Eiffel Tower": DOCUMENTS["Eiffel Tower"]})
    report = await checker.check_text(
        "I think Paris is lovely. The Eiffel Tower is in Paris. Have you been there?"
    )
    assert [c.claim for c in report.claims] == ["The Eiffel Tower is in Paris."]
    assert [s for s, _ in report.skipped] == ["I think Paris is lovely.", "Have you been there?"]
    result = report.claims[0]
    assert result.verdict.label is Label.SUPPORTED
    assert not result.verdict.needs_review
    assert result.evidence[0].stance.label == "supports"  # most decisive passage first
    assert report.usage.requests > 0 and report.usage.cost_usd > 0


async def test_no_detect_checks_every_sentence():
    checker = make_checker({"Eiffel Tower": DOCUMENTS["Eiffel Tower"]})
    report = await checker.check_text("Is it in Paris? It is.", detect_claims=False)
    assert len(report.claims) == 2 and not report.skipped


async def test_conflicting_sources_are_reported():
    checker = make_checker(DOCUMENTS)
    result = await checker.check_claim("The Eiffel Tower is in Rome.")
    assert result.verdict.label is Label.CONFLICTING
    assert result.verdict.needs_review


async def test_irrelevant_documents_are_filtered_before_stance():
    jev = FakeJev()
    checker = make_checker(DOCUMENTS, jev, max_documents=1)
    await checker.check_claim("The Eiffel Tower is tall.")
    judged_titles = {state["evidence"]["source"] for state, _ in jev.calls if "evidence" in state}
    assert judged_titles == {"Eiffel Tower"}


async def test_pronoun_claim_borrows_previous_sentence():
    jev = FakeJev()
    checker = make_checker({"Eiffel Tower": DOCUMENTS["Eiffel Tower"]}, jev)
    await checker.check_text("The Eiffel Tower is in Paris. It has three levels.", detect_claims=False)
    assert "The Eiffel Tower is in Paris. It has three levels." in checker.source.queries
    contexts = [s.get("text_before_claim") for s, _ in jev.calls if "evidence" in s]
    assert "The Eiffel Tower is in Paris." in contexts


async def test_stance_averages_out_first_option_bias():
    judge = JevJudge(FakeJev(first_option_bias=0.5))
    stance = await judge.stance("claim", Passage("t", "Doc", "neutral text"))
    # The forward ordering alone is dragged to "supports" (0.525) by the bias; averaging
    # with the reversed ordering restores the right answer and flags the disagreement.
    assert stance.label == "says_nothing"
    assert stance.supports == pytest.approx(0.275)
    assert not stance.order_agreement


async def test_stance_request_shape():
    jev = FakeJev()
    await JevJudge(jev).stance("c", Passage("t", "Doc", "text", section="History"), context="before")
    state, questions = jev.calls[0]
    assert state == {
        "text_before_claim": "before",
        "claim": "c",
        "evidence": {"source": "Doc", "text": "text", "section": "History"},
    }
    assert all(isinstance(q, Choice) for q in questions.values())
    assert questions["forward"].instructions["note"].startswith("`claim` continues")
    assert list(questions["forward"].criteria) == list(STANCE_OPTIONS)
    assert list(questions["reverse"].criteria) == list(reversed(STANCE_OPTIONS))


async def test_detection_batches_large_texts():
    jev = FakeJev()
    scores = await JevJudge(jev).detect_claims([f"Fact number {i}." for i in range(95)])
    assert len(scores) == 95
    assert len(jev.calls) == 3
    assert all(isinstance(q, Noul) for _, qs in jev.calls for q in qs.values())


class FailingJev(FakeJev):
    def __init__(self, error: Exception) -> None:
        super().__init__()
        self.error = error

    async def system_one(self, state, questions):
        if "evidence" in state:
            raise self.error
        return await super().system_one(state, questions)


def _api_error(cls, status):
    return cls(status, None, httpx2.Headers(), "boom", "/v1/systemone")


async def test_server_error_on_one_claim_becomes_flagged_result():
    error = _api_error(TypeSafeInternalServerError, 500)
    checker = make_checker(DOCUMENTS, FailingJev(error))
    result = await checker.check_claim("The Eiffel Tower is in Paris.")
    assert result.error and result.verdict.needs_review
    assert result.verdict.label is Label.NOT_ENOUGH_EVIDENCE


async def test_authentication_error_propagates():
    error = _api_error(TypeSafeAuthenticationError, 401)
    checker = make_checker(DOCUMENTS, FailingJev(error))
    with pytest.raises(TypeSafeAuthenticationError):
        await checker.check_claim("The Eiffel Tower is in Paris.")


async def test_report_serializes_and_renders():
    checker = make_checker({"Eiffel Tower": DOCUMENTS["Eiffel Tower"]})
    report = await checker.check_text("The Eiffel Tower is in Paris.")
    data = report_to_dict(report)
    json.dumps(data)  # must be JSON-serializable
    assert data["claims"][0]["verdict"] == "SUPPORTED"
    assert data["claims"][0]["evidence"][0]["url"].startswith("https://")
    text = render_text(report, color=False, verbose=True)
    assert "SUPPORTED" in text and "Eiffel Tower" in text


async def test_close_closes_client():
    jev = FakeJev()
    async with make_checker(DOCUMENTS, jev):
        pass
    assert jev.closed


async def test_consensus_learning_updates_site_weights():
    documents = {
        "Wire One": ["The Eiffel Tower is in Paris. [SUPPORTS]"],
        "Wire Two": ["The Eiffel Tower stands in Paris, France. [SUPPORTS]"],
        "Hoax Blog": ["The Eiffel Tower was moved to Rome. [CONTRADICTS]"],
    }
    reliability = SourceReliability(priors={"wire-one.example": 0.95, "wire-two.example": 0.95})
    checker = make_checker(documents, reliability=reliability, learn=True, min_document_relevance=0.0)
    result = await checker.check_claim("The Eiffel Tower is in Paris.")
    assert result.verdict.label is Label.SUPPORTED
    assert {j.site for j in result.evidence} == {"wire-one.example", "wire-two.example", "hoax-blog.example"}
    assert reliability.records["hoax-blog.example"].disagreements > 0
    assert reliability.weight("hoax-blog.example") < reliability.prior("hoax-blog.example")


async def test_learning_is_off_unless_enabled_and_labels_can_teach():
    documents = {"Wire One": ["The Eiffel Tower is in Paris. [SUPPORTS]"], "Other": ["[CONTRADICTS] no"]}
    reliability = SourceReliability()
    checker = make_checker(documents, reliability=reliability, min_document_relevance=0.0)
    result = await checker.check_claim("The Eiffel Tower is in Paris.")
    assert reliability.records == {}
    assert checker.learn_from_label(result, Label.SUPPORTED) == 2
    assert reliability.records["wire-one.example"].agreements > 0
    assert reliability.records["other.example"].disagreements > 0
    assert checker.learn_from_label(result, Label.NOT_ENOUGH_EVIDENCE) == 0


async def test_passage_selection_covers_every_site_first():
    documents = {
        "Big Site": [f"Eiffel Tower fact number {i} about the tower in Paris." for i in range(10)],
        "Small Site": ["A short note that mentions the tower once, briefly."],
    }
    jev = FakeJev()
    checker = make_checker(documents, jev, max_passages=2, min_document_relevance=0.0)
    await checker.check_claim("The Eiffel Tower is in Paris.")
    judged_sites = {state["evidence"]["source"] for state, _ in jev.calls if "evidence" in state}
    assert judged_sites == {"Big Site", "Small Site"}


def test_best_excerpt_centers_on_matching_sentence():
    text = (
        "Facebook is a social network owned by Meta Platforms. It was founded in 2004 by Mark Zuckerberg. "
        "It had three billion users in 2021. The company's headquarters are in Menlo Park, California. "
        "It sells advertising."
    )
    excerpt = best_excerpt(text, "The company is headquartered in Menlo Park.", 120)
    assert "Menlo Park" in excerpt and len(excerpt) <= 126
    assert excerpt.startswith("… ")
    assert best_excerpt("Short text.", "anything", 120) == "Short text."


async def test_claim_with_unresolved_subject_is_not_checked():
    jev = FakeJev()
    checker = make_checker({"Walk Score": ["The company is headquartered in Seattle. [SUPPORTS]"]}, jev)
    result = await checker.check_claim("The company is headquartered in Seattle.")
    assert result.verdict.label is Label.NOT_ENOUGH_EVIDENCE and result.verdict.needs_review
    assert "the company" in result.verdict.reasons[0].lower()
    assert jev.calls == []  # no search, no model calls

    named = await checker.check_claim("The company Walk Score is headquartered in Seattle.")
    assert named.verdict.label is Label.SUPPORTED


def test_unresolved_reference():
    from factchecker.pipeline import unresolved_reference

    assert unresolved_reference("It was built in 1889.") == "It"
    assert unresolved_reference("The company is based in Ohio.") == "The company"
    assert unresolved_reference("The company Apple was founded in 1976.") is None
    assert unresolved_reference("Italy is in Europe.") is None
    assert unresolved_reference("Its capital is Rome.") == "Its"


async def test_unavailable_source_is_noted_in_the_verdict():
    import httpx

    from factchecker.sources import MultiSource

    from .test_sources import StaticSource

    class Down(StaticSource):
        async def search(self, query, limit):
            raise httpx.HTTPStatusError(
                "429", request=httpx.Request("POST", "https://x"), response=httpx.Response(429)
            )

    source = MultiSource([Down("web"), FakeSource({"Eiffel Tower": DOCUMENTS["Eiffel Tower"]})])
    checker = FactChecker(JevJudge(FakeJev()), source)
    result = await checker.check_claim("The Eiffel Tower is in Paris.")
    assert result.verdict.label is Label.SUPPORTED
    assert "The web source was unavailable (HTTP 429); checked without it." in result.verdict.reasons
