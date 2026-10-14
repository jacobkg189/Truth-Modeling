from factchecker.config import Config
from factchecker.jev import Stance
from factchecker.sources import Passage
from factchecker.verdict import Judgment, Label, aggregate

CONFIG = Config()


def judgment(supports: float, contradicts: float, text: str = "evidence", agree: bool = True) -> Judgment:
    stance = Stance(supports, contradicts, max(0.0, 1 - supports - contradicts), agree)
    return Judgment(Passage("test", "Doc", text), stance)


def test_no_evidence_is_not_enough_and_flagged():
    verdict = aggregate("claim", [], CONFIG)
    assert verdict.label is Label.NOT_ENOUGH_EVIDENCE
    assert verdict.needs_review


def test_one_strong_supporting_passage_wins_over_irrelevant_ones():
    verdict = aggregate("claim", [judgment(0.05, 0.0)] * 5 + [judgment(0.97, 0.01)], CONFIG)
    assert verdict.label is Label.SUPPORTED
    assert verdict.confidence > 0.9
    assert not verdict.needs_review


def test_strong_contradiction_refutes():
    verdict = aggregate("claim", [judgment(0.02, 0.95), judgment(0.1, 0.0)], CONFIG)
    assert verdict.label is Label.REFUTED
    assert verdict.confidence > 0.8


def test_both_sides_strong_is_conflicting_and_flagged():
    verdict = aggregate("claim", [judgment(0.9, 0.0), judgment(0.0, 0.8)], CONFIG)
    assert verdict.label is Label.CONFLICTING
    assert verdict.needs_review


def test_weak_signals_are_not_enough_evidence():
    verdict = aggregate("claim", [judgment(0.4, 0.1), judgment(0.2, 0.3)], CONFIG)
    assert verdict.label is Label.NOT_ENOUGH_EVIDENCE


def test_moderate_support_is_flagged_for_review():
    verdict = aggregate("claim", [judgment(0.65, 0.05)], CONFIG)
    assert verdict.label is Label.SUPPORTED
    assert verdict.needs_review


def test_number_missing_from_supporting_evidence_is_flagged():
    claim = "The tower is 330 metres tall."
    ok = aggregate(claim, [judgment(0.95, 0.0, "It stands 330 m (1,083 ft) tall.")], CONFIG)
    assert ok.label is Label.SUPPORTED and not ok.needs_review

    bad = aggregate(claim, [judgment(0.95, 0.0, "It stands 300 m tall.")], CONFIG)
    assert bad.label is Label.SUPPORTED and bad.needs_review
    assert any("330" in reason for reason in bad.reasons)


def test_order_disagreement_on_decisive_passage_is_flagged():
    verdict = aggregate("claim", [judgment(0.95, 0.0, agree=False)], CONFIG)
    assert verdict.label is Label.SUPPORTED
    assert verdict.needs_review
