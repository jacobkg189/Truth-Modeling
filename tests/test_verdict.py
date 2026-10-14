import pytest

from factchecker.config import Config
from factchecker.jev import Stance
from factchecker.sources import Passage
from factchecker.verdict import Judgment, Label, aggregate, consensus_observations

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


def site_judgment(site: str, reliability: float, supports: float, contradicts: float, text: str = "e") -> Judgment:
    stance = Stance(supports, contradicts, max(0.0, 1 - supports - contradicts), True)
    return Judgment(Passage("web", site, text, f"https://{site}/"), stance, site, reliability)


def test_independent_sites_pool_and_weak_sites_cannot_decide_alone():
    one_unknown = aggregate("c", [site_judgment("a.com", 0.6, 1.0, 0.0)], CONFIG)
    assert one_unknown.label is Label.SUPPORTED and one_unknown.needs_review

    two_unknown = aggregate(
        "c", [site_judgment("a.com", 0.6, 1.0, 0.0), site_judgment("b.com", 0.6, 1.0, 0.0)], CONFIG
    )
    assert two_unknown.support == pytest.approx(1 - 0.4 * 0.4)
    assert two_unknown.label is Label.SUPPORTED

    low_weight = aggregate("c", [site_judgment("reddit.com", 0.3, 1.0, 0.0)], CONFIG)
    assert low_weight.label is Label.NOT_ENOUGH_EVIDENCE


def test_one_site_counts_once_however_many_passages():
    many = [site_judgment("a.com", 0.6, 1.0, 0.0, text=f"p{i}") for i in range(5)]
    assert aggregate("c", many, CONFIG).support == pytest.approx(0.6)


def test_trusted_refutation_outweighs_low_weight_support_but_flags_dissent():
    verdict = aggregate(
        "c",
        [site_judgment("reuters.com", 0.92, 0.0, 0.98), site_judgment("random-blog.com", 0.6, 0.9, 0.0)],
        CONFIG,
    )
    assert verdict.label is Label.REFUTED
    assert verdict.needs_review
    assert any("random-blog.com" in reason for reason in verdict.reasons)


def test_comparable_sites_disagreeing_is_conflicting():
    verdict = aggregate(
        "c", [site_judgment("a.org", 0.85, 0.95, 0.0), site_judgment("b.org", 0.85, 0.0, 0.9)], CONFIG
    )
    assert verdict.label is Label.CONFLICTING


def test_consensus_rewards_corroborated_sites_and_penalizes_outliers():
    agreeing = [
        site_judgment("apnews.com", 0.92, 0.97, 0.0),
        site_judgment("reuters.com", 0.92, 0.95, 0.0),
        site_judgment("bbc.com", 0.88, 0.9, 0.0),
        site_judgment("neutral.com", 0.6, 0.05, 0.05),
    ]
    observations = {site: agreed for site, agreed, _ in consensus_observations("c", agreeing, CONFIG)}
    assert observations == {"apnews.com": True, "reuters.com": True, "bbc.com": True}

    with_hoax = agreeing[:2] + [site_judgment("hoax.com", 0.6, 0.0, 0.9)]
    observations = {site: agreed for site, agreed, _ in consensus_observations("c", with_hoax, CONFIG)}
    # The hoax is contradicted by two wire services and loses weight. The wire services
    # aren't rewarded here: leaving either out, the evidence left is contested.
    assert observations == {"hoax.com": False}


def test_consensus_needs_independent_confident_evidence():
    lone = [site_judgment("a.com", 0.9, 1.0, 0.0)]
    assert consensus_observations("c", lone, CONFIG) == []  # nothing to compare against

    weak_crowd = [
        site_judgment("x.com", 0.25, 1.0, 0.0),
        site_judgment("tiktok.com", 0.2, 1.0, 0.0),
        site_judgment("reuters.com", 0.92, 0.0, 0.95),
    ]
    observations = {site for site, _, _ in consensus_observations("c", weak_crowd, CONFIG)}
    # Two low-weight sites agreeing can't form a confident consensus against the wire service.
    assert "reuters.com" not in observations
