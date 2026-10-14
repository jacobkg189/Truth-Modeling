"""Turn per-passage stance judgments into one verdict per claim.

This step is plain code on purpose: Jev's calibrated probabilities make the
combination transparent and tunable, and every verdict can be traced back to
the passages and websites that produced it.

Evidence is pooled per website. Each site contributes its single strongest
position, scaled by its reliability weight, and independent sites combine like
independent witnesses:

    support = 1 − ∏ over sites (1 − weight × site's strongest support)

So one reliable site can carry a verdict, two independent reliable sites make it
stronger, and a low-weight site on its own cannot decide anything.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from enum import Enum

from .config import Config
from .jev import Stance
from .reliability import SiteStance
from .sources import Passage
from .text import extract_numbers


class Label(str, Enum):
    SUPPORTED = "SUPPORTED"
    REFUTED = "REFUTED"
    CONFLICTING = "CONFLICTING"
    NOT_ENOUGH_EVIDENCE = "NOT_ENOUGH_EVIDENCE"


@dataclass(frozen=True)
class Judgment:
    passage: Passage
    stance: Stance
    site: str = ""  # the website (or corpus) the passage came from
    reliability: float = 1.0  # that site's weight when the claim was judged

    @property
    def weight(self) -> float:
        """How decisive this passage is, in either direction."""
        return max(self.stance.supports, self.stance.contradicts)

    @property
    def site_key(self) -> str:
        return self.site or self.passage.title


@dataclass
class Verdict:
    label: Label
    confidence: float
    support: float  # pooled, reliability-weighted support across sites
    refute: float  # pooled, reliability-weighted contradiction across sites
    needs_review: bool
    reasons: list[str] = field(default_factory=list)


def site_stances(judgments: list[Judgment]) -> list[SiteStance]:
    """Each site's strongest support and strongest contradiction, counted once per site."""
    best: dict[str, tuple[float, float]] = {}
    for j in judgments:
        s, c = best.get(j.site_key, (0.0, 0.0))
        best[j.site_key] = (max(s, j.stance.supports), max(c, j.stance.contradicts))
    return [SiteStance(site, s, c) for site, (s, c) in best.items()]


def pooled(judgments: list[Judgment]) -> tuple[float, float]:
    """Reliability-weighted (support, refute), combining sites as independent witnesses."""
    reliability = {j.site_key: j.reliability for j in judgments}
    no_support = no_refute = 1.0
    for stance in site_stances(judgments):
        r = reliability[stance.site]
        no_support *= 1 - r * stance.supports
        no_refute *= 1 - r * stance.contradicts
    return 1 - no_support, 1 - no_refute


def aggregate(claim: str, judgments: list[Judgment], config: Config) -> Verdict:
    reasons: list[str] = []
    if not judgments:
        return Verdict(
            Label.NOT_ENOUGH_EVIDENCE, 1.0, 0.0, 0.0, True, ["No evidence passages were retrieved."]
        )

    support, refute = pooled(judgments)

    both_strong = support >= config.conflict_threshold and refute >= config.conflict_threshold
    if both_strong and abs(support - refute) < config.conflict_margin:
        label, confidence = Label.CONFLICTING, min(support, refute)
        reasons.append("Comparably trusted sources support and contradict the claim.")
    elif support >= config.decide_threshold and support > refute:
        label, confidence = Label.SUPPORTED, support * (1 - refute)
    elif refute >= config.decide_threshold and refute > support:
        label, confidence = Label.REFUTED, refute * (1 - support)
    else:
        label, confidence = Label.NOT_ENOUGH_EVIDENCE, 1 - max(support, refute)
        reasons.append("No retrieved source clearly supports or contradicts the claim.")

    needs_review = label is Label.CONFLICTING or confidence < config.review_below

    if label in (Label.SUPPORTED, Label.REFUTED):
        dissent = [
            s.site
            for s in site_stances(judgments)
            if (s.contradicts if label is Label.SUPPORTED else s.supports) >= config.decide_threshold
        ]
        if dissent:
            needs_review = True
            reasons.append(f"Outweighed but disagreeing: {', '.join(sorted(dissent))}.")

    decisive = [j for j in judgments if j.weight >= config.decide_threshold]
    if any(not j.stance.order_agreement for j in decisive):
        needs_review = True
        reasons.append("Jev's answer changed when the stance options were reordered.")

    if label is Label.SUPPORTED:
        # Jev is weak at numeric precision, so check numbers in code: every number in the
        # claim should appear in at least one passage that supports it.
        claim_numbers = extract_numbers(claim)
        supporting = [j for j in judgments if j.stance.supports >= config.conflict_threshold]
        evidence_numbers = set().union(*(extract_numbers(j.passage.text) for j in supporting))
        missing = sorted(claim_numbers - evidence_numbers)
        if missing:
            needs_review = True
            reasons.append(
                f"Number(s) {', '.join(missing)} in the claim do not appear in the supporting evidence."
            )

    if needs_review and confidence < config.review_below:
        reasons.append(f"Confidence {confidence:.2f} is below the review threshold {config.review_below}.")

    return Verdict(label, confidence, support, refute, needs_review, reasons)


def consensus_observations(
    claim: str, judgments: list[Judgment], config: Config
) -> list[tuple[str, bool, float]]:
    """(site, agreed, strength) for each site that took a side, judged leave-one-out.

    A site is scored against the verdict reached from every *other* site's evidence,
    and only when that verdict is confident. A site can't confirm itself, and a lone
    source with nothing independent to compare against is left untouched.
    """
    observations = []
    for stance in site_stances(judgments):
        side = max(stance.supports, stance.contradicts)
        if side < config.decide_threshold:
            continue
        others = [j for j in judgments if j.site_key != stance.site]
        if not others:
            continue
        reference = aggregate(claim, others, config)
        if reference.label not in (Label.SUPPORTED, Label.REFUTED):
            continue
        if reference.confidence < config.learn_min_confidence:
            continue
        said = Label.SUPPORTED if stance.supports > stance.contradicts else Label.REFUTED
        observations.append((stance.site, said is reference.label, side * reference.confidence))
    return observations
