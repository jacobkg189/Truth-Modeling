"""Turn per-passage stance judgments into one verdict per claim.

This step is plain code on purpose: Jev's calibrated probabilities make the
combination transparent and tunable, and every verdict can be traced back to
the passages that produced it.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from enum import Enum

from .config import Config
from .jev import Stance
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

    @property
    def weight(self) -> float:
        """How decisive this passage is, in either direction."""
        return max(self.stance.supports, self.stance.contradicts)


@dataclass
class Verdict:
    label: Label
    confidence: float
    support: float  # strongest support from any passage
    refute: float  # strongest contradiction from any passage
    needs_review: bool
    reasons: list[str] = field(default_factory=list)


def aggregate(claim: str, judgments: list[Judgment], config: Config) -> Verdict:
    """Combine passage stances into a verdict.

    One clear passage is enough to settle most factual claims, so the decision uses
    the strongest support and strongest contradiction rather than an average, which
    irrelevant passages would dilute.
    """
    reasons: list[str] = []
    if not judgments:
        return Verdict(
            Label.NOT_ENOUGH_EVIDENCE, 1.0, 0.0, 0.0, True, ["No evidence passages were retrieved."]
        )

    support = max(j.stance.supports for j in judgments)
    refute = max(j.stance.contradicts for j in judgments)

    if support >= config.conflict_threshold and refute >= config.conflict_threshold:
        label, confidence = Label.CONFLICTING, min(support, refute)
        reasons.append("Some sources support the claim and others contradict it.")
    elif support >= config.decide_threshold and support > refute:
        label, confidence = Label.SUPPORTED, support * (1 - refute)
    elif refute >= config.decide_threshold and refute > support:
        label, confidence = Label.REFUTED, refute * (1 - support)
    else:
        label, confidence = Label.NOT_ENOUGH_EVIDENCE, 1 - max(support, refute)
        reasons.append("No retrieved passage clearly supports or contradicts the claim.")

    needs_review = label is Label.CONFLICTING or confidence < config.review_below

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
