"""How much to trust each website, learned from how often it agrees with the evidence.

Every site starts from a prior (a starting weight between 0 and 1) and moves as
evidence accumulates:

    weight = (prior × strength + agreements) / (strength + agreements + disagreements)

``strength`` is how many observations the prior is worth, so one claim can only
nudge a weight; it takes a consistent record to move it far.

Observations come from two places:

* **Ground truth** (``record`` with a known label, e.g. from ``factcheck-eval --learn``):
  the strongest signal.
* **Consensus** (``learn_from_consensus``): after a confident verdict, each site that
  took a side is compared with the verdict reached *without its own evidence*
  (leave-one-out). This guards against the usual failure modes of "agree with the
  majority" learning:

  - a site can't vouch for itself, because its own evidence is excluded;
  - a lone source with no independent corroboration is neither rewarded nor punished;
  - low-weight sites can't outvote anyone, because consensus is pooled by weight,
    so a few weak sites never reach the confidence needed to trigger an update;
  - each site counts once per claim, however many of its passages were judged.
"""

from __future__ import annotations

import json
import math
import os
from collections.abc import Iterable
from dataclasses import dataclass
from pathlib import Path
from urllib.parse import urlparse

# Starting weights. Tune freely: learned observations are stored separately, so a
# prior can change without losing what has been learned.
DEFAULT_PRIORS: dict[str, float] = {
    # Wire services, reference works, scientific publishers, fact-checkers.
    "reuters.com": 0.92, "apnews.com": 0.92, "afp.com": 0.9, "britannica.com": 0.92,
    "nobelprize.org": 0.93, "nature.com": 0.92, "science.org": 0.92, "nejm.org": 0.92,
    "thelancet.com": 0.92, "sciencedirect.com": 0.85, "springer.com": 0.85, "jstor.org": 0.85,
    "factcheck.org": 0.92, "snopes.com": 0.9, "politifact.com": 0.9, "fullfact.org": 0.9,
    "un.org": 0.88, "worldbank.org": 0.88, "imf.org": 0.88, "oecd.org": 0.88,
    # Major news organizations.
    "bbc.co.uk": 0.88, "bbc.com": 0.88, "npr.org": 0.88, "pbs.org": 0.88,
    "nytimes.com": 0.85, "washingtonpost.com": 0.85, "wsj.com": 0.85, "theguardian.com": 0.85,
    "economist.com": 0.87, "ft.com": 0.87, "bloomberg.com": 0.85, "cnn.com": 0.8,
    "cbsnews.com": 0.8, "nbcnews.com": 0.8, "abcnews.go.com": 0.8, "aljazeera.com": 0.78,
    # Encyclopedias and preprints.
    "wikipedia.org": 0.85, "arxiv.org": 0.7, "fandom.com": 0.5,
    # User-generated, social and Q&A sites: anyone can publish anything.
    "medium.com": 0.45, "substack.com": 0.45, "blogspot.com": 0.35, "wordpress.com": 0.35,
    "youtube.com": 0.35, "scribd.com": 0.35, "chegg.com": 0.35, "slideshare.net": 0.3,
    "quora.com": 0.3, "reddit.com": 0.3, "coursehero.com": 0.3, "crosswordsolver.org": 0.3,
    "answers.com": 0.25, "brainly.com": 0.25, "facebook.com": 0.25, "instagram.com": 0.25,
    "x.com": 0.25, "twitter.com": 0.25, "pinterest.com": 0.25, "tiktok.com": 0.2,
    # A local corpus is a set of documents the user chose to trust.
    "local": 0.9,
}
# Matched as a suffix ("nih.gov") or before a country code ("nhs.gov.uk"-style, "ox.ac.uk").
SUFFIX_PRIORS: dict[str, float] = {".gov": 0.9, ".mil": 0.85, ".edu": 0.85, ".ac": 0.85, ".int": 0.88}
DEFAULT_PRIOR = 0.6

# Second-level labels under country codes: "bbc.co.uk" is a site, "co.uk" is not.
_SECOND_LEVEL = {"co", "com", "gov", "ac", "org", "net", "edu", "go", "or", "ne", "gob", "nic"}


def site_of(url: str | None, fallback: str = "unknown") -> str:
    """The registrable site behind a URL: 'https://pmc.ncbi.nlm.nih.gov/x' -> 'nih.gov'."""
    if not url or not url.startswith(("http://", "https://")):
        return fallback
    host = (urlparse(url).hostname or "").lower().rstrip(".")
    if not host:
        return fallback
    labels = host.split(".")
    if len(labels) >= 3 and len(labels[-1]) == 2 and labels[-2] in _SECOND_LEVEL:
        return ".".join(labels[-3:])
    return ".".join(labels[-2:])


@dataclass
class SiteRecord:
    agreements: float = 0.0
    disagreements: float = 0.0


@dataclass(frozen=True)
class SiteStance:
    """One site's strongest position on one claim."""

    site: str
    supports: float
    contradicts: float


class SourceReliability:
    def __init__(
        self,
        path: str | Path | None = None,
        *,
        priors: dict[str, float] | None = None,
        prior_strength: float = 20.0,
        floor: float = 0.05,
        ceiling: float = 0.97,
    ) -> None:
        self.path = Path(path).expanduser() if path else None
        self.priors = DEFAULT_PRIORS if priors is None else priors
        self.prior_strength = prior_strength
        self.floor = floor
        self.ceiling = ceiling
        self.records: dict[str, SiteRecord] = {}
        self.dirty = False
        if self.path and self.path.is_file():
            self._load()

    def prior(self, site: str) -> float:
        if site in self.priors:
            return self.priors[site]
        for suffix, value in SUFFIX_PRIORS.items():
            if site.endswith(suffix) or f"{suffix}." in f"{site}.":
                return value
        return DEFAULT_PRIOR

    def weight(self, site: str) -> float:
        record = self.records.get(site, SiteRecord())
        k = self.prior_strength
        value = (self.prior(site) * k + record.agreements) / (
            k + record.agreements + record.disagreements
        )
        return min(self.ceiling, max(self.floor, value))

    def record(self, site: str, agreed: bool, strength: float = 1.0) -> None:
        """Add one observation of the site agreeing (or not) with the established answer."""
        if strength <= 0 or math.isnan(strength):
            return
        entry = self.records.setdefault(site, SiteRecord())
        if agreed:
            entry.agreements += strength
        else:
            entry.disagreements += strength
        self.dirty = True

    def learn_from_label(self, stances: Iterable[SiteStance], label_is_true: bool, min_side: float) -> int:
        """Score each site that took a side against a known true/false label."""
        updates = 0
        for stance in stances:
            side = max(stance.supports, stance.contradicts)
            if side < min_side:
                continue
            said_true = stance.supports > stance.contradicts
            self.record(stance.site, said_true == label_is_true, side)
            updates += 1
        return updates

    def table(self) -> list[tuple[str, float, float, float, float]]:
        """(site, weight, prior, agreements, disagreements) for every site with a record."""
        rows = [
            (site, self.weight(site), self.prior(site), r.agreements, r.disagreements)
            for site, r in self.records.items()
        ]
        return sorted(rows, key=lambda row: (-(row[3] + row[4]), row[0]))

    def save(self) -> None:
        if not self.path or not self.dirty:
            return
        self.path.parent.mkdir(parents=True, exist_ok=True)
        data = {
            "version": 1,
            "sites": {
                site: {"agreements": round(r.agreements, 4), "disagreements": round(r.disagreements, 4)}
                for site, r in sorted(self.records.items())
            },
        }
        temporary = self.path.with_suffix(f".{os.getpid()}.tmp")
        temporary.write_text(json.dumps(data, indent=1))
        temporary.replace(self.path)
        self.dirty = False

    def _load(self) -> None:
        try:
            data = json.loads(self.path.read_text())  # type: ignore[union-attr]
            for site, values in data.get("sites", {}).items():
                self.records[site] = SiteRecord(
                    float(values.get("agreements", 0)), float(values.get("disagreements", 0))
                )
        except (OSError, ValueError, AttributeError):
            self.records = {}  # a corrupt file must not stop fact checking


def default_reliability_path() -> Path:
    base = os.environ.get("FACTCHECK_CACHE_DIR", "~/.cache/jev-factchecker")
    return Path(os.environ.get("FACTCHECK_RELIABILITY_PATH", f"{base}/source_reliability.json")).expanduser()
