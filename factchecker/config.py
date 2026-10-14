"""Tunable thresholds for the fact-checking pipeline.

Every number here is a starting point. Run ``factcheck-eval`` on labelled claims
from your own domain and adjust; Jev's probabilities are calibrated, so moving a
threshold trades coverage for precision predictably.
"""

from __future__ import annotations

import os
from dataclasses import dataclass
from pathlib import Path


@dataclass(frozen=True)
class Config:
    # Claim detection: a sentence is check-worthy when Jev's Noul is at least this.
    claim_threshold: float = 0.5

    # Retrieval.
    search_limit: int = 6  # candidate documents fetched from each evidence source
    max_documents: int = 3  # documents kept per source after Jev's relevance filter
    min_document_relevance: float = 0.3  # Noul floor for keeping a candidate document
    max_passages: int = 8  # passages sent to Jev for a stance judgment, per claim
    passage_words: int = 120  # target passage length when chunking documents

    # Verdict aggregation (see verdict.py).
    decide_threshold: float = 0.6  # pooled support/contradiction needed to commit
    conflict_threshold: float = 0.5  # both sides at least this ...
    conflict_margin: float = 0.3  # ... and closer than this -> CONFLICTING
    review_below: float = 0.75  # verdict confidence under this is flagged for review

    # Source reliability learning (see reliability.py).
    learn_min_confidence: float = 0.8  # leave-one-out verdict needed to score a site

    # Throughput.
    max_concurrency: int = 32  # simultaneous Jev requests (API limit is ~80 rps)


def load_dotenv(path: str | Path = ".env") -> None:
    """Load KEY=VALUE lines into os.environ without overriding variables already set."""
    path = Path(path)
    if not path.is_file():
        return
    for line in path.read_text().splitlines():
        line = line.strip()
        if not line or line.startswith("#") or "=" not in line:
            continue
        key, _, value = line.partition("=")
        os.environ.setdefault(key.strip(), value.strip().strip("'\""))
