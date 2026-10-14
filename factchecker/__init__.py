"""Evidence-grounded fact checking with TypeSafe's Jev as the decision model."""

from .config import Config
from .pipeline import ClaimResult, FactChecker, Report
from .verdict import Label, Verdict

__all__ = ["ClaimResult", "Config", "FactChecker", "Label", "Report", "Verdict"]
