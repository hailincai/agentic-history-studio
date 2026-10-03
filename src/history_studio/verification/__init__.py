"""Claim-driven bounded investigation; evidence acceptance and verdicts remain separate."""
from .agent import FactChecker
from .investigation import InvestigationOutcome, InvestigationStopReason

__all__ = ["FactChecker", "InvestigationOutcome", "InvestigationStopReason"]
