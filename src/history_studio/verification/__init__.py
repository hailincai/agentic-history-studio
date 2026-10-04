"""Claim-driven bounded investigation; evidence acceptance and verdicts remain separate."""
from .agent import FactChecker
from .investigation import InvestigationOutcome, InvestigationStopReason
from .submission import VerificationSubmission

__all__ = ["FactChecker", "InvestigationOutcome", "InvestigationStopReason", "VerificationSubmission"]
