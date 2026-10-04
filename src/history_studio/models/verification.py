"""Claim-specific verification outcomes; semantic judgments belong to the Fact Checker."""
from enum import StrEnum
from typing import Self

from pydantic import Field, field_validator, model_validator

from .base import Contract, Identifier, Text
from .research import EvidenceReference


class VerificationStatus(StrEnum):
    """Terminal outcomes of independent claim-specific investigation.

    VERIFIED: independent evidence supports the core claim without a material contradiction.
    PARTIALLY_VERIFIED: supports part/core direction, with material qualifications unresolved.
    DISPUTED: credible competing support/contradiction cannot currently be resolved.
    REJECTED: independent contradiction makes the core claim unsustainable.
    UNVERIFIED: adequate investigation leaves insufficient independent evidence for a verdict.
    Insufficient evidence alone is not rejection.
    """

    VERIFIED = "VERIFIED"
    PARTIALLY_VERIFIED = "PARTIALLY_VERIFIED"
    DISPUTED = "DISPUTED"
    REJECTED = "REJECTED"
    UNVERIFIED = "UNVERIFIED"


class VerificationEvidence(EvidenceReference):
    """Canonical provenance used during independent verification, not research proposal evidence.

    Reuses the quotation bounds and paired span/version identity contract. Source identity
    alone does not establish independence; that remains a semantic judgment.
    """


def validate_verification_structure(status: VerificationStatus, supporting: bool,
                                    conflicting: bool, unresolved: bool) -> None:
    """Shared status shape for canonical results and explicitly unaccepted proposals."""
    if status == VerificationStatus.VERIFIED and not supporting:
        raise ValueError("VERIFIED requires verification evidence")
    if status == VerificationStatus.DISPUTED and not (supporting and conflicting):
        raise ValueError("DISPUTED requires verification and contradiction evidence")
    if status == VerificationStatus.PARTIALLY_VERIFIED and (not supporting or not (unresolved or conflicting)):
        raise ValueError("PARTIALLY_VERIFIED requires verification evidence and unresolved issues or contradiction evidence")
    if status == VerificationStatus.REJECTED and not conflicting:
        raise ValueError("REJECTED requires contradiction evidence")
    if status == VerificationStatus.UNVERIFIED and not unresolved:
        raise ValueError("UNVERIFIED requires an unresolved issue")


class VerificationResult(Contract):
    """Evaluation of exactly one candidate claim, without modifying its ResearchFact."""

    verification_id: Identifier
    research_fact_id: Identifier
    claim_snapshot: str = Field(min_length=1, max_length=600,
        description="Exact ResearchFact claim evaluated; preserve its text without rewriting.")
    status: VerificationStatus
    verification_evidence: list[VerificationEvidence] = Field(default_factory=list)
    contradiction_evidence: list[VerificationEvidence] = Field(default_factory=list)
    unresolved_issues: list[Text] = Field(default_factory=list)
    independence_note: Text
    rationale: Text

    @field_validator("claim_snapshot")
    @classmethod
    def nonblank_snapshot(cls, value: str) -> str:
        if not value.strip():
            raise ValueError("Claim snapshot must not be blank")
        return value

    @model_validator(mode="after")
    def outcome_structure(self) -> Self:
        validate_verification_structure(self.status, bool(self.verification_evidence),
                                        bool(self.contradiction_evidence), bool(self.unresolved_issues))
        return self
