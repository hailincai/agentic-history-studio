"""Terminal semantic proposals; canonical verification evidence acceptance is deferred."""
from typing import Self

from pydantic import Field, model_validator

from history_studio.models.base import Contract, Identifier, Text
from history_studio.models.verification import VerificationStatus, validate_verification_structure


class VerificationEvidenceSelection(Contract):
    """Source/locator proposal, not an accepted quotation or proof of independence."""
    source_id: Identifier = Field(description="Select a source from original context or executed investigation observations; not a call/span/evidence-table ID.")
    locator: Text | None = Field(default=None, max_length=600,
        description="Optional location identifying relevant investigation material. A proposal only; Python has not extracted or accepted canonical evidence.")


class VerificationSubmissionInput(Contract):
    """Agent-owned semantic fields only; no target identity or canonical excerpt overrides."""
    status: VerificationStatus
    verification_evidence: list[VerificationEvidenceSelection] = Field(default_factory=list,
        description="Proposed supporting selections, not accepted VerificationEvidence.")
    contradiction_evidence: list[VerificationEvidenceSelection] = Field(default_factory=list,
        description="Proposed contradicting selections, not accepted VerificationEvidence.")
    unresolved_issues: list[Text] = Field(default_factory=list)
    independence_note: Text
    rationale: Text

    @model_validator(mode="after")
    def outcome_structure(self) -> Self:
        validate_verification_structure(self.status, bool(self.verification_evidence),
                                        bool(self.contradiction_evidence), bool(self.unresolved_issues))
        return self


class VerificationSubmission(VerificationSubmissionInput):
    """Runtime-bound terminal proposal. Not a VerificationResult or trusted evidence."""
    research_fact_id: Identifier
    claim_snapshot: str = Field(min_length=1, max_length=600)
