"""Bounded working projection; never persisted as an authoritative knowledge artifact."""
from typing import Self

from pydantic import Field, model_validator

from .artifact_reference import ArtifactReference
from .base import Contract, Identifier, Text, require_unique
from .historical_time import HistoricalTime
from .story_package import StoryFactReference, StoryFactUse
from .verification import VerificationEvidence, VerificationStatus
from .production_brief import ProductionBrief


class StoryContextFact(Contract):
    """Only accepted verification semantics plus exact research chronology.

    Evidence excerpts are canonical accepted quotations (each bounded by its existing
    contract), not source bodies. This projection is semantic, not a token budget.
    Status is copied by the builder, never inferred from research confidence.
    """

    research_fact_id: Identifier
    claim_snapshot: str = Field(min_length=1, max_length=600)
    historical_time: HistoricalTime
    status: VerificationStatus
    verification_evidence: tuple[VerificationEvidence, ...] = ()
    contradiction_evidence: tuple[VerificationEvidence, ...] = ()
    unresolved_issues: tuple[Text, ...] = ()
    rationale: Text

    @property
    def allowed_uses(self) -> tuple[StoryFactUse, ...]:
        return {
            VerificationStatus.VERIFIED: (StoryFactUse.AFFIRMATIVE, StoryFactUse.QUALIFIED),
            VerificationStatus.PARTIALLY_VERIFIED: (StoryFactUse.QUALIFIED,),
            VerificationStatus.DISPUTED: (StoryFactUse.DISPUTE,),
            VerificationStatus.REJECTED: (),
            VerificationStatus.UNVERIFIED: (),
        }[self.status]

    @property
    def qualification_required(self) -> bool:
        return self.status in (VerificationStatus.PARTIALLY_VERIFIED, VerificationStatus.DISPUTED)


class StoryPendingClaim(Contract):
    """Captured membership without an accepted result; never relabeled UNVERIFIED."""

    research_fact_id: Identifier
    claim_snapshot: str = Field(min_length=1, max_length=600)


class StoryContext(Contract):
    """Runtime supplies human approval; this context neither proves nor grants it.

    Tuple order preserves captured membership, without numeric chronology sorting.
    Excluded and pending claims are cautionary input, never narrative grounding.
    Construction through build_story_context authenticates stored snapshot lineage.
    Standalone Pydantic construction proves shape only, not source authenticity.
    """

    verification_input_ref: ArtifactReference
    production_brief: ProductionBrief | None = None
    research_input_ref: ArtifactReference
    eligible_facts: tuple[StoryContextFact, ...] = ()
    excluded_facts: tuple[StoryContextFact, ...] = ()
    pending_claims: tuple[StoryPendingClaim, ...] = ()

    @model_validator(mode="after")
    def context_shape(self) -> Self:
        if self.verification_input_ref.artifact_type != "verification":
            raise ValueError("Story input must reference a verification artifact")
        if self.research_input_ref.artifact_type != "research":
            raise ValueError("Chronology input must reference a research artifact")
        if self.verification_input_ref.project_id != self.research_input_ref.project_id:
            raise ValueError("Context snapshot projects must match")
        require_unique([f.research_fact_id for f in
                        (*self.eligible_facts, *self.excluded_facts, *self.pending_claims)], "Context fact IDs")
        if any(not f.allowed_uses for f in self.eligible_facts):
            raise ValueError("Ineligible status cannot appear in eligible facts")
        if any(f.allowed_uses for f in self.excluded_facts):
            raise ValueError("Eligible status cannot appear in excluded facts")
        return self

    def fact(self, research_fact_id: str) -> StoryContextFact:
        """Lookup accepted results, including excluded verdicts; pending has no verdict."""
        validated = type(self).model_validate(self.model_dump(mode="json"))
        for fact in (*validated.eligible_facts, *validated.excluded_facts):
            if fact.research_fact_id == research_fact_id:
                return fact
        raise ValueError("Fact has no accepted result in approved snapshot")

    def validate_fact_reference(self, reference: StoryFactReference, *,
                                expected_claim: str | None = None) -> None:
        """Check identity, verdict, use and qualification shape, never prose adequacy."""
        ref = StoryFactReference.model_validate(reference.model_dump(mode="json"))
        fact = self.fact(ref.research_fact_id)
        if expected_claim is not None and expected_claim != fact.claim_snapshot:
            raise ValueError("Expected claim must exactly match approved claim")
        if ref.status != fact.status:
            raise ValueError("Reference status must match authoritative verification status")
        if ref.use not in fact.allowed_uses:
            raise ValueError("Reference use is not eligible")
