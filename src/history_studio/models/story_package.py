"""Grounded Story Architect output, separate from the legacy duration outline.

No package loading, approval, execution, or semantic prose validation occurs here.
"""
from enum import StrEnum
from typing import Literal, Self

from pydantic import Field, model_validator

from .artifact_reference import ArtifactReference
from .base import Contract, Identifier, Text, require_unique
from .historical_time import HistoricalTime
from .verification import VerificationStatus


class StoryFactUse(StrEnum):
    AFFIRMATIVE = "AFFIRMATIVE"
    QUALIFIED = "QUALIFIED"
    DISPUTE = "DISPUTE"


class StoryFactReference(Contract):
    """Stable ResearchFact identity with a declared, unaltered source verdict.

    Later runtime must resolve this ID in verification_input_ref, check exact status
    equality, and assess prose against claim_snapshot, rationale and unresolved_issues.
    A qualification's presence cannot prove its semantic adequacy.
    REJECTED and UNVERIFIED cannot ground historical narrative in this contract.
    """

    research_fact_id: Identifier
    status: VerificationStatus
    use: StoryFactUse
    qualification: Text | None = None

    @model_validator(mode="after")
    def status_policy(self) -> Self:
        allowed = {
            VerificationStatus.VERIFIED: {StoryFactUse.AFFIRMATIVE, StoryFactUse.QUALIFIED},
            VerificationStatus.PARTIALLY_VERIFIED: {StoryFactUse.QUALIFIED},
            VerificationStatus.DISPUTED: {StoryFactUse.DISPUTE},
            VerificationStatus.REJECTED: set(),
            VerificationStatus.UNVERIFIED: set(),
        }
        if self.use not in allowed[self.status]:
            raise ValueError("Narrative use is not eligible for this verification status")
        if self.use in (StoryFactUse.QUALIFIED, StoryFactUse.DISPUTE) and self.qualification is None:
            raise ValueError("Qualified or disputed use requires an explicit qualification")
        return self


class StoryNarrativeBeat(Contract):
    """Historical prose requires references; structural prose makes no historical claims.

    Structural beats can describe an opening/transition/conclusion, but cannot hide
    historical substance. Later semantic review must also ground titles, thesis,
    purposes and any other prose containing historical assertions.
    Chronology is separate from prose and must later be checked against source facts.
    UNKNOWN/APPROXIMATE dates remain expressible; sequence is never a date sort key.
    """

    beat_id: Identifier
    kind: Literal["historical", "structural"] = "historical"
    narrative_role: Text
    summary: Text
    fact_refs: tuple[StoryFactReference, ...] = Field(default_factory=tuple)
    historical_time: HistoricalTime | None = None
    uncertainty_notes: tuple[Text, ...] = Field(default_factory=tuple)

    @model_validator(mode="after")
    def grounding_shape(self) -> Self:
        require_unique([ref.research_fact_id for ref in self.fact_refs], "Beat fact references")
        if self.kind == "historical":
            if not self.fact_refs:
                raise ValueError("Historical beat requires fact grounding")
            if self.historical_time is None:
                raise ValueError("Historical beat requires chronology metadata")
        elif self.fact_refs or self.historical_time is not None:
            raise ValueError("Structural beat cannot carry historical grounding or chronology")
        return self


class StorySection(Contract):
    section_id: Identifier
    purpose: Text
    beats: tuple[StoryNarrativeBeat, ...] = Field(min_length=1)

    @model_validator(mode="after")
    def unique_beats(self) -> Self:
        require_unique([beat.beat_id for beat in self.beats], "Beat IDs")
        return self


class StoryStructure(Contract):
    """Sections and beats are in intended strict chronological narrative order.

    List position specifies presentation order, not proof of chronology. Later runtime
    must validate chronological relations, retaining ambiguous/overlapping dates for
    resolution rather than arbitrarily sorting their numeric bounds.
    IDs are explicitly supplied stable identities (e.g. section_01, beat_01), not
    hashes of mutable prose or randomly generated values.
    """

    title: Text
    narrative_thesis: Text
    sections: tuple[StorySection, ...] = Field(min_length=1)

    @model_validator(mode="after")
    def unique_membership(self) -> Self:
        require_unique([section.section_id for section in self.sections], "Section IDs")
        require_unique([beat.beat_id for section in self.sections for beat in section.beats], "Beat IDs")
        return self


class StoryPackage(Contract):
    """Durable output from exactly one human-approved verification snapshot.

    Runtime must supply approved_verification as verification_input_ref and later
    prove approval, project lineage, fact membership, verdict equality, grounding,
    uncertainty preservation and chronology. Pydantic proves only structural shape.
    schema_version is a format version, never an ArtifactStore snapshot version.
    No workflow binding or authorization state is stored here.
    """

    schema_version: Literal[1] = 1
    verification_input_ref: ArtifactReference
    plan: StoryStructure

    @model_validator(mode="after")
    def verification_provenance(self) -> Self:
        if self.verification_input_ref.artifact_type != "verification":
            raise ValueError("Story input must reference a verification artifact")
        return self
