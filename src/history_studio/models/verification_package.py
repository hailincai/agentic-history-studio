"""Accepted per-fact knowledge for one exact research snapshot, without execution state."""
from typing import Literal, Self

from pydantic import ConfigDict, Field, field_validator, model_validator

from .artifact_reference import ArtifactReference
from .base import Contract, Identifier, require_unique
from .research_package import ResearchPackage
from .verification import VerificationResult


class ResearchFactSnapshot(Contract):
    """Compact ordered membership: fact identity and exact assertion, without source content."""

    model_config = ConfigDict(frozen=True)

    research_fact_id: Identifier
    claim_snapshot: str = Field(min_length=1, max_length=600)

    @field_validator("claim_snapshot")
    @classmethod
    def nonblank_claim(cls, value: str) -> str:
        if not value.strip():
            raise ValueError("Claim snapshot must not be blank")
        return value


class VerificationPackage(Contract):
    """Zero or more accepted results, permanently bound to captured research membership.

    Tuple membership/results and frozen fields support copy-on-update. Nested result models
    retain their existing contracts; update APIs reparse and detach all nested values.
    ArtifactStore versions, rather than a separate package hash, identify durable packages.
    """

    model_config = ConfigDict(frozen=True)

    schema_version: Literal[1] = 1
    research_input_ref: ArtifactReference
    research_facts: tuple[ResearchFactSnapshot, ...]
    results: tuple[VerificationResult, ...] = Field(default_factory=tuple)

    @model_validator(mode="after")
    def result_membership(self) -> Self:
        if self.research_input_ref.artifact_type != "research":
            raise ValueError("Verification package input must reference a research artifact")
        require_unique([fact.research_fact_id for fact in self.research_facts], "Research membership IDs")
        require_unique([result.research_fact_id for result in self.results], "Accepted result fact IDs")
        claims = {fact.research_fact_id: fact.claim_snapshot for fact in self.research_facts}
        for result in self.results:
            if result.research_input_ref != self.research_input_ref:
                raise ValueError("Result research snapshot must match the verification package")
            if result.research_fact_id not in claims:
                raise ValueError("Result fact must belong to captured research membership")
            if result.claim_snapshot != claims[result.research_fact_id]:
                raise ValueError("Result claim must exactly match captured research claim")
        return self

    @property
    def completed_fact_ids(self) -> list[str]:
        completed = {result.research_fact_id for result in self.results}
        return [fact.research_fact_id for fact in self.research_facts if fact.research_fact_id in completed]

    @property
    def pending_fact_ids(self) -> list[str]:
        completed = {result.research_fact_id for result in self.results}
        return [fact.research_fact_id for fact in self.research_facts if fact.research_fact_id not in completed]

    @property
    def is_complete(self) -> bool:
        """All captured facts have a result; empty membership has no pending work."""
        return not self.pending_fact_ids


def create_verification_package(research_package: ResearchPackage, *,
                                research_input_ref: ArtifactReference) -> VerificationPackage:
    """Capture ordered membership from the version explicitly selected by Runtime.

    Runtime supplies the reference of its loaded artifact. This function cannot infer a
    version from content and neither loads nor writes artifacts.
    """
    research = ResearchPackage.model_validate(research_package.model_dump(mode="json"))
    reference = ArtifactReference.model_validate(research_input_ref.model_dump(mode="json"))
    if reference.project_id != research.project_id:
        raise ValueError("Research snapshot project_id must match ResearchPackage")
    return VerificationPackage.model_validate({
        "research_input_ref": reference.model_dump(mode="json"),
        "research_facts": [{"research_fact_id": fact.fact_id, "claim_snapshot": fact.claim}
                           for fact in research.facts],
    })


def add_verification_result(package: VerificationPackage, result: VerificationResult) -> VerificationPackage:
    """Return detached accepted knowledge, rejecting duplicate or mismatched results.

    Reparse the complete payload so mutated nested values cannot bypass validation.
    No semantic reinterpretation, investigation, provider/tool calls, or persistence occurs.
    """
    data = package.model_dump(mode="json")
    data["results"].append(result.model_dump(mode="json"))
    return VerificationPackage.model_validate(data)
