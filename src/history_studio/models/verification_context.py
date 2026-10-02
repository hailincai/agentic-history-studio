"""Deterministic claim-bounded input projection, without verification or prompt generation."""
from typing import Literal, Self

from pydantic import Field, model_validator

from .base import Contract, require_unique
from .research import ResearchFact
from .research_package import ResearchPackage
from .sources import SourceReference


class VerificationContext(Contract):
    """Original research is inspectable input, not independent verification evidence.

    Original evidence is composed inside target_fact, without a duplicate evidence list.
    This is a one-fact projection, not a byte/token budget or an investigation result.
    """

    target_fact: ResearchFact
    sources: list[SourceReference] = Field(default_factory=list)
    investigation_boundary: Literal["TARGET_CLAIM_ONLY"] = "TARGET_CLAIM_ONLY"
    original_evidence_role: Literal["RESEARCH_INPUT_NOT_INDEPENDENT_VERIFICATION"] = (
        "RESEARCH_INPUT_NOT_INDEPENDENT_VERIFICATION")
    whole_topic_research_allowed: Literal[False] = False

    @model_validator(mode="after")
    def required_source_metadata(self) -> Self:
        require_unique([source.source_id for source in self.sources], "Context source IDs")
        required = {e.source_id for e in self.target_fact.evidence}
        included = {s.source_id for s in self.sources}
        if required - included:
            raise ValueError("Missing source metadata for target research evidence")
        if included - required:
            raise ValueError("Context sources must be referenced by target research evidence")
        return self


def build_verification_context(package: ResearchPackage, research_fact_id: str) -> VerificationContext:
    """Select one fact and its evidence sources in package order, with detached input data.

    Check lookup integrity even if a caller mutated a previously validated package.
    No original evidence is converted into VerificationEvidence or accepted as independent.
    """
    matches = [fact for fact in package.facts if fact.fact_id == research_fact_id]
    if not matches:
        raise ValueError("Target research_fact_id not found in ResearchPackage")
    if len(matches) != 1:
        raise ValueError("Target research_fact_id is not unique in ResearchPackage")
    target = matches[0]
    required = {e.source_id for e in target.evidence}
    sources = [source for source in package.sources if source.source_id in required]
    # Reparse serialized values so nested lists/models do not alias durable package objects.
    return VerificationContext.model_validate({
        "target_fact": target.model_dump(mode="json"),
        "sources": [source.model_dump(mode="json") for source in sources],
    })
