"""Agent-owned narrative proposals and pure Runtime structural acceptance.

No prose judgment, approval, persistence, model requests or tool execution occurs.
"""
from typing import Literal, Self

from pydantic import Field, model_validator

from history_studio.models.base import Contract, Identifier, Text, require_unique
from history_studio.models.story_context import StoryContext
from history_studio.models.story_package import StoryFactReference, StoryFactUse, StoryPackage


class StoryFactProposal(Contract):
    """Select identity/use only; status and chronology cannot be supplied by the Agent."""

    research_fact_id: Identifier
    use: StoryFactUse
    qualification: Text | None = None


class StoryBeatProposal(Contract):
    beat_id: Identifier
    kind: Literal["historical", "structural"] = "historical"
    narrative_role: Text
    summary: Text
    fact_proposals: tuple[StoryFactProposal, ...] = Field(default_factory=tuple)
    uncertainty_notes: tuple[Text, ...] = Field(default_factory=tuple)

    @model_validator(mode="after")
    def grounding_shape(self) -> Self:
        require_unique([fact.research_fact_id for fact in self.fact_proposals], "Beat fact proposals")
        if self.kind == "historical" and not self.fact_proposals:
            raise ValueError("Historical beat requires fact grounding")
        if self.kind == "structural" and self.fact_proposals:
            raise ValueError("Structural beat cannot carry fact proposals")
        return self


class StorySectionProposal(Contract):
    section_id: Identifier
    purpose: Text
    beats: tuple[StoryBeatProposal, ...] = Field(min_length=1)

    @model_validator(mode="after")
    def unique_beats(self) -> Self:
        require_unique([beat.beat_id for beat in self.beats], "Beat IDs")
        return self


class StorySubmission(Contract):
    """Unaccepted semantic proposal, not a durable StoryPackage or approval record."""

    title: Text
    narrative_thesis: Text
    sections: tuple[StorySectionProposal, ...] = Field(min_length=1)

    @model_validator(mode="after")
    def unique_membership(self) -> Self:
        require_unique([section.section_id for section in self.sections], "Section IDs")
        require_unique([beat.beat_id for section in self.sections for beat in section.beats], "Beat IDs")
        return self


def finalize_story_submission(context: StoryContext, submission: StorySubmission) -> StoryPackage:
    """Resolve eligible identity/status/time and bind exact context provenance.

    Runtime must supply the approved, lineage-authenticated P4-B context. This pure
    boundary revalidates and detaches both inputs; it cannot itself prove approval
    or source authenticity. Agent order is preserved, including ambiguous chronology.
    Qualification presence is structural, not proof of semantic adequacy. Unsupported
    motives, emotion, dialogue, scenes, thesis claims, hidden structural assertions
    and historical ordering correctness require later semantic review/Human Gate.
    """
    if not isinstance(context, StoryContext) or not isinstance(submission, StorySubmission):
        raise TypeError("Finalization requires StoryContext and StorySubmission")
    validated = StoryContext.model_validate(context.model_dump(mode="json"))
    proposal = StorySubmission.model_validate(submission.model_dump(mode="json"))
    eligible = {fact.research_fact_id: fact for fact in validated.eligible_facts}
    sections = []
    for section in proposal.sections:
        beats = []
        for beat in section.beats:
            references, chronology = [], []
            for selection in beat.fact_proposals:
                fact = eligible.get(selection.research_fact_id)
                if fact is None:
                    raise ValueError("Proposed fact must exist in narratively eligible context knowledge")
                # P4-A is the canonical status/use/qualification structural policy.
                reference = StoryFactReference(research_fact_id=fact.research_fact_id,
                    status=fact.status, use=selection.use, qualification=selection.qualification)
                references.append(reference.model_dump(mode="json"))
                chronology.append(dict(research_fact_id=fact.research_fact_id,
                                       historical_time=fact.historical_time.model_dump(mode="json")))
            beats.append(beat.model_dump(mode="json", exclude={"fact_proposals"}) | {
                "fact_refs": references, "fact_chronology": chronology})
        sections.append(dict(section_id=section.section_id, purpose=section.purpose, beats=beats))
    return StoryPackage.model_validate(dict(
        verification_input_ref=validated.verification_input_ref.model_dump(mode="json"),
        plan=dict(title=proposal.title, narrative_thesis=proposal.narrative_thesis, sections=sections)))
