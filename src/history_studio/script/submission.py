"""Untrusted Script proposals and pure Runtime identity/authorization finalization."""
from typing import Self

from pydantic import Field, model_validator

from history_studio.models.artifact_reference import ArtifactReference
from history_studio.models.base import Contract, Identifier, Text, require_unique
from history_studio.models.script_context import ScriptContext
from history_studio.models.script_package import ScriptPackage, ScriptSegmentKind


class ScriptGroundingSubmission(Contract):
    """Agent selects IDs only; these do not authenticate upstream authority."""

    story_beat_id: Identifier
    research_fact_ids: tuple[Identifier, ...] = Field(min_length=1)

    @model_validator(mode="after")
    def unique_facts(self) -> Self:
        require_unique(list(self.research_fact_ids), "Selected fact IDs")
        return self


class ScriptSegmentSubmission(Contract):
    segment_id: Identifier
    kind: ScriptSegmentKind
    narration: Text
    grounding: ScriptGroundingSubmission | None = None

    @model_validator(mode="after")
    def grounding_shape(self) -> Self:
        if self.kind == ScriptSegmentKind.HISTORICAL and self.grounding is None:
            raise ValueError("Historical segment requires grounding")
        if self.kind == ScriptSegmentKind.STRUCTURAL and self.grounding is not None:
            raise ValueError("Structural segment cannot carry grounding")
        return self


class ScriptSectionSubmission(Contract):
    """section_id selects an approved Story section; title is audience-facing wording."""

    section_id: Identifier
    title: Text
    segments: tuple[ScriptSegmentSubmission, ...] = Field(min_length=1)

    @model_validator(mode="after")
    def unique_segments(self) -> Self:
        require_unique([segment.segment_id for segment in self.segments], "Segment IDs")
        return self


class ScriptSubmission(Contract):
    """Semantic proposal only, without provenance, verdicts, chronology or approval."""

    title: Text
    sections: tuple[ScriptSectionSubmission, ...] = Field(min_length=1)

    @model_validator(mode="after")
    def unique_membership(self) -> Self:
        require_unique([section.section_id for section in self.sections], "Section IDs")
        require_unique([segment.segment_id for section in self.sections
                        for segment in section.segments], "Segment IDs")
        return self


def finalize_script_submission(context: ScriptContext, submission: ScriptSubmission) -> ScriptPackage:
    """Authenticate selected section/beat/fact identities and bind exact provenance.

    Runtime must supply an approved, authenticated context. Revalidation establishes
    shape, not approval or origin. Sections may be omitted but not reordered; historical
    beat positions within each section must be nondecreasing, allowing several segments
    for one beat and omission of beats/facts. Structural segments carry no beat identity
    under P5-A; their containing section is authenticated, without inferring a beat.
    No prose/qualification/chronology semantics are judged or repaired here. Human
    semantic acceptance, generated-Script validation and coverage policy remain deferred.
    """
    if not isinstance(context, ScriptContext) or not isinstance(submission, ScriptSubmission):
        raise TypeError("Finalization requires ScriptContext and ScriptSubmission")
    # Preserve strict version validation even for deliberately unvalidated copies.
    reference = ArtifactReference(project_id=context.story_input_ref.project_id,
        artifact_type=context.story_input_ref.artifact_type, version=context.story_input_ref.version)
    validated = ScriptContext.model_validate(context.model_dump(mode="json") | {
        "story_input_ref": reference.model_dump(mode="json")})
    proposal = ScriptSubmission.model_validate(submission.model_dump(mode="json"))
    sections = {section.section_id: (position, section)
                for position, section in enumerate(validated.sections)}
    previous_section = -1
    for section in proposal.sections:
        selected_section = sections.get(section.section_id)
        if selected_section is None:
            raise ValueError("Submitted section must exist in approved Story context")
        position, source = selected_section
        if position <= previous_section:
            raise ValueError("Submitted sections must preserve approved Story section order")
        previous_section = position
        beats = {beat.beat_id: (index, beat) for index, beat in enumerate(source.beats)}
        previous_beat = -1
        for segment in section.segments:
            if segment.kind == ScriptSegmentKind.STRUCTURAL:
                continue
            selection = segment.grounding
            selected_beat = beats.get(selection.story_beat_id)
            if selected_beat is None:
                raise ValueError("Selected beat must exist in the submitted Story section")
            beat_position, beat = selected_beat
            if beat.kind != "historical":
                raise ValueError("Historical segment must target a historical Story beat")
            if beat_position < previous_beat:
                raise ValueError("Historical segments must preserve approved Story beat order")
            previous_beat = beat_position
            authorized = {fact.research_fact_id for fact in beat.fact_refs}
            if any(fact_id not in authorized for fact_id in selection.research_fact_ids):
                raise ValueError("Selected fact must be authorized by the exact Story beat")
    # The validated proposal has exactly the output semantic shape. Reparse into the
    # existing durable types; never expand selections or rewrite normalized narration.
    return ScriptPackage.model_validate(proposal.model_dump(mode="json") | {
        "story_input_ref": validated.story_input_ref.model_dump(mode="json")})
