"""Untrusted visual proposals and pure Runtime structural finalization."""
from typing import Self

from pydantic import Field, model_validator

from history_studio.models.artifact_reference import ArtifactReference
from history_studio.models.base import Contract, Text, require_unique
from history_studio.models.storyboard_package import (
    StoryboardPackage, StoryboardSection, StoryboardShot,
)
from history_studio.models.visual_director_context import VisualDirectorContext


class GenerationMethodNotAllowedError(ValueError):
    """Safe Runtime policy diagnostic, distinct from provider exception text."""

    def __init__(self, message: str, *, shot_id: str, source_segment_id: str) -> None:
        super().__init__(message)
        self.shot_id = shot_id
        self.source_segment_id = source_segment_id


class StoryboardShotSubmission(StoryboardShot):
    """Agent-owned shot fields; local shape does not authenticate source authority."""


class StoryboardSectionSubmission(StoryboardSection):
    """Select an approved section; proposed title is replaced by its canonical title."""

    shots: tuple[StoryboardShotSubmission, ...] = Field(min_length=1)


class StoryboardSubmission(Contract):
    """Untrusted proposal without provenance or duplicated historical authority.

    Title fields are required proposal text, but Runtime uses context titles.
    """

    title: Text
    sections: tuple[StoryboardSectionSubmission, ...] = Field(min_length=1)

    @model_validator(mode="after")
    def unique_membership(self) -> Self:
        require_unique([section.section_id for section in self.sections], "Section IDs")
        require_unique([shot.shot_id for section in self.sections
                        for shot in section.shots], "Shot IDs")
        return self


def finalize_storyboard_submission(context: VisualDirectorContext,
                                   submission: StoryboardSubmission) -> StoryboardPackage:
    """Authenticate complete Script coverage, ownership, kind and presentation order.

    Runtime supplies an approved, authenticated context. Revalidation proves shape,
    not approval or origin. Titles and exact provenance are canonical context data;
    shot fields and within-segment shot order are preserved without semantic judgment.
    No prose repair, artifact access, persistence or timing computation occurs.
    """
    if not isinstance(context, VisualDirectorContext) or not isinstance(submission, StoryboardSubmission):
        raise TypeError("Finalization requires VisualDirectorContext and StoryboardSubmission")
    # Reconstruct raw reference fields before serialization can coerce invalid copies.
    reference = ArtifactReference(project_id=context.script_input_ref.project_id,
        artifact_type=context.script_input_ref.artifact_type, version=context.script_input_ref.version)
    validated = VisualDirectorContext.model_validate(context.model_dump(mode="json") | {
        "script_input_ref": reference.model_dump(mode="json")})
    proposal = StoryboardSubmission.model_validate(submission.model_dump(mode="json"))
    authorized_ids = [section.section_id for section in validated.sections]
    proposed_ids = [section.section_id for section in proposal.sections]
    if any(section_id not in authorized_ids for section_id in proposed_ids):
        raise ValueError("Submitted section must exist in approved Script context")
    if set(proposed_ids) != set(authorized_ids):
        raise ValueError("Submitted sections must cover every approved Script section")
    if proposed_ids != authorized_ids:
        raise ValueError("Submitted sections must preserve approved Script section order")
    sections = []
    for section, source in zip(proposal.sections, validated.sections):
        segments = {segment.segment_id: (position, segment)
                    for position, segment in enumerate(source.segments)}
        previous_position = -1
        covered = set()
        for shot in section.shots:
            if validated.production_brief is not None and shot.generation_method not in validated.production_brief.allowed_generation_methods:
                raise GenerationMethodNotAllowedError(
                    f"Shot {shot.shot_id} generation method {shot.generation_method.value} violates "
                    "production brief allowed_generation_methods: " + ", ".join(
                        method.value for method in validated.production_brief.allowed_generation_methods),
                    shot_id=shot.shot_id, source_segment_id=shot.source_segment_id)
            selected = segments.get(shot.source_segment_id)
            if selected is None:
                raise ValueError("Shot source segment must exist in its submitted Script section")
            position, segment = selected
            if shot.kind.value != segment.kind.value:
                raise ValueError("Shot kind must match its approved Script segment kind")
            if position < previous_position:
                raise ValueError("Shots must preserve approved Script segment order")
            previous_position = position
            covered.add(shot.source_segment_id)
        if covered != set(segments):
            raise ValueError("Shots must cover every approved Script segment")
        sections.append(section.model_dump(mode="json") | {"title": source.title})
    return StoryboardPackage.model_validate(dict(
        script_input_ref=validated.script_input_ref.model_dump(mode="json"),
        title=validated.title, sections=sections))
