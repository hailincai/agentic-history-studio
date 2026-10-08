"""Bounded, detached Visual Director input; not a durable output artifact."""
from typing import Self

from pydantic import Field, model_validator

from .artifact_reference import ArtifactReference
from .base import Contract, Text, require_unique
from .script_package import ScriptSection, ScriptSegment
from .production_brief import ProductionBrief


class VisualDirectorContextSegment(ScriptSegment):
    """Approved narration and grounding IDs, without independent historical facts.

    Reuses Script's kind/grounding shape. IDs identify the approved boundary; they
    do not authorize research or expansion of the narration. The builder detaches
    grounding values from the source artifact.
    """


class VisualDirectorContextSection(ScriptSection):
    """Approved section identity and title, preserving segment presentation order."""

    segments: tuple[VisualDirectorContextSegment, ...] = Field(min_length=1)


class VisualDirectorContext(Contract):
    """Working projection of one exact Runtime-selected approved Script.

    Approval is a caller precondition, not proved here. Collection order is the
    approved presentation order. This bounds semantic authority rather than token
    count: narration is complete, without upstream artifacts or durable envelope.
    Standalone validation proves shape, not authenticity or historical truth.
    """

    script_input_ref: ArtifactReference
    production_brief: ProductionBrief | None = None
    title: Text
    sections: tuple[VisualDirectorContextSection, ...] = Field(min_length=1)

    @model_validator(mode="after")
    def script_provenance_and_unique_membership(self) -> Self:
        if self.script_input_ref.artifact_type != "script":
            raise ValueError("Visual Director input must reference a script artifact")
        require_unique([section.section_id for section in self.sections], "Section IDs")
        require_unique([segment.segment_id for section in self.sections
                        for segment in section.segments], "Segment IDs")
        return self
