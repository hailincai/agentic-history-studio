"""Durable Script narration contracts; validation proves structural shape only."""
from enum import StrEnum
from typing import Literal, Self

from pydantic import Field, model_validator

from .artifact_reference import ArtifactReference
from .base import Contract, Identifier, Text, require_unique


class ScriptSegmentKind(StrEnum):
    """Declared narration role, not semantic classification of its text."""

    HISTORICAL = "HISTORICAL"
    STRUCTURAL = "STRUCTURAL"


class ScriptGrounding(Contract):
    """Selected fact identities for one Story beat, without authoritative snapshots.

    Later Runtime must authenticate the beat and its selected facts in the exact
    approved Story. This contract does not load artifacts or prove membership.
    """

    story_beat_id: Identifier
    research_fact_ids: tuple[Identifier, ...] = Field(min_length=1)

    @model_validator(mode="after")
    def unique_facts(self) -> Self:
        require_unique(list(self.research_fact_ids), "Grounding fact IDs")
        return self


class ScriptSegment(Contract):
    segment_id: Identifier
    kind: ScriptSegmentKind
    narration: Text
    grounding: ScriptGrounding | None = None

    @model_validator(mode="after")
    def grounding_shape(self) -> Self:
        if self.kind == ScriptSegmentKind.HISTORICAL and self.grounding is None:
            raise ValueError("Historical segment requires grounding")
        if self.kind == ScriptSegmentKind.STRUCTURAL and self.grounding is not None:
            raise ValueError("Structural segment cannot carry grounding")
        return self


class ScriptSection(Contract):
    section_id: Identifier
    title: Text
    segments: tuple[ScriptSegment, ...] = Field(min_length=1)

    @model_validator(mode="after")
    def unique_segments(self) -> Self:
        require_unique([segment.segment_id for segment in self.segments], "Segment IDs")
        return self


class ScriptPackage(Contract):
    """Durable narration derived from exactly one approved Story snapshot.

    Runtime must supply the exact approved Story reference and later authenticate
    approval, project lineage, beat/fact membership and compatible fact usage.
    Prose truth, grounding adequacy, qualification, chronology, style and duration
    are not proved here. schema_version is format identity, not artifact version.
    """

    schema_version: Literal[1] = 1
    story_input_ref: ArtifactReference
    title: Text
    sections: tuple[ScriptSection, ...] = Field(min_length=1)

    @model_validator(mode="after")
    def story_provenance_and_unique_membership(self) -> Self:
        if self.story_input_ref.artifact_type != "story":
            raise ValueError("Script input must reference a story artifact")
        require_unique([section.section_id for section in self.sections], "Section IDs")
        require_unique([segment.segment_id for section in self.sections
                        for segment in section.segments], "Segment IDs")
        return self
