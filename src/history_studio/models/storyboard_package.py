"""Durable visual planning contracts; validation proves structural shape only."""
from enum import StrEnum
from typing import Literal, Self

from pydantic import Field, model_validator

from .artifact_reference import ArtifactReference
from .base import Contract, Identifier, PositiveSeconds, Text, require_unique
from .storyboard import GenerationMethod


class StoryboardShotKind(StrEnum):
    """Declared visual role, not semantic classification of creative prose."""

    HISTORICAL = "HISTORICAL"
    STRUCTURAL = "STRUCTURAL"


class ShotFraming(StrEnum):
    EXTREME_WIDE = "EXTREME_WIDE"
    WIDE = "WIDE"
    MEDIUM = "MEDIUM"
    CLOSE_UP = "CLOSE_UP"
    EXTREME_CLOSE_UP = "EXTREME_CLOSE_UP"


class CameraMotion(StrEnum):
    NONE = "NONE"
    PAN = "PAN"
    TILT = "TILT"
    PUSH_IN = "PUSH_IN"
    PULL_OUT = "PULL_OUT"
    TRACKING = "TRACKING"


class StoryboardShot(Contract):
    """One visual belongs to one ScriptSegment; a segment may have many shots.

    The segment is the immediate historical authority boundary. Its existence and
    kind compatibility require later authentication against the exact Script.
    Creative descriptions and prompts are not historical authority. Duration is
    a planning estimate, not actual TTS/audio timing.
    """

    shot_id: Identifier
    kind: StoryboardShotKind
    source_segment_id: Identifier
    visual_description: Text
    generation_prompt: Text
    generation_method: GenerationMethod
    framing: ShotFraming
    camera_motion: CameraMotion
    estimated_duration_seconds: PositiveSeconds


class StoryboardSection(Contract):
    section_id: Identifier
    title: Text
    shots: tuple[StoryboardShot, ...] = Field(min_length=1)

    @model_validator(mode="after")
    def unique_shots(self) -> Self:
        require_unique([shot.shot_id for shot in self.shots], "Shot IDs")
        return self


class StoryboardPackage(Contract):
    """Durable visual output derived from exactly one approved Script snapshot.

    Collection position is presentation order. schema_version identifies the
    format, independently of the referenced artifact version. Later Runtime must
    authenticate approval, project lineage, source membership and kind matching.
    Prose truth, prompt grounding, visual chronology and narration timing are not
    proved here. No upstream authority or workflow state is duplicated.
    """

    schema_version: Literal[1] = 1
    script_input_ref: ArtifactReference
    title: Text
    sections: tuple[StoryboardSection, ...] = Field(min_length=1)

    @model_validator(mode="after")
    def script_provenance_and_unique_membership(self) -> Self:
        if self.script_input_ref.artifact_type != "script":
            raise ValueError("Storyboard input must reference a script artifact")
        require_unique([section.section_id for section in self.sections], "Section IDs")
        require_unique([shot.shot_id for section in self.sections
                        for shot in section.shots], "Shot IDs")
        return self
