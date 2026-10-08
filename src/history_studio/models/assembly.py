"""Immutable assembly plans contain identities and integer timing only."""
from typing import Annotated, Literal, Self

from pydantic import ConfigDict, Field, model_validator

from .artifact_reference import ArtifactReference
from .base import Contract, Identifier, require_unique
from .media_package import MediaAssetReference, MediaType

Milliseconds = Annotated[int, Field(strict=True, ge=0)]


class Interval(Contract):
    model_config = ConfigDict(frozen=True)

    start_ms: Milliseconds
    end_ms: Milliseconds

    @model_validator(mode="after")
    def positive_interval(self) -> Self:
        if self.end_ms <= self.start_ms:
            raise ValueError("Timing interval must have positive duration")
        return self


class SegmentTiming(Interval):
    segment_id: Identifier
    narration_asset: MediaAssetReference

    @model_validator(mode="after")
    def audio_identity(self) -> Self:
        if self.narration_asset.media_type != MediaType.AUDIO:
            raise ValueError("Segment narration must be AUDIO")
        return self


class ShotTiming(Interval):
    shot_id: Identifier
    source_segment_id: Identifier
    visual_asset: MediaAssetReference

    @model_validator(mode="after")
    def visual_identity(self) -> Self:
        if self.visual_asset.media_type not in {MediaType.IMAGE, MediaType.VIDEO}:
            raise ValueError("Shot visual must be IMAGE or VIDEO")
        return self


class SubtitleCue(Interval):
    index: int = Field(strict=True, gt=0)
    text: str

    @model_validator(mode="after")
    def nonempty_text(self) -> Self:
        if not self.text.strip():
            raise ValueError("Subtitle narration must not be empty")
        if any(ord(char) < 32 and char not in "\r\n\t" for char in self.text):
            raise ValueError("Subtitle narration contains unsupported control characters")
        return self


class TimelinePlan(Contract):
    model_config = ConfigDict(frozen=True)

    schema_version: Literal[1] = 1
    media_input_ref: ArtifactReference
    storyboard_input_ref: ArtifactReference
    script_input_ref: ArtifactReference
    segments: tuple[SegmentTiming, ...] = Field(min_length=1)
    shots: tuple[ShotTiming, ...] = Field(min_length=1)
    total_duration_ms: int = Field(strict=True, gt=0)

    @model_validator(mode="after")
    def conserved_timeline(self) -> Self:
        refs = (self.media_input_ref, self.storyboard_input_ref, self.script_input_ref)
        if tuple(ref.artifact_type for ref in refs) != ("media", "storyboard", "script"):
            raise ValueError("Timeline requires exact media, storyboard and script references")
        if len({ref.project_id for ref in refs}) != 1:
            raise ValueError("Timeline references must belong to one project")
        require_unique([segment.segment_id for segment in self.segments], "Timeline segment IDs")
        require_unique([shot.shot_id for shot in self.shots], "Timeline shot IDs")
        require_unique([segment.narration_asset.asset_id for segment in self.segments]
                       + [shot.visual_asset.asset_id for shot in self.shots], "Timeline asset IDs")
        offset = 0
        remaining = iter(self.shots)
        shot = next(remaining, None)
        for segment in self.segments:
            if segment.start_ms != offset:
                raise ValueError("Segments must be contiguous from zero")
            count = 0
            while shot is not None and shot.source_segment_id == segment.segment_id:
                if shot.start_ms != offset or shot.end_ms > segment.end_ms:
                    raise ValueError("Shots must be contiguous within their segment")
                offset = shot.end_ms
                count += 1
                shot = next(remaining, None)
            if not count or offset != segment.end_ms:
                raise ValueError("Shots must exactly cover every segment")
        if shot is not None or offset != self.total_duration_ms:
            raise ValueError("Timeline total and shot coverage must match segments")
        return self
