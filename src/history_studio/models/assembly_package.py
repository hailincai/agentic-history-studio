"""Minimal immutable publication identity of a completed documentary."""
from typing import Literal, Self

from pydantic import ConfigDict, Field, model_validator

from .artifact_reference import ArtifactReference
from .base import Contract, PositiveSeconds, Text
from .media_package import MediaAssetReference, MediaType


class AssemblyPackage(Contract):
    model_config = ConfigDict(frozen=True)

    schema_version: Literal[1] = 1
    media_input_ref: ArtifactReference
    final_video: MediaAssetReference
    subtitles: MediaAssetReference
    timeline_duration_ms: int = Field(strict=True, gt=0)
    measured_duration_seconds: PositiveSeconds
    ffmpeg_version: Text | None = None

    @model_validator(mode="after")
    def publication_shape(self) -> Self:
        if self.media_input_ref.artifact_type != "media":
            raise ValueError("Assembly input must identify an exact media artifact")
        if self.final_video.media_type != MediaType.VIDEO or not self.final_video.relative_path.lower().endswith(".mp4"):
            raise ValueError("Assembly final video must be a VIDEO MP4")
        if self.subtitles.media_type != MediaType.SUBTITLE or not self.subtitles.relative_path.lower().endswith(".srt"):
            raise ValueError("Assembly subtitles must be an independent SUBTITLE SRT")
        if (self.final_video.asset_id == self.subtitles.asset_id
                or self.final_video.relative_path == self.subtitles.relative_path):
            raise ValueError("Assembly outputs must have distinct asset identities and paths")
        return self
