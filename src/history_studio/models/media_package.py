"""Durable media manifests; binary storage and authority checks belong to Runtime."""
from enum import StrEnum
from typing import Annotated, Literal, Self

from pydantic import ConfigDict, StringConstraints, field_validator, model_validator

from .artifact_reference import ArtifactReference
from .base import Contract, Identifier, PositiveSeconds, Text, require_unique
from .storyboard import GenerationMethod


class MediaType(StrEnum):
    AUDIO = "AUDIO"
    IMAGE = "IMAGE"
    VIDEO = "VIDEO"
    SUBTITLE = "SUBTITLE"


class MediaAssetReference(Contract):
    """Portable binary identity, without bytes, filesystem access or hash computation.

    Paths use POSIX separators relative to the project/media root. SHA-256 is
    lowercase hexadecimal. Existence and digest verification are runtime concerns.
    """

    model_config = ConfigDict(frozen=True)

    asset_id: Identifier
    relative_path: str
    media_type: MediaType
    sha256: Annotated[str, StringConstraints(pattern=r"^[0-9a-f]{64}$")]

    @field_validator("relative_path")
    @classmethod
    def safe_relative_path(cls, value: str) -> str:
        parts = value.split("/")
        reserved = {"CON", "PRN", "AUX", "NUL", *(f"COM{i}" for i in range(1, 10)),
                    *(f"LPT{i}" for i in range(1, 10))}
        if (any(character in value for character in '\\:<>"|?*')
                or any(ord(character) < 32 or ord(character) == 127 for character in value)
                or any(part in {"", ".", ".."} or part != part.strip()
                       or part.endswith(".") or part.split(".")[0].upper() in reserved
                       for part in parts)):
            raise ValueError("Media path must be a safe POSIX relative path")
        return value


class GenerationMetadata(Contract):
    """Optional durable generator identity; excludes requests, secrets and responses."""

    model_config = ConfigDict(frozen=True)

    provider: Text
    model: Text


class NarrationAsset(Contract):
    """One Script segment's audio with actual measured generated duration.

    StoryboardShot.estimated_duration_seconds is only a planning estimate.
    This duration does not allocate shot timing; that belongs to Phase 8.
    """

    model_config = ConfigDict(frozen=True)

    segment_id: Identifier
    asset: MediaAssetReference
    duration_seconds: PositiveSeconds
    generation_metadata: GenerationMetadata | None = None

    @model_validator(mode="after")
    def audio_asset(self) -> Self:
        if self.asset.media_type != MediaType.AUDIO:
            raise ValueError("Narration asset must be AUDIO")
        return self


class VisualAsset(Contract):
    """One shot's visual output; creative and historical authority stays in Storyboard.

    Source membership and generation-method matching require the exact approved
    Storyboard at a later cross-authority boundary, not local model validation.
    """

    model_config = ConfigDict(frozen=True)

    shot_id: Identifier
    source_segment_id: Identifier
    asset: MediaAssetReference
    generation_method: GenerationMethod
    generation_metadata: GenerationMetadata | None = None

    @model_validator(mode="after")
    def visual_asset(self) -> Self:
        if self.asset.media_type not in {MediaType.IMAGE, MediaType.VIDEO}:
            raise ValueError("Visual asset must be IMAGE or VIDEO")
        return self


class MediaPackage(Contract):
    """Manifest bound to one exact immutable Storyboard snapshot.

    Runtime must supply RuntimeState.artifacts.approved_storyboard and later
    authenticate approval, project lineage, coverage, segment/shot membership,
    and generation-method matching. Local validity proves none of those.
    Empty collections are allowed: completeness requires upstream authority.
    schema_version identifies the format, independently of artifact version.
    No binary content, duplicated grounding or timeline allocation is stored.
    """

    model_config = ConfigDict(frozen=True)

    schema_version: Literal[1] = 1
    storyboard_input_ref: ArtifactReference
    title: Text
    narration_assets: tuple[NarrationAsset, ...]
    visual_assets: tuple[VisualAsset, ...]

    @model_validator(mode="after")
    def storyboard_provenance_and_unique_identities(self) -> Self:
        if self.storyboard_input_ref.artifact_type != "storyboard":
            raise ValueError("Media input must reference a storyboard artifact")
        require_unique([item.segment_id for item in self.narration_assets], "Narration segment IDs")
        require_unique([item.shot_id for item in self.visual_assets], "Visual shot IDs")
        require_unique([item.asset.asset_id for item in (*self.narration_assets, *self.visual_assets)],
                       "Media asset IDs")
        return self
