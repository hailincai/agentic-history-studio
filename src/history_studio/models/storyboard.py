from enum import StrEnum
from typing import Self

from pydantic import Field, model_validator

from .base import Contract, Identifier, Nonnegative, PositiveSeconds, Text, require_unique


class GenerationMethod(StrEnum):
    IMAGE_TO_VIDEO = "IMAGE_TO_VIDEO"
    TEXT_TO_VIDEO = "TEXT_TO_VIDEO"
    STATIC_IMAGE = "STATIC_IMAGE"


class Shot(Contract):
    shot_id: Identifier
    scene_id: Identifier
    sequence: int = Field(gt=0, strict=True)
    start_seconds: Nonnegative
    duration_seconds: PositiveSeconds
    visual_description: Text
    characters: list[Text] = Field(default_factory=list)
    location: Text
    period: Text
    generation_method: GenerationMethod
    camera_motion: Text
    historical_constraints: list[Text] = Field(default_factory=list)
    prompt: Text


class Storyboard(Contract):
    """Shot start times use the documentary's global timeline; gaps are allowed."""

    shots: list[Shot] = Field(min_length=1)
    estimated_media_cost_usd: Nonnegative

    @model_validator(mode="after")
    def valid_timeline(self) -> Self:
        require_unique([shot.shot_id for shot in self.shots], "Shot IDs")
        sequences = [shot.sequence for shot in self.shots]
        require_unique(sequences, "Shot sequences")
        if sequences != sorted(sequences):
            raise ValueError("Shots must be in sequence order")
        for previous, current in zip(self.shots, self.shots[1:]):
            if current.start_seconds < previous.start_seconds + previous.duration_seconds:
                raise ValueError("Shots must not overlap or move backwards")
        return self
