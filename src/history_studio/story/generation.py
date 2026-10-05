"""Transient generation result; no model prose or private reasoning is retained."""
from enum import StrEnum
from typing import Self

from pydantic import Field, model_validator

from history_studio.models.base import Contract
from history_studio.models.story_package import StoryPackage


class StoryGenerationStopReason(StrEnum):
    SUBMITTED = "SUBMITTED"
    LIMIT_REACHED = "LIMIT_REACHED"


class StoryGenerationOutcome(Contract):
    steps: int = Field(strict=True, ge=1)
    stop_reason: StoryGenerationStopReason
    package: StoryPackage | None = None

    @model_validator(mode="after")
    def terminal_shape(self) -> Self:
        if (self.stop_reason == StoryGenerationStopReason.SUBMITTED) != (self.package is not None):
            raise ValueError("Only submitted generation has a StoryPackage")
        return self
