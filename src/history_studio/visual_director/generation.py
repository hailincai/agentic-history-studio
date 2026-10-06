"""Transient typed generation result; no model prose or reasoning is retained."""
from enum import StrEnum
from typing import Self

from pydantic import Field, model_validator

from history_studio.models.base import Contract
from history_studio.models.storyboard_package import StoryboardPackage


class VisualDirectorGenerationStopReason(StrEnum):
    SUBMITTED = "SUBMITTED"
    LIMIT_REACHED = "LIMIT_REACHED"


class VisualDirectorGenerationOutcome(Contract):
    steps: int = Field(strict=True, ge=1)
    stop_reason: VisualDirectorGenerationStopReason
    package: StoryboardPackage | None = None

    @model_validator(mode="after")
    def terminal_shape(self) -> Self:
        if (self.stop_reason == VisualDirectorGenerationStopReason.SUBMITTED) != (self.package is not None):
            raise ValueError("Only submitted generation has a StoryboardPackage")
        return self
