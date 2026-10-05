"""Transient generation result; no model prose or private reasoning is retained."""
from enum import StrEnum
from typing import Self

from pydantic import Field, model_validator

from history_studio.models.base import Contract
from history_studio.models.script_package import ScriptPackage


class ScriptGenerationStopReason(StrEnum):
    SUBMITTED = "SUBMITTED"
    LIMIT_REACHED = "LIMIT_REACHED"


class ScriptGenerationOutcome(Contract):
    steps: int = Field(strict=True, ge=1)
    stop_reason: ScriptGenerationStopReason
    package: ScriptPackage | None = None

    @model_validator(mode="after")
    def terminal_shape(self) -> Self:
        if (self.stop_reason == ScriptGenerationStopReason.SUBMITTED) != (self.package is not None):
            raise ValueError("Only submitted generation has a ScriptPackage")
        return self
