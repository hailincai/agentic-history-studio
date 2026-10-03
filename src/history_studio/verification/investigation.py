"""Transient investigation orchestration results, not historical verification judgments."""
from enum import StrEnum

from pydantic import Field

from history_studio.model_io import ModelResponse
from history_studio.models.base import Contract
from history_studio.research.boundaries import ToolObservation


class InvestigationStopReason(StrEnum):
    LIMIT_REACHED = "LIMIT_REACHED"
    MODEL_TEXT = "MODEL_TEXT"
    NO_TOOL_CALL = "NO_TOOL_CALL"


class InvestigationOutcome(Contract):
    """Runtime reporting only; observation history is never replayed to the model."""
    final_response: ModelResponse
    observations: list[ToolObservation] = Field(default_factory=list)
    steps: int = Field(ge=1)
    stop_reason: InvestigationStopReason
