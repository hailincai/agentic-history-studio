"""Agent-neutral contracts for a single model request and returned output."""
from typing import Any, Literal, Protocol

from pydantic import Field

from .models.base import Contract, Nonnegative


class Usage(Contract):
    model: str | None = None
    input_tokens: int | None = Field(default=None, ge=0)
    output_tokens: int | None = Field(default=None, ge=0)
    estimated_model_cost_usd: Nonnegative | None = None
    estimated_tool_cost_usd: Nonnegative | None = None


class ModelRequest(Contract):
    instructions: str
    input: str | list[dict[str, Any]]
    tools: list[dict[str, Any]] = Field(default_factory=list)
    tool_choice: Literal["auto", "none", "required"] | dict[str, Any]
    max_output_tokens: int = Field(gt=0)


class NativeToolCall(Contract):
    """Native call with raw argument JSON; caller adapters own decoding and validation."""
    call_id: str
    name: str
    arguments: str


class ModelResponse(Contract):
    tool_calls: list[NativeToolCall] = Field(default_factory=list)
    text: str = ""
    usage: Usage
    status: str | None = None
    incomplete_reason: str | None = None


class ModelProvider(Protocol):
    def decide(self, request: ModelRequest) -> ModelResponse: ...
