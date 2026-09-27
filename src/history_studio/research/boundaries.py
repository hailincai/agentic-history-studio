from typing import Any, Literal, Protocol

from pydantic import Field

from history_studio.models.base import Contract, Nonnegative, Text
from history_studio.models.sources import SourceReference


class Usage(Contract):
    model: str | None = None
    input_tokens: int | None = Field(default=None, ge=0)
    output_tokens: int | None = Field(default=None, ge=0)
    estimated_model_cost_usd: Nonnegative | None = None
    estimated_tool_cost_usd: Nonnegative | None = None


class ToolCall(Contract):
    call_id: Text
    name: Text
    arguments: dict[str, Any]


class ModelReply(Contract):
    call: ToolCall
    usage: Usage


class ToolObservation(Contract):
    kind: Literal["search", "source"]
    sources: list[SourceReference] = Field(default_factory=list)
    text: str = ""
    source_id: str | None = None
    truncated: bool = False
    usage: Usage = Field(default_factory=Usage)


class ResearchProvider(Protocol):
    def reserve_cost(self, context: str, observation: tuple[ToolCall, str] | None,
                     max_output_tokens: int) -> float: ...
    def decide(self, context: str, observation: tuple[ToolCall, str] | None,
               max_output_tokens: int) -> ModelReply: ...


class ResearchTools(Protocol):
    def search_reserve_cost(self, query: str) -> float: ...
    def search_web(self, query: str) -> ToolObservation: ...
    def read_source(self, source: SourceReference, max_chars: int) -> ToolObservation: ...
