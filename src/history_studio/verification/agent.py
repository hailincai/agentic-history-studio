"""Claim-bounded preparation, single decisions, and explicit dispatch; no loop or verdicts."""
import json

from history_studio.model_io import ModelProvider, ModelRequest, ModelResponse, NativeToolCall
from history_studio.models.verification_context import VerificationContext
from history_studio.research.actions import ReadRequest, SearchRequest
from history_studio.research.boundaries import ResearchTools, ToolObservation
from .context import (INSTRUCTIONS, OBSERVATION_INSTRUCTIONS, build_context, serialize_context,
                      serialize_observation_context)
from .tools import investigation_tools


class FactChecker:
    """Prepare one atomic claim and request one model decision without executing tools."""

    def __init__(self, context: VerificationContext, tools: ResearchTools | None = None, *,
                 provider: ModelProvider | None = None) -> None:
        if not isinstance(context, VerificationContext):
            raise TypeError("FactChecker requires one VerificationContext")
        # Own a detached, validated snapshot instead of an alias to caller working state.
        self._context = VerificationContext.model_validate(context.model_dump(mode="json"))
        self.tools = tools
        self.provider = provider
        # Metadata lookup only: neither source discovery nor a read authorizes evidence here.
        self._sources = {source.source_id: source.model_copy(deep=True) for source in self._context.sources}

    def prepare(self) -> str:
        """Return instructions followed by structured JSON; performs no model request."""
        return build_context(self._context)

    def tool_definitions(self) -> list[dict]:
        """Describe configured capabilities without invoking them or preparing a verdict."""
        return investigation_tools() if self.tools is not None else []

    def decide_next_action(self, max_output_tokens: int = 3000) -> ModelResponse:
        """One model decision, returned unchanged; no tool execution or verdict interpretation.

        Required tools apply only to this investigation-action stage, not a future terminal
        submission policy. With no configured tools, text is permitted using tool_choice=none.
        """
        if self.provider is None:
            raise RuntimeError("decide_next_action requires an injected ModelProvider")
        tools = self.tool_definitions()
        request = ModelRequest(instructions=INSTRUCTIONS, input=serialize_context(self._context),
                               tools=tools, tool_choice="required" if tools else "none",
                               max_output_tokens=max_output_tokens)
        return self.provider.decide(request)

    def decide_after_observation(self, observation: ToolObservation, *,
                                 max_output_tokens: int = 3000) -> ModelResponse:
        """One fresh reasoning turn over original context and one transient observation; no dispatch."""
        if self.provider is None:
            raise RuntimeError("decide_after_observation requires an injected ModelProvider")
        tools = self.tool_definitions()
        request = ModelRequest(instructions=INSTRUCTIONS + "\n" + OBSERVATION_INSTRUCTIONS,
            input=serialize_observation_context(self._context, observation), tools=tools,
            tool_choice="required" if tools else "none", max_output_tokens=max_output_tokens)
        return self.provider.decide(request)

    def execute_tool_call(self, call: NativeToolCall, *, max_chars: int = 8000) -> ToolObservation:
        """Execute one explicit search/read request and stop at investigation material.

        call_id stays on the caller-owned NativeToolCall for correlation. Observations retain
        their existing contract; no provenance identity is derived from the call ID.
        """
        if self.tools is None:
            raise RuntimeError("execute_tool_call requires configured ResearchTools")
        if not isinstance(call, NativeToolCall):
            raise TypeError("execute_tool_call requires one NativeToolCall")
        if call.name not in {"search_web", "read_source"}:
            raise ValueError("Unsupported Fact Checker investigation tool")
        try:
            arguments = json.loads(call.arguments)
        except json.JSONDecodeError as exc:
            raise ValueError("Invalid tool arguments JSON") from exc
        if not isinstance(arguments, dict):
            raise ValueError("Tool arguments must be a JSON object")
        if call.name == "search_web":
            request = SearchRequest.model_validate(arguments)
            result = self.tools.search_web(request.query)
            if not isinstance(result, ToolObservation) or result.kind != "search":
                raise ValueError("Search must return a search ToolObservation")
            for source in result.sources:
                self._sources[source.source_id] = source.model_copy(deep=True)
            return result
        request = ReadRequest.model_validate(arguments)
        source = self._sources.get(request.source_id)
        if source is None:
            raise ValueError("Read source_id must identify an original or discovered source")
        if isinstance(max_chars, bool) or not isinstance(max_chars, int) or max_chars <= 0:
            raise ValueError("max_chars must be a positive integer")
        result = self.tools.read_source(source.model_copy(deep=True), max_chars)
        if not isinstance(result, ToolObservation) or result.kind != "source" or result.source_id != request.source_id:
            raise ValueError("Read must return a source ToolObservation matching the requested source_id")
        return result
