"""Claim-bounded preparation and a single model decision; no dispatch or verdict generation."""
from history_studio.model_io import ModelProvider, ModelRequest, ModelResponse
from history_studio.models.verification_context import VerificationContext
from history_studio.research.boundaries import ResearchTools
from .context import INSTRUCTIONS, build_context, serialize_context
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
