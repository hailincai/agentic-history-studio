"""Minimal Fact Checker boundary; no execution, dispatch, or verdict generation."""
from history_studio.models.verification_context import VerificationContext
from history_studio.research.boundaries import ResearchTools
from .context import build_context
from .tools import investigation_tools


class FactChecker:
    """Prepare one atomic claim and its original research provenance for future execution."""

    def __init__(self, context: VerificationContext, tools: ResearchTools | None = None) -> None:
        if not isinstance(context, VerificationContext):
            raise TypeError("FactChecker requires one VerificationContext")
        # Own a detached, validated snapshot instead of an alias to caller working state.
        self._context = VerificationContext.model_validate(context.model_dump(mode="json"))
        self.tools = tools

    def prepare(self) -> str:
        """Return instructions followed by structured JSON; performs no model request."""
        return build_context(self._context)

    def tool_definitions(self) -> list[dict]:
        """Describe configured capabilities without invoking them or preparing a verdict."""
        return investigation_tools() if self.tools is not None else []
