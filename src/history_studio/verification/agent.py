"""Minimal Fact Checker boundary; no provider execution, tools, or verdict generation."""
from history_studio.models.verification_context import VerificationContext
from .context import build_context


class FactChecker:
    """Prepare one atomic claim and its original research provenance for future execution."""

    def __init__(self, context: VerificationContext) -> None:
        if not isinstance(context, VerificationContext):
            raise TypeError("FactChecker requires one VerificationContext")
        # Own a detached, validated snapshot instead of an alias to caller working state.
        self._context = VerificationContext.model_validate(context.model_dump(mode="json"))

    def prepare(self) -> str:
        """Return instructions followed by structured JSON; performs no model request."""
        return build_context(self._context)
