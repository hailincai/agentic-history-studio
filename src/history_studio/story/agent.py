"""Preparation only; no provider, decision turn, tools, or autonomous execution."""
from history_studio.models.story_context import StoryContext
from .preparation import build_context


class StoryArchitect:
    """Own a detached StoryContext for deterministic narrative-planning preparation.

    Context is semantically bounded by P4-B, not a tokenizer or fact-ranking policy.
    Preparation preserves every fact and caution; it never silently truncates input.
    Runtime is responsible for supplying context built from the approved snapshot.
    """

    def __init__(self, context: StoryContext) -> None:
        if not isinstance(context, StoryContext):
            raise TypeError("StoryArchitect requires one StoryContext")
        self._context = StoryContext.model_validate(context.model_dump(mode="json"))

    def prepare(self) -> str:
        """Return instructions and structured context; performs no model request."""
        return build_context(self._context)
