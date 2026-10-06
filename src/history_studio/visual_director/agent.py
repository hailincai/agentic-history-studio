"""Preparation-only Visual Director over the bounded approved Script projection."""
from history_studio.models.visual_director_context import VisualDirectorContext
from .preparation import build_context


class VisualDirector:
    """Own detached working input; preparation has no execution capabilities."""

    def __init__(self, context: VisualDirectorContext) -> None:
        if not isinstance(context, VisualDirectorContext):
            raise TypeError("VisualDirector requires one VisualDirectorContext")
        self._context = VisualDirectorContext.model_validate(context.model_dump(mode="json"))

    def prepare(self) -> str:
        """Return deterministic instructions and structured input without a model call."""
        return build_context(self._context)
