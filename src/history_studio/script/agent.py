"""Preparation-only Script Writer; no provider, tools or generation capability."""
from history_studio.models.script_context import ScriptContext
from .preparation import build_context


class ScriptWriter:
    """Own a detached bounded context supplied from the approved Story by Runtime."""

    def __init__(self, context: ScriptContext) -> None:
        if not isinstance(context, ScriptContext):
            raise TypeError("ScriptWriter requires one ScriptContext")
        self._context = ScriptContext.model_validate(context.model_dump(mode="json"))

    def prepare(self) -> str:
        """Return deterministic instructions and context without execution or persistence."""
        return build_context(self._context)
