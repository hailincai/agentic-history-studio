"""Bounded narrative generation over approved context, without research or persistence."""
import json

from history_studio.model_io import ModelProvider, ModelRequest, ModelResponse, NativeToolCall
from history_studio.models.script_context import ScriptContext
from .preparation import INSTRUCTIONS, build_context, serialize_context
from .submission import ScriptSubmission, finalize_script_submission
from .generation import ScriptGenerationOutcome, ScriptGenerationStopReason
from .tools import script_tools


class ScriptWriter:
    """Own a detached ScriptContext for deterministic narration preparation.

    Context is semantically bounded by P5-B, not a tokenizer or fact-ranking policy.
    Preparation preserves every fact and caution; it never silently truncates input.
    Runtime is responsible for supplying context built from the approved snapshot.
    """

    def __init__(self, context: ScriptContext, *, provider: ModelProvider | None = None) -> None:
        if not isinstance(context, ScriptContext):
            raise TypeError("ScriptWriter requires one ScriptContext")
        self._context = ScriptContext.model_validate(context.model_dump(mode="json"))
        self.provider = provider

    def prepare(self) -> str:
        """Return instructions and structured context; performs no model request."""
        return build_context(self._context)

    def tool_definitions(self) -> list[dict]:
        return script_tools()

    def submit_script(self, call: NativeToolCall) -> ScriptSubmission:
        """Parse an untrusted terminal proposal; no finalization or external dispatch."""
        if not isinstance(call, NativeToolCall) or call.name != "submit_script":
            raise ValueError("Submission requires one submit_script NativeToolCall")
        try:
            arguments = json.loads(call.arguments)
        except json.JSONDecodeError as exc:
            raise ValueError("Invalid script submission JSON") from exc
        if not isinstance(arguments, dict):
            raise ValueError("Script submission arguments must be a JSON object")
        return ScriptSubmission.model_validate(arguments)

    def generate(self, *, max_steps: int = 4, max_output_tokens: int = 6000) -> ScriptGenerationOutcome:
        """Fresh bounded requests until terminal submission or the deterministic ceiling.

        Non-submitting text/empty responses are not proposals. Retry only those responses
        with identical prepared context; never replay text, calls or reasoning. Protocol,
        provider, proposal and P5-D failures propagate immediately without repair/retry.
        No state or transcript survives this invocation. Semantic prose review is deferred.
        """
        for name, value in (("max_steps", max_steps), ("max_output_tokens", max_output_tokens)):
            if isinstance(value, bool) or not isinstance(value, int) or value <= 0:
                raise ValueError(f"{name} must be a positive integer")
        if self.provider is None:
            raise RuntimeError("generate requires an injected ModelProvider")
        context = ScriptContext.model_validate(self._context.model_dump(mode="json"))
        payload = serialize_context(context)
        for step in range(1, max_steps + 1):
            request = ModelRequest(instructions=INSTRUCTIONS, input=payload,
                tools=self.tool_definitions(), tool_choice="required", max_output_tokens=max_output_tokens)
            response = self.provider.decide(request)
            if not isinstance(response, ModelResponse):
                raise TypeError("Script generation requires ModelResponse")
            response = ModelResponse.model_validate(response.model_dump(mode="json"))
            if response.status not in (None, "completed") or response.incomplete_reason is not None:
                raise ValueError("Script generation requires a completed model response")
            if len(response.tool_calls) > 1:
                raise ValueError("Script generation permits exactly one terminal tool call")
            if not response.tool_calls:
                continue
            proposal = self.submit_script(response.tool_calls[0])
            package = finalize_script_submission(context, proposal)
            return ScriptGenerationOutcome(steps=step, stop_reason=ScriptGenerationStopReason.SUBMITTED,
                                          package=package)
        return ScriptGenerationOutcome(steps=max_steps, stop_reason=ScriptGenerationStopReason.LIMIT_REACHED)
