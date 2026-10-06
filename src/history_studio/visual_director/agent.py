"""Bounded visual generation over approved Script context, without persistence."""
import json

from history_studio.model_io import ModelProvider, ModelRequest, ModelResponse, NativeToolCall
from history_studio.models.visual_director_context import VisualDirectorContext
from .preparation import INSTRUCTIONS, build_context, serialize_context
from .submission import StoryboardSubmission, finalize_storyboard_submission
from .generation import VisualDirectorGenerationOutcome, VisualDirectorGenerationStopReason
from .tools import storyboard_tools


class VisualDirector:
    """Own detached working input; preparation performs no model request."""

    def __init__(self, context: VisualDirectorContext, *, provider: ModelProvider | None = None) -> None:
        if not isinstance(context, VisualDirectorContext):
            raise TypeError("VisualDirector requires one VisualDirectorContext")
        self._context = VisualDirectorContext.model_validate(context.model_dump(mode="json"))
        self.provider = provider

    def prepare(self) -> str:
        """Return deterministic instructions and structured input without a model call."""
        return build_context(self._context)

    def tool_definitions(self) -> list[dict]:
        return storyboard_tools()

    def submit_storyboard(self, call: NativeToolCall) -> StoryboardSubmission:
        """Parse an untrusted terminal proposal; no external tool dispatch."""
        if not isinstance(call, NativeToolCall) or call.name != "submit_storyboard":
            raise ValueError("Submission requires one submit_storyboard NativeToolCall")
        try:
            arguments = json.loads(call.arguments)
        except json.JSONDecodeError as exc:
            raise ValueError("Invalid storyboard submission JSON") from exc
        if not isinstance(arguments, dict):
            raise ValueError("Storyboard submission arguments must be a JSON object")
        return StoryboardSubmission.model_validate(arguments)

    def generate(self, *, max_steps: int = 4, max_output_tokens: int = 6000) -> VisualDirectorGenerationOutcome:
        """Fresh bounded requests until terminal submission or the step ceiling.

        Retry only text/empty responses with identical input, without replaying any
        transcript. Provider, protocol, proposal and finalization errors propagate
        immediately. Semantic review and persistence remain outside generation.
        """
        for name, value in (("max_steps", max_steps), ("max_output_tokens", max_output_tokens)):
            if isinstance(value, bool) or not isinstance(value, int) or value <= 0:
                raise ValueError(f"{name} must be a positive integer")
        if self.provider is None:
            raise RuntimeError("generate requires an injected ModelProvider")
        context = VisualDirectorContext.model_validate(self._context.model_dump(mode="json"))
        payload = serialize_context(context)
        for step in range(1, max_steps + 1):
            request = ModelRequest(instructions=INSTRUCTIONS, input=payload,
                tools=self.tool_definitions(), tool_choice="required", max_output_tokens=max_output_tokens)
            response = self.provider.decide(request)
            if not isinstance(response, ModelResponse):
                raise TypeError("Visual Director generation requires ModelResponse")
            response = ModelResponse.model_validate(response.model_dump(mode="json"))
            if response.status not in (None, "completed") or response.incomplete_reason is not None:
                raise ValueError("Visual Director generation requires a completed model response")
            if len(response.tool_calls) > 1:
                raise ValueError("Visual Director generation permits exactly one terminal tool call")
            if not response.tool_calls:
                continue
            proposal = self.submit_storyboard(response.tool_calls[0])
            package = finalize_storyboard_submission(context, proposal)
            return VisualDirectorGenerationOutcome(steps=step,
                stop_reason=VisualDirectorGenerationStopReason.SUBMITTED, package=package)
        return VisualDirectorGenerationOutcome(steps=max_steps,
            stop_reason=VisualDirectorGenerationStopReason.LIMIT_REACHED)
