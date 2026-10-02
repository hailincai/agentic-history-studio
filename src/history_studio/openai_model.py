"""Single-turn OpenAI transport; caller adapters own Agent-specific response rules."""
from typing import Any

from openai import OpenAI

from .model_io import ModelRequest, ModelResponse, NativeToolCall, Usage


def usage_of(response: Any, input_rate: float, output_rate: float, tool_cost: float = 0) -> Usage:
    raw = response.usage
    return Usage(model=response.model, input_tokens=raw.input_tokens if raw else None,
                 output_tokens=raw.output_tokens if raw else None,
                 estimated_model_cost_usd=((raw.input_tokens * input_rate + raw.output_tokens * output_rate)
                                           / 1_000_000 if raw else None),
                 estimated_tool_cost_usd=tool_cost)


class OpenAIModelProvider:
    def __init__(self, client: OpenAI, model: str, input_rate: float, output_rate: float) -> None:
        self.client = client
        self.model = model
        self.input_rate = input_rate
        self.output_rate = output_rate

    def request_body(self, request: ModelRequest) -> dict[str, Any]:
        return dict(model=self.model, **request.model_dump(), parallel_tool_calls=False, store=False)

    def decide(self, request: ModelRequest) -> ModelResponse:
        """One SDK request, no loop, retry, tool execution, or domain interpretation."""
        response = self.client.responses.create(**self.request_body(request))
        calls = [NativeToolCall(call_id=item.call_id, name=item.name, arguments=item.arguments)
                 for item in response.output if item.type == "function_call"]
        text = "".join(content.text for item in response.output if item.type == "message"
                       for content in item.content if content.type == "output_text")
        return ModelResponse(tool_calls=calls, text=text, status=response.status,
            incomplete_reason=getattr(getattr(response, "incomplete_details", None), "reason", None),
            usage=usage_of(response, self.input_rate, self.output_rate))
