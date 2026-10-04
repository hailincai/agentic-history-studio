"""One fake model decision, no tool execution or verification result generation."""
import json
from typing import get_type_hints

import pytest
from pydantic import ValidationError

from history_studio.model_io import ModelProvider, ModelRequest, ModelResponse, NativeToolCall, Usage
from history_studio.models import build_verification_context
from history_studio.models.research import EvidenceReference
from history_studio.research.agent import ResearchAgent
from history_studio.research.boundaries import ResearchProvider
from history_studio.verification import FactChecker
from history_studio.verification.context import INSTRUCTIONS
from test_research_agent import FakeTools
from test_verification_context import research_ref, package_data


class FakeModelProvider:
    def __init__(self, response):
        self.response = response
        self.requests = []

    def decide(self, request):
        self.requests.append(request)
        if isinstance(self.response, Exception):
            raise self.response
        return self.response


def input_context():
    return build_verification_context(package_data(), "RF-target", research_input_ref=research_ref())


@pytest.mark.parametrize("name, arguments", [
    ("search_web", '{"query":"independent date contradiction"}'),
    ("read_source", '{"source_id":"SRC-A"}'),
])
def test_one_decision_passes_claim_only_request_and_never_executes_tools(name, arguments, tmp_path, monkeypatch):
    monkeypatch.chdir(tmp_path)
    context = input_context()
    before = context.model_dump_json()
    response = ModelResponse(tool_calls=[NativeToolCall(call_id="next", name=name, arguments=arguments)],
        usage=Usage(model="fake", input_tokens=50, output_tokens=10, estimated_model_cost_usd=0.001),
        status="completed")
    provider = FakeModelProvider(response)
    tools = FakeTools()
    checker = FactChecker(context, tools=tools, provider=provider)
    prepared = checker.prepare()
    assert provider.requests == []
    assert checker.decide_next_action(max_output_tokens=256) is response
    assert len(provider.requests) == 1
    request = provider.requests[0]
    assert isinstance(request, ModelRequest)
    assert request.instructions == INSTRUCTIONS
    assert request.input == prepared[len(INSTRUCTIONS) + 1:]
    state = json.loads(request.input)
    assert state == context.model_dump(mode="json")
    assert state["target_fact"]["fact_id"] == "RF-target"
    assert state["target_fact"]["claim"] == context.target_fact.claim
    assert state["original_evidence_role"] == "RESEARCH_INPUT_NOT_INDEPENDENT_VERIFICATION"
    assert state["whole_topic_research_allowed"] is False
    assert "RF-other" not in request.input and "SRC-unrelated" not in request.input
    assert "SRC-discovered" not in request.input and "plan" not in state and "facts" not in state
    assert [d["name"] for d in request.tools] == ["search_web", "read_source", "submit_verification"]
    assert request.tools == checker.tool_definitions()
    assert request.tool_choice == "required" and request.max_output_tokens == 256
    assert "checkpoint_research" not in json.dumps(request.model_dump())
    assert tools.queries == [] and tools.reads == []
    assert response.usage.estimated_model_cost_usd == 0.001
    assert checker.prepare() == prepared and context.model_dump_json() == before
    assert list(tmp_path.iterdir()) == []


def test_without_tools_text_remains_uninterpreted_model_output():
    context = input_context()
    response = ModelResponse(text="VERIFIED: ordinary model text is not a typed verdict.",
                             usage=Usage(), status="completed")
    provider = FakeModelProvider(response)
    checker = FactChecker(context, provider=provider)
    assert checker.decide_next_action() is response
    assert len(provider.requests) == 1
    assert provider.requests[0].tools == [] and provider.requests[0].tool_choice == "none"
    assert provider.requests[0].max_output_tokens == 3000
    assert response.model_dump()["text"].startswith("VERIFIED")
    assert "verification_evidence" not in response.model_dump()
    assert "verification_id" not in response.model_dump()
    assert all(type(e) is EvidenceReference for e in context.target_fact.evidence)


@pytest.mark.parametrize("tools", [None, "fake"])
def test_missing_provider_fails_clearly_without_affecting_preparation(tools):
    checker = FactChecker(input_context(), tools=FakeTools() if tools else None)
    before = checker.prepare()
    with pytest.raises(RuntimeError, match="injected ModelProvider"):
        checker.decide_next_action()
    assert checker.prepare() == before


def test_provider_exception_propagates_without_retry_or_tool_execution():
    error = RuntimeError("offline provider failed")
    provider = FakeModelProvider(error)
    tools = FakeTools()
    checker = FactChecker(input_context(), tools=tools, provider=provider)
    with pytest.raises(RuntimeError) as caught:
        checker.decide_next_action()
    assert caught.value is error and len(provider.requests) == 1
    assert tools.queries == [] and tools.reads == []


def test_invalid_output_limit_fails_before_provider_request():
    provider = FakeModelProvider(ModelResponse(usage=Usage()))
    with pytest.raises(ValidationError):
        FactChecker(input_context(), provider=provider).decide_next_action(max_output_tokens=0)
    assert provider.requests == []


def test_agent_provider_contracts_remain_distinct():
    assert get_type_hints(FactChecker.__init__)["provider"] == ModelProvider | None
    assert get_type_hints(ResearchAgent.__init__)["provider"] is ResearchProvider
