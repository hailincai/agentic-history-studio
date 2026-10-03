"""One current investigation observation, one fake reasoning turn, no automatic action."""
import json

import pytest
from pydantic import ValidationError

from history_studio.model_io import ModelResponse, NativeToolCall, Usage
from history_studio.research.boundaries import ToolObservation
from history_studio.verification import FactChecker
from history_studio.verification.context import INSTRUCTIONS, MAX_OBSERVATION_CHARS
from test_fact_checker_decision import FakeModelProvider, input_context
from test_fact_checker_dispatch import RecordingTools, native


@pytest.mark.parametrize("kind", ["search", "source"])
def test_follow_up_preserves_original_and_current_input_without_dispatch(kind, tmp_path, monkeypatch):
    monkeypatch.chdir(tmp_path)
    context = input_context()
    before = context.model_dump_json()
    tools = RecordingTools()
    observation = ToolObservation(kind=kind, sources=context.sources,
        source_id="SRC-A" if kind == "source" else None,
        text="中文 investigation text or search snippet", truncated=True,
        usage=Usage(input_tokens=123))
    observation_before = observation.model_dump_json()
    response = ModelResponse(tool_calls=[native("read_source", '{"source_id":"SRC-A"}')],
        text="Not a parsed verdict", usage=Usage(output_tokens=12), status="completed")
    provider = FakeModelProvider(response)
    checker = FactChecker(context, tools=tools, provider=provider)
    prepared = checker.prepare()
    assert checker.decide_after_observation(observation, max_output_tokens=256) is response
    assert len(provider.requests) == 1 and tools.calls == []
    request = provider.requests[0]
    state = json.loads(request.input)
    assert state["verification_context"] == context.model_dump(mode="json")
    assert state["current_observation"] == observation.model_dump(mode="json", exclude={"usage"})
    assert state["observation_role"] == "EXECUTED_ACTION_INVESTIGATION_MATERIAL_NOT_ACCEPTED_EVIDENCE"
    assert request.instructions.startswith(INSTRUCTIONS)
    for guidance in ("previously executed", "not accepted VerificationEvidence", "relevance",
                     "contradiction", "Source IDs", "span IDs", "not automatically selected"):
        assert guidance in request.instructions
    assert [t["name"] for t in request.tools] == ["search_web", "read_source"]
    assert request.tool_choice == "required" and request.max_output_tokens == 256
    assert "checkpoint_research" not in json.dumps(request.tools)
    assert response.usage.output_tokens == 12
    assert checker.prepare() == prepared and context.model_dump_json() == before
    assert observation.model_dump_json() == observation_before
    assert "verification_id" not in state and "verification_evidence" not in state
    assert list(tmp_path.iterdir()) == []


def test_explicit_reason_act_observe_reason_stops_before_second_action():
    response = ModelResponse(tool_calls=[native()], usage=Usage())
    provider = FakeModelProvider(response)
    tools = RecordingTools()
    checker = FactChecker(input_context(), tools=tools, provider=provider)
    first = checker.decide_next_action()
    observation = checker.execute_tool_call(first.tool_calls[0])
    second = checker.decide_after_observation(observation)
    assert second is response and len(provider.requests) == 2
    assert tools.calls == [("search", "independent date")]
    # Runtime lookup survives; observation reasoning never performs a read itself.
    assert checker.execute_tool_call(native("read_source", '{"source_id":"SRC-new"}')).source_id == "SRC-new"
    assert len(provider.requests) == 2


def test_independent_follow_ups_do_not_accumulate_observations_or_runtime_sources():
    provider = FakeModelProvider(ModelResponse(text="VERIFIED remains ordinary text", usage=Usage()))
    tools = RecordingTools()
    checker = FactChecker(input_context(), tools=tools, provider=provider)
    checker.execute_tool_call(native())
    checker.decide_after_observation(ToolObservation(kind="source", text="old-page-marker"))
    checker.decide_after_observation(ToolObservation(kind="search", text="current-marker"))
    assert len(provider.requests) == 2
    assert "old-page-marker" not in provider.requests[1].input
    assert "SRC-new" not in provider.requests[1].input
    assert "current-marker" in provider.requests[1].input
    assert len(tools.calls) == 1


def test_no_tools_text_response_is_returned_with_usage_and_no_verdict():
    response = ModelResponse(text="VERIFIED", status="completed", usage=Usage(output_tokens=5))
    provider = FakeModelProvider(response)
    checker = FactChecker(input_context(), provider=provider)
    assert checker.decide_after_observation(ToolObservation(kind="search")) is response
    assert len(provider.requests) == 1
    assert provider.requests[0].tools == [] and provider.requests[0].tool_choice == "none"
    assert provider.requests[0].max_output_tokens == 3000
    assert response.usage.output_tokens == 5 and "verification_id" not in response.model_dump()


def test_provider_failure_propagates_once_without_retry_or_tool_execution():
    failure = RuntimeError("offline model failure")
    provider = FakeModelProvider(failure)
    tools = RecordingTools()
    checker = FactChecker(input_context(), tools=tools, provider=provider)
    with pytest.raises(RuntimeError) as caught:
        checker.decide_after_observation(ToolObservation(kind="search"))
    assert caught.value is failure and len(provider.requests) == 1 and tools.calls == []


@pytest.mark.parametrize("value", [None, {}, [ToolObservation(kind="search")]])
def test_follow_up_requires_exactly_one_typed_observation(value):
    provider = FakeModelProvider(ModelResponse(usage=Usage()))
    with pytest.raises(TypeError, match="one ToolObservation"):
        FactChecker(input_context(), provider=provider).decide_after_observation(value)
    assert provider.requests == []


def test_missing_provider_and_invalid_output_limit_fail_before_request():
    observation = ToolObservation(kind="search")
    with pytest.raises(RuntimeError, match="injected ModelProvider"):
        FactChecker(input_context()).decide_after_observation(observation)
    provider = FakeModelProvider(ModelResponse(usage=Usage()))
    with pytest.raises(ValidationError):
        FactChecker(input_context(), provider=provider).decide_after_observation(observation, max_output_tokens=0)
    assert provider.requests == []


@pytest.mark.parametrize("oversized_field", ["text", "metadata"])
def test_entire_observation_is_bounded_without_mutation(oversized_field):
    context = input_context()
    observation = ToolObservation(kind="source", sources=context.sources, text="exact text", truncated=True)
    if oversized_field == "text":
        observation.text = "中" * MAX_OBSERVATION_CHARS
    else:
        observation.sources[0].notes = "x" * MAX_OBSERVATION_CHARS
    before = observation.model_dump_json()
    provider = FakeModelProvider(ModelResponse(usage=Usage()))
    with pytest.raises(ValueError, match="character limit"):
        FactChecker(context, provider=provider).decide_after_observation(observation)
    assert provider.requests == [] and observation.model_dump_json() == before


def test_serialized_observation_boundary_counts_json_escaping():
    provider = FakeModelProvider(ModelResponse(usage=Usage()))
    checker = FactChecker(input_context(), provider=provider)
    empty = ToolObservation(kind="source")
    overhead = len(json.dumps(empty.model_dump(mode="json", exclude={"usage"}), ensure_ascii=False, separators=(",", ":")))
    empty.text = "x" * (MAX_OBSERVATION_CHARS - overhead)
    checker.decide_after_observation(empty)
    empty.text += "x"
    with pytest.raises(ValueError, match="character limit"):
        checker.decide_after_observation(empty)
    empty.text = '"' * MAX_OBSERVATION_CHARS
    with pytest.raises(ValueError, match="character limit"):
        checker.decide_after_observation(empty)
    assert len(provider.requests) == 1
