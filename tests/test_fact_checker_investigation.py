"""Bounded autonomous investigation using only fake providers and tools."""
import json

import pytest
from pydantic import ValidationError

from history_studio.model_io import ModelResponse, Usage
from history_studio.verification import FactChecker, InvestigationOutcome, InvestigationStopReason
from test_fact_checker_decision import input_context
from test_fact_checker_dispatch import RecordingTools, native


class SequenceProvider:
    def __init__(self, replies):
        self.replies = iter(replies)
        self.requests = []

    def decide(self, request):
        self.requests.append(request)
        result = next(self.replies)
        if isinstance(result, Exception):
            raise result
        return result


def decision(*calls, text="", status="completed"):
    return ModelResponse(tool_calls=list(calls), text=text, status=status,
                         usage=Usage(input_tokens=20, output_tokens=5))


def test_search_read_reason_preserves_boundaries_and_current_observation(tmp_path, monkeypatch):
    monkeypatch.chdir(tmp_path)
    original = input_context()
    before = original.model_dump_json()
    final = decision(text="DISPUTED: ordinary text, not a verdict")
    provider = SequenceProvider([decision(native()),
        decision(native("read_source", '{"source_id":"SRC-new"}')), final])
    tools = RecordingTools()
    checker = FactChecker(original, tools=tools, provider=provider)
    prepared = checker.prepare()
    outcome = checker.investigate(max_steps=3, max_output_tokens=256)
    assert isinstance(outcome, InvestigationOutcome)
    assert outcome.stop_reason == InvestigationStopReason.MODEL_TEXT
    assert outcome.final_response is final and outcome.steps == 3
    assert outcome.observations == [tools.search_result, tools.read_result]
    assert len(provider.requests) == 3 and len(tools.calls) == 2
    assert tools.calls[0] == ("search", "independent date")
    assert tools.calls[1][0] == "read" and tools.calls[1][1]["source_id"] == "SRC-new"
    assert tools.calls[1][2] == 8000
    assert json.loads(provider.requests[0].input) == original.model_dump(mode="json")
    for request, observation in zip(provider.requests[1:], outcome.observations):
        state = json.loads(request.input)
        assert state["verification_context"] == original.model_dump(mode="json")
        assert state["current_observation"] == observation.model_dump(mode="json", exclude={"usage"})
        assert set(state) == {"verification_context", "current_observation", "observation_role"}
    assert all(r.tool_choice == "required" and r.max_output_tokens == 256 for r in provider.requests)
    assert checker.prepare() == prepared and original.model_dump_json() == before
    assert final.usage.input_tokens == 20 and final.usage.output_tokens == 5
    assert "verification_id" not in outcome.model_dump() and "verification_evidence" not in outcome.model_dump()
    assert list(tmp_path.iterdir()) == []


@pytest.mark.parametrize("limit", [1, 2, 4])
def test_ceiling_counts_one_decision_and_at_most_one_action_per_step(limit):
    responses = [decision(native(call_id=f"call-{i}")) for i in range(limit)]
    provider = SequenceProvider(responses)
    tools = RecordingTools()
    outcome = FactChecker(input_context(), tools=tools, provider=provider).investigate(max_steps=limit)
    assert outcome.stop_reason == InvestigationStopReason.LIMIT_REACHED
    assert outcome.steps == len(provider.requests) == len(tools.calls) == limit
    assert outcome.final_response is responses[-1]
    assert len(outcome.observations) == limit
    assert outcome.stop_reason.value != "UNVERIFIED"


def test_default_is_four_steps_and_new_invocation_does_not_replay_old_observations():
    provider = SequenceProvider([decision(native()) for _ in range(8)])
    tools = RecordingTools()
    checker = FactChecker(input_context(), tools=tools, provider=provider)
    first = checker.investigate()
    second = checker.investigate()
    assert first.steps == second.steps == 4 and len(provider.requests) == len(tools.calls) == 8
    assert provider.requests[0].input == provider.requests[4].input
    assert "current_observation" not in json.loads(provider.requests[4].input)
    for request in provider.requests[1:4] + provider.requests[5:8]:
        assert "observations" not in json.loads(request.input)


@pytest.mark.parametrize("text,reason", [("VERIFIED", "MODEL_TEXT"), ("", "NO_TOOL_CALL"), (" \n ", "NO_TOOL_CALL")])
@pytest.mark.parametrize("configured", [True, False])
def test_text_or_absent_call_is_orchestration_outcome_not_historical_judgment(text, reason, configured):
    response = decision(text=text)
    provider = SequenceProvider([response])
    tools = RecordingTools()
    outcome = FactChecker(input_context(), tools=tools if configured else None, provider=provider).investigate()
    assert outcome.final_response is response and outcome.stop_reason.value == reason
    assert outcome.steps == 1 and outcome.observations == []
    assert len(provider.requests) == 1 and tools.calls == []
    assert provider.requests[0].tool_choice == ("required" if configured else "none")


@pytest.mark.parametrize("calls,error,message", [
    ([native(), native("read_source", '{"source_id":"SRC-A"}')], ValueError, "exactly one"),
    ([native("checkpoint_research", "{}")], ValueError, "Unsupported"),
    ([native("unknown", "{}")], ValueError, "Unsupported"),
    ([native(arguments="{broken")], ValueError, "Invalid tool arguments JSON"),
    ([native(arguments="[]")], ValueError, "JSON object"),
    ([native(arguments="{}")], ValidationError, "query"),
])
def test_bad_decisions_execute_no_tools(calls, error, message):
    provider = SequenceProvider([decision(*calls)])
    tools = RecordingTools()
    with pytest.raises(error, match=message):
        FactChecker(input_context(), tools=tools, provider=provider).investigate()
    assert len(provider.requests) == 1 and tools.calls == []


def test_invalid_later_decision_does_not_execute_first_of_multiple_calls():
    provider = SequenceProvider([decision(native()), decision(native(), native())])
    tools = RecordingTools()
    with pytest.raises(ValueError, match="exactly one"):
        FactChecker(input_context(), tools=tools, provider=provider).investigate()
    assert len(provider.requests) == 2 and len(tools.calls) == 1


@pytest.mark.parametrize("where", ["provider", "tool", "follow_up"])
def test_failure_propagates_without_retry_or_outcome(where):
    failure = RuntimeError("offline failure")
    replies = [failure] if where == "provider" else [decision(native()), failure]
    provider = SequenceProvider(replies)
    tools = RecordingTools()
    if where == "tool":
        def failed(query):
            tools.calls.append(("search", query))
            raise failure
        tools.search_web = failed
    with pytest.raises(RuntimeError) as caught:
        FactChecker(input_context(), tools=tools, provider=provider).investigate()
    assert caught.value is failure
    assert len(provider.requests) == (2 if where == "follow_up" else 1)
    assert len(tools.calls) == (0 if where == "provider" else 1)


@pytest.mark.parametrize("status", ["incomplete", "failed"])
def test_incomplete_response_is_protocol_error_before_dispatch(status):
    provider = SequenceProvider([decision(native(), status=status)])
    tools = RecordingTools()
    with pytest.raises(ValueError, match="completed model response"):
        FactChecker(input_context(), tools=tools, provider=provider).investigate()
    assert len(provider.requests) == 1 and tools.calls == []


@pytest.mark.parametrize("limit", [0, -1, True, 1.5, "4"])
def test_invalid_ceiling_fails_before_decision(limit):
    provider = SequenceProvider([])
    with pytest.raises(ValueError, match="positive integer"):
        FactChecker(input_context(), provider=provider).investigate(max_steps=limit)
    assert provider.requests == []


def test_provider_or_tools_missing_and_invalid_output_bound_fail_clearly():
    with pytest.raises(RuntimeError, match="injected ModelProvider"):
        FactChecker(input_context()).investigate()
    provider = SequenceProvider([decision(native())])
    with pytest.raises(RuntimeError, match="configured ResearchTools"):
        FactChecker(input_context(), provider=provider).investigate()
    assert len(provider.requests) == 1
    provider = SequenceProvider([])
    with pytest.raises(ValidationError):
        FactChecker(input_context(), provider=provider).investigate(max_output_tokens=0)
    assert provider.requests == []
