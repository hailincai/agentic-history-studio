import json

import pytest

from history_studio.model_io import ModelRequest, ModelResponse, NativeToolCall
from history_studio.research.openai_provider import strict_schema
from history_studio.story import StoryArchitect, StorySubmission, finalize_story_submission
from history_studio.story.preparation import INSTRUCTIONS, serialize_context
from test_story_architect import context
from test_story_submission import submission, with_beats, beat, selection


class FakeProvider:
    def __init__(self, responses):
        self.responses = list(responses)
        self.requests = []

    def decide(self, request: ModelRequest) -> ModelResponse:
        self.requests.append(request)
        response = self.responses.pop(0)
        if isinstance(response, Exception):
            raise response
        return response


def call(arguments=None, name="submit_story"):
    return NativeToolCall(call_id="call_1", name=name, arguments=
        submission().model_dump_json() if arguments is None else arguments)


def response(calls=None, text="", **kwargs):
    return ModelResponse(tool_calls=[] if calls is None else calls, text=text, usage={}, **kwargs)


def test_generation_prepared_request_schema_and_canonical_output(monkeypatch):
    import history_studio.story.agent as agent
    source = context()
    before = source.model_dump_json()
    provider = FakeProvider([response([call()]), RuntimeError("must not be called")])
    finalized = []
    def spy(ctx, proposal):
        finalized.append(proposal)
        return finalize_story_submission(ctx, proposal)
    monkeypatch.setattr(agent, "finalize_story_submission", spy)
    architect = StoryArchitect(source, provider=provider)
    outcome = architect.generate(max_output_tokens=1234)
    assert outcome.stop_reason == "SUBMITTED" and outcome.steps == 1
    assert outcome.package == finalize_story_submission(source, submission())
    assert len(finalized) == len(provider.requests) == 1
    request = provider.requests[0]
    assert request.instructions == INSTRUCTIONS and request.input == serialize_context(source)
    assert architect.prepare() == request.instructions + "\n" + request.input
    assert request.tool_choice == "required" and request.max_output_tokens == 1234
    assert [tool["name"] for tool in request.tools] == ["submit_story"]
    assert request.tools[0]["parameters"] == strict_schema(StorySubmission.model_json_schema())
    assert request.tools[0]["strict"] is True
    assert source.model_dump_json() == before
    assert outcome.package.verification_input_ref == source.verification_input_ref
    accepted = outcome.package.plan.sections[0].beats[0]
    assert accepted.fact_refs[0].status == source.eligible_facts[0].status
    assert accepted.fact_chronology[0].historical_time == source.eligible_facts[0].historical_time
    assert not hasattr(architect, "tools")


def test_only_agent_fields_in_native_schema():
    schema = StoryArchitect(context()).tool_definitions()[0]["parameters"]
    encoded = json.dumps(schema)
    for forbidden in ("VerificationStatus", "HistoricalTime", "verification_input_ref", "project_id",
                      "source_version", "research_confidence", "fact_chronology"):
        assert forbidden not in encoded


@pytest.mark.parametrize("text", ["Here is my final story plan...", "", " "])
def test_no_submission_reaches_bound_without_fake_output_or_transcript(text):
    provider = FakeProvider([response(text=text) for _ in range(3)])
    outcome = StoryArchitect(context(), provider=provider).generate(max_steps=3)
    assert outcome.package is None and outcome.stop_reason == "LIMIT_REACHED" and outcome.steps == 3
    assert len(provider.requests) == 3
    assert provider.requests[0] == provider.requests[1] == provider.requests[2]
    assert "final story plan" not in provider.requests[2].input
    assert "text" not in outcome.model_dump()


def test_submission_at_ceiling_terminal_and_generation_state_does_not_leak():
    source = context()
    provider = FakeProvider([response(text="Private reasoning"), response([call()]), response([call()])])
    architect = StoryArchitect(source, provider=provider)
    first = architect.generate(max_steps=2)
    second = architect.generate(max_steps=2)
    assert first.steps == 2 and second.steps == 1
    assert first.package == second.package
    assert provider.requests[0] == provider.requests[1] == provider.requests[2]
    assert "Private reasoning" not in second.model_dump_json()


@pytest.mark.parametrize("arguments", ["{", "[]", "null", "{}", '{"title": "x", "status": "VERIFIED"}'])
def test_malformed_or_invalid_submission_fails_without_retry(arguments):
    provider = FakeProvider([response([call(arguments)])])
    with pytest.raises(ValueError):
        StoryArchitect(context(), provider=provider).generate()
    assert len(provider.requests) == 1


@pytest.mark.parametrize("fact_id,use,qualification", [
    ("unknown", "AFFIRMATIVE", None), ("RF-3", "AFFIRMATIVE", None),
    ("RF-4", "AFFIRMATIVE", None), ("RF-pending", "AFFIRMATIVE", None),
    ("RF-1", "QUALIFIED", None), ("RF-2", "AFFIRMATIVE", None),
])
def test_grounding_failure_propagates_from_p4d(fact_id, use, qualification):
    proposal = with_beats([beat(fact_proposals=[selection(fact_id, use, qualification)])])
    provider = FakeProvider([response([call(proposal.model_dump_json())])])
    with pytest.raises(ValueError):
        StoryArchitect(context(), provider=provider).generate()
    assert len(provider.requests) == 1


@pytest.mark.parametrize("calls", [[call(), call()], [call(name="read_source")], [call(name="search_web")]])
def test_multiple_or_external_calls_fail_closed(calls):
    provider = FakeProvider([response(calls)])
    with pytest.raises(ValueError):
        StoryArchitect(context(), provider=provider).generate()
    assert len(provider.requests) == 1


@pytest.mark.parametrize("status", ["incomplete", "failed", "in_progress"])
def test_noncompleted_response_fails(status):
    provider = FakeProvider([response([call()], status=status)])
    with pytest.raises(ValueError, match="completed"):
        StoryArchitect(context(), provider=provider).generate()


def test_provider_exception_propagates_without_retry():
    error = RuntimeError("provider failure")
    provider = FakeProvider([error])
    with pytest.raises(RuntimeError, match="provider failure") as caught:
        StoryArchitect(context(), provider=provider).generate()
    assert caught.value is error and len(provider.requests) == 1


@pytest.mark.parametrize("value", [0, -1, True, 1.5])
@pytest.mark.parametrize("parameter", ["max_steps", "max_output_tokens"])
def test_invalid_bounds_make_no_request(parameter, value):
    provider = FakeProvider([])
    with pytest.raises(ValueError, match="positive integer"):
        StoryArchitect(context(), provider=provider).generate(**{parameter: value})
    assert provider.requests == []


def test_generation_requires_injected_provider_and_tools_are_detached():
    architect = StoryArchitect(context())
    with pytest.raises(RuntimeError, match="injected ModelProvider"):
        architect.generate()
    tools = architect.tool_definitions()
    tools[0]["name"] = "read_source"
    assert architect.tool_definitions()[0]["name"] == "submit_story"


def test_same_fake_response_and_context_same_semantic_result():
    source = context()
    first = StoryArchitect(source, provider=FakeProvider([response([call()])])).generate()
    second = StoryArchitect(source, provider=FakeProvider([response([call()])])).generate()
    assert first == second
