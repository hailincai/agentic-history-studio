import json

import pytest
from pydantic import ValidationError

from history_studio.model_io import ModelResponse, NativeToolCall
from history_studio.models import StoryboardPackage
from history_studio.research.openai_provider import strict_schema
from history_studio.visual_director import (
    VisualDirector, VisualDirectorGenerationOutcome, StoryboardSubmission, finalize_storyboard_submission,
)
from history_studio.visual_director.preparation import INSTRUCTIONS, serialize_context
from test_visual_director_preparation import context
from test_storyboard_submission import submission, shot


class FakeProvider:
    def __init__(self, responses):
        self.responses = list(responses)
        self.requests = []

    def decide(self, request):
        self.requests.append(request)
        result = self.responses.pop(0)
        if isinstance(result, Exception):
            raise result
        return result


def call(arguments=None, name="submit_storyboard"):
    return NativeToolCall(call_id="call1", name=name,
        arguments=submission().model_dump_json() if arguments is None else arguments)


def response(calls=None, text="", **changes):
    return ModelResponse(tool_calls=calls or [], text=text, usage={}, **changes)


def test_native_submission_finalizes_once_with_exact_input(monkeypatch):
    import history_studio.visual_director.agent as agent
    source = context()
    before = source.model_dump_json()
    data = submission().model_dump(mode="json")
    data["sections"][0]["shots"].insert(1, shot("extra"))
    proposal = StoryboardSubmission.model_validate(data)
    provider = FakeProvider([response([call(proposal.model_dump_json())]), RuntimeError("must stop")])
    finalized = []

    def spy(ctx, proposed):
        finalized.append(proposed)
        return finalize_storyboard_submission(ctx, proposed)

    monkeypatch.setattr(agent, "finalize_storyboard_submission", spy)
    director = VisualDirector(source, provider=provider)
    outcome = director.generate(max_output_tokens=1234)
    assert outcome.stop_reason == "SUBMITTED" and outcome.steps == 1
    assert isinstance(outcome.package, StoryboardPackage)
    assert outcome.package == finalize_storyboard_submission(source, proposal)
    assert outcome.package.script_input_ref == source.script_input_ref
    assert len(provider.requests) == len(finalized) == 1
    assert isinstance(finalized[0], StoryboardSubmission)
    request = provider.requests[0]
    assert request.instructions == INSTRUCTIONS and request.input == serialize_context(source)
    assert director.prepare() == request.instructions + "\n" + request.input
    assert request.tool_choice == "required" and request.max_output_tokens == 1234
    assert source.model_dump_json() == before


def test_strict_schema_derived_from_actual_proposal_and_detached():
    director = VisualDirector(context())
    tools = director.tool_definitions()
    assert len(tools) == 1 and tools[0]["name"] == "submit_storyboard" and tools[0]["strict"] is True
    schema = tools[0]["parameters"]
    assert schema == strict_schema(StoryboardSubmission.model_json_schema())
    assert set(schema["properties"]) == set(schema["required"]) == {"title", "sections"}
    section = schema["$defs"]["StoryboardSectionSubmission"]
    assert set(section["properties"]) == set(section["required"]) == {"section_id", "title", "shots"}
    assert "title" not in schema["properties"]["title"]
    assert "title" not in section["properties"]["title"]
    shot_schema = schema["$defs"]["StoryboardShotSubmission"]
    assert set(shot_schema["properties"]) == set(shot_schema["required"]) == {
        "shot_id", "kind", "source_segment_id", "visual_description", "generation_prompt",
        "generation_method", "framing", "camera_motion", "estimated_duration_seconds"}
    for obj in (schema, section, shot_schema):
        assert obj["additionalProperties"] is False
    for forbidden in ("script_input_ref", "schema_version", "research_fact_ids", "story_beat_id", "evidence"):
        assert forbidden not in json.dumps(schema)
    tools[0]["name"] = "search"
    assert director.tool_definitions()[0]["name"] == "submit_storyboard"


@pytest.mark.parametrize("text", ["My finished plan", "", " ", '{"title":"fake package"}'])
def test_non_tool_responses_exhaust_without_transcript(text):
    provider = FakeProvider([response(text=text) for _ in range(3)])
    outcome = VisualDirector(context(), provider=provider).generate(max_steps=3)
    assert outcome.stop_reason == "LIMIT_REACHED" and outcome.package is None and outcome.steps == 3
    assert len(provider.requests) == 3
    assert provider.requests[0] == provider.requests[1] == provider.requests[2]
    assert provider.requests[2].input == serialize_context(context())
    assert set(outcome.model_dump()) == {"steps", "stop_reason", "package"}


def test_success_at_ceiling_and_fresh_invocations():
    provider = FakeProvider([response(text="Private reasoning"), response([call()]), response([call()])])
    director = VisualDirector(context(), provider=provider)
    first, second = director.generate(max_steps=2), director.generate(max_steps=2)
    assert first.steps == 2 and second.steps == 1 and first.package == second.package
    assert provider.requests[0] == provider.requests[1] == provider.requests[2]
    assert provider.requests[0].max_output_tokens == 6000
    assert "Private reasoning" not in second.model_dump_json()


@pytest.mark.parametrize("arguments", ["{", "[]", "null", "{}"])
def test_malformed_arguments_fail_without_retry(arguments):
    provider = FakeProvider([response([call(arguments)])])
    with pytest.raises(ValueError):
        VisualDirector(context(), provider=provider).generate()
    assert len(provider.requests) == 1


@pytest.mark.parametrize("failure", ["enum", "duration", "coverage", "unknown", "kind", "order", "duplicate", "section"])
def test_proposal_and_finalization_errors_propagate(failure):
    data = submission().model_dump(mode="json")
    shots = data["sections"][0]["shots"]
    if failure == "enum":
        shots[0]["generation_method"] = "UNKNOWN"
    elif failure == "duration":
        shots[0]["estimated_duration_seconds"] = 0
    elif failure == "coverage":
        shots.pop()
    elif failure == "unknown":
        shots[0]["source_segment_id"] = "unknown"
    elif failure == "kind":
        shots[0]["kind"] = "STRUCTURAL"
    elif failure == "order":
        shots.reverse()
    elif failure == "duplicate":
        shots[1]["shot_id"] = shots[0]["shot_id"]
    else:
        shots[0]["source_segment_id"] = "closing"
    provider = FakeProvider([response([call(json.dumps(data))])])
    with pytest.raises(ValueError):
        VisualDirector(context(), provider=provider).generate()
    assert len(provider.requests) == 1


@pytest.mark.parametrize("calls", [[call(), call()], [call(name="search")], [call(name="submit_script")]])
def test_multiple_and_unsupported_calls(calls):
    provider = FakeProvider([response(calls)])
    with pytest.raises(ValueError):
        VisualDirector(context(), provider=provider).generate()
    assert len(provider.requests) == 1


@pytest.mark.parametrize("changes", [{"status": "incomplete"}, {"status": "failed"},
                                     {"status": "in_progress"}, {"incomplete_reason": "max_output_tokens"}])
def test_incomplete_response_rejected(changes):
    with pytest.raises(ValueError, match="completed"):
        VisualDirector(context(), provider=FakeProvider([response([call()], **changes)])).generate()


def test_provider_errors_and_wrong_response_types():
    error = RuntimeError("provider failed")
    provider = FakeProvider([error])
    with pytest.raises(RuntimeError) as caught:
        VisualDirector(context(), provider=provider).generate()
    assert caught.value is error and len(provider.requests) == 1
    with pytest.raises(TypeError, match="ModelResponse"):
        VisualDirector(context(), provider=FakeProvider([{}])).generate()


@pytest.mark.parametrize("value", [0, -1, True, 1.5])
@pytest.mark.parametrize("parameter", ["max_steps", "max_output_tokens"])
def test_invalid_bounds_make_no_request(parameter, value):
    provider = FakeProvider([])
    with pytest.raises(ValueError, match="positive integer"):
        VisualDirector(context(), provider=provider).generate(**{parameter: value})
    assert provider.requests == []


def test_default_step_limit_and_optional_provider():
    director = VisualDirector(context())
    assert director.prepare()
    with pytest.raises(RuntimeError, match="injected ModelProvider"):
        director.generate()
    provider = FakeProvider([response() for _ in range(4)])
    assert VisualDirector(context(), provider=provider).generate().steps == 4


def test_no_artifact_or_real_api_access_and_no_semantic_rewriting(monkeypatch):
    from history_studio.storage.artifact_store import ArtifactStore
    from openai.resources.responses import Responses

    def forbidden(*args, **kwargs):
        pytest.fail("Generation must not access artifacts or a real API")

    for method in ("load", "load_latest", "save", "list_versions"):
        monkeypatch.setattr(ArtifactStore, method, forbidden)
    monkeypatch.setattr(Responses, "create", forbidden)
    result = VisualDirector(context(), provider=FakeProvider([response([call()])])).generate()
    assert result.package.sections[0].shots[0].generation_prompt == submission().sections[0].shots[0].generation_prompt


@pytest.mark.parametrize("reason,package", [("SUBMITTED", None),
    ("LIMIT_REACHED", finalize_storyboard_submission(context(), submission()))])
def test_outcome_success_and_limit_shape(reason, package):
    with pytest.raises(ValidationError):
        VisualDirectorGenerationOutcome(steps=1, stop_reason=reason, package=package)
