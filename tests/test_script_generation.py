import json

import pytest
from pydantic import ValidationError

from history_studio.model_io import ModelResponse, NativeToolCall
from history_studio.models import ScriptPackage
from history_studio.research.openai_provider import strict_schema
from history_studio.script import (
    ScriptWriter, ScriptSubmission, ScriptGenerationOutcome, finalize_script_submission,
)
from history_studio.script.preparation import INSTRUCTIONS, serialize_context
from test_script_writer import context
from test_script_submission import submission, section, segment


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


def call(arguments=None, name="submit_script"):
    return NativeToolCall(call_id="call_1", name=name,
        arguments=submission().model_dump_json() if arguments is None else arguments)


def response(calls=None, text="", **changes):
    return ModelResponse(tool_calls=calls or [], text=text, usage={}, **changes)


def test_terminal_native_submission_uses_p5d_and_exact_provenance(monkeypatch):
    import history_studio.script.agent as agent
    source = context()
    before = source.model_dump_json()
    provider = FakeProvider([response([call()]), RuntimeError("must stop")])
    finalized = []

    def spy(ctx, proposal):
        finalized.append(proposal)
        return finalize_script_submission(ctx, proposal)

    monkeypatch.setattr(agent, "finalize_script_submission", spy)
    writer = ScriptWriter(source, provider=provider)
    outcome = writer.generate(max_output_tokens=1234)
    assert outcome.stop_reason == "SUBMITTED" and outcome.steps == 1
    assert isinstance(outcome.package, ScriptPackage)
    assert outcome.package == finalize_script_submission(source, submission())
    assert outcome.package.story_input_ref == source.story_input_ref
    assert len(provider.requests) == len(finalized) == 1
    assert isinstance(finalized[0], ScriptSubmission)
    request = provider.requests[0]
    assert request.instructions == INSTRUCTIONS and request.input == serialize_context(source)
    assert writer.prepare() == request.instructions + "\n" + request.input
    assert request.tool_choice == "required" and request.max_output_tokens == 1234
    assert [t["name"] for t in request.tools] == ["submit_script"]
    assert request.tools[0]["strict"] is True
    assert request.tools[0]["parameters"] == strict_schema(ScriptSubmission.model_json_schema())
    assert source.model_dump_json() == before


def test_schema_contains_only_proposal_fields_and_detached_tool_definitions():
    writer = ScriptWriter(context())
    tools = writer.tool_definitions()
    schema = tools[0]["parameters"]
    encoded = json.dumps(schema)
    for forbidden in ("story_input_ref", "VerificationStatus", "StoryFactUse", "HistoricalTime",
                      "source_id", "schema_version", "approval_identity", "evidence"):
        assert forbidden not in encoded
    assert "sections" in schema["properties"]
    assert schema["additionalProperties"] is False
    tools[0]["name"] = "search"
    assert writer.tool_definitions()[0]["name"] == "submit_script"


@pytest.mark.parametrize("text", ["My finished narration", "", " ", '{"title":"fake package"}'])
def test_text_or_empty_response_retries_to_limit_without_package(text):
    provider = FakeProvider([response(text=text) for _ in range(3)])
    outcome = ScriptWriter(context(), provider=provider).generate(max_steps=3)
    assert outcome.stop_reason == "LIMIT_REACHED" and outcome.package is None and outcome.steps == 3
    assert len(provider.requests) == 3
    assert provider.requests[0] == provider.requests[1] == provider.requests[2]
    assert "My finished narration" not in provider.requests[2].input
    assert set(outcome.model_dump()) == {"steps", "stop_reason", "package"}


def test_submission_at_ceiling_and_fresh_invocations_without_transcripts():
    provider = FakeProvider([response(text="Private reasoning"), response([call()]), response([call()])])
    writer = ScriptWriter(context(), provider=provider)
    first, second = writer.generate(max_steps=2), writer.generate(max_steps=2)
    assert first.steps == 2 and second.steps == 1
    assert first.package == second.package
    assert provider.requests[0] == provider.requests[1] == provider.requests[2]
    assert provider.requests[0].max_output_tokens == 6000
    assert "Private reasoning" not in second.model_dump_json()


@pytest.mark.parametrize("arguments", ["{", "[]", "null", "{}", '{"title":"x","status":"VERIFIED"}'])
def test_malformed_or_invalid_submission_propagates_without_retry(arguments):
    provider = FakeProvider([response([call(arguments)])])
    with pytest.raises(ValueError):
        ScriptWriter(context(), provider=provider).generate()
    assert len(provider.requests) == 1


@pytest.mark.parametrize("beat_id,fact_id", [("unknown", "f0"), ("beat_01", "unknown"),
                                           ("transition_01", "f0")])
def test_finalization_failure_propagates(beat_id, fact_id):
    proposal = submission(sections=[section(segments=[segment(grounding=dict(
        story_beat_id=beat_id, research_fact_ids=[fact_id]))])])
    provider = FakeProvider([response([call(proposal.model_dump_json())])])
    with pytest.raises(ValueError):
        ScriptWriter(context(), provider=provider).generate()
    assert len(provider.requests) == 1


@pytest.mark.parametrize("calls", [[call(), call()], [call(name="read_source")],
                                   [call(name="search")], [call(name="submit_story")]])
def test_multiple_and_unsupported_calls_fail_closed(calls):
    provider = FakeProvider([response(calls)])
    with pytest.raises(ValueError):
        ScriptWriter(context(), provider=provider).generate()
    assert len(provider.requests) == 1


@pytest.mark.parametrize("changes", [{"status": "incomplete"}, {"status": "failed"},
                                     {"status": "in_progress"}, {"incomplete_reason": "max_output_tokens"}])
def test_incomplete_response_propagates(changes):
    provider = FakeProvider([response([call()], **changes)])
    with pytest.raises(ValueError, match="completed"):
        ScriptWriter(context(), provider=provider).generate()


def test_provider_errors_and_wrong_response_type_propagate():
    error = RuntimeError("provider failure")
    provider = FakeProvider([error])
    with pytest.raises(RuntimeError) as caught:
        ScriptWriter(context(), provider=provider).generate()
    assert caught.value is error and len(provider.requests) == 1
    with pytest.raises(TypeError, match="ModelResponse"):
        ScriptWriter(context(), provider=FakeProvider([{}])).generate()


@pytest.mark.parametrize("value", [0, -1, True, 1.5])
@pytest.mark.parametrize("parameter", ["max_steps", "max_output_tokens"])
def test_invalid_bounds_make_no_request(parameter, value):
    provider = FakeProvider([])
    with pytest.raises(ValueError, match="positive integer"):
        ScriptWriter(context(), provider=provider).generate(**{parameter: value})
    assert provider.requests == []


def test_default_step_bound_and_provider_optional_for_preparation():
    writer = ScriptWriter(context())
    assert writer.prepare()
    with pytest.raises(RuntimeError, match="injected ModelProvider"):
        writer.generate()
    provider = FakeProvider([response() for _ in range(4)])
    assert ScriptWriter(context(), provider=provider).generate().steps == 4
    assert len(provider.requests) == 4


def test_no_store_lookup_semantic_rewriting_or_duration_behavior(monkeypatch):
    from history_studio.storage.artifact_store import ArtifactStore

    def forbidden(*args, **kwargs):
        pytest.fail("Generation must not access artifacts")

    for method in ("load", "load_latest", "save", "list_versions"):
        monkeypatch.setattr(ArtifactStore, method, forbidden)
    proposal = submission(sections=[section(segments=[segment(narration="Unchecked prose", grounding=dict(
        story_beat_id="beat_01", research_fact_ids=["f1"]))])])
    provider = FakeProvider([response([call(proposal.model_dump_json())])])
    accepted = ScriptWriter(context(), provider=provider).generate().package.sections[0].segments[0]
    assert accepted.narration == "Unchecked prose" and accepted.grounding.research_fact_ids == ("f1",)


@pytest.mark.parametrize("reason,package", [("SUBMITTED", None),
    ("LIMIT_REACHED", finalize_script_submission(context(), submission()))])
def test_outcome_cannot_confuse_success_with_bounded_stop(reason, package):
    with pytest.raises(ValidationError):
        ScriptGenerationOutcome(steps=1, stop_reason=reason, package=package)


def test_same_context_and_fake_submission_equivalent_output():
    first = ScriptWriter(context(), provider=FakeProvider([response([call()])])).generate()
    second = ScriptWriter(context(), provider=FakeProvider([response([call()])])).generate()
    assert first == second
