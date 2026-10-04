"""Terminal proposals remain separate from accepted evidence and final verification results."""
import json

import pytest
from pydantic import ValidationError

from history_studio.models.verification import VerificationEvidence, VerificationResult
from history_studio.model_io import ModelResponse, Usage
from history_studio.openai_model import OpenAIModelProvider
from history_studio.verification import FactChecker, VerificationSubmission
from history_studio.verification.submission import VerificationEvidenceSelection, VerificationSubmissionInput
from test_fact_checker_decision import input_context
from test_fact_checker_dispatch import RecordingTools, native
from test_fact_checker_investigation import SequenceProvider, decision


def payload(status="UNVERIFIED", source_id="SRC-A"):
    support = status in {"VERIFIED", "PARTIALLY_VERIFIED", "DISPUTED"}
    conflict = status in {"DISPUTED", "REJECTED"}
    return dict(status=status,
        verification_evidence=[dict(source_id=source_id, locator="Section 2")] if support else [],
        contradiction_evidence=[dict(source_id=source_id, locator=None)] if conflict else [],
        unresolved_issues=["Insufficient independent material"] if status in {"UNVERIFIED", "PARTIALLY_VERIFIED"} else [],
        independence_note="Underlying origin remains an Agent assessment", rationale="Concise judgment")


def terminal(data):
    return native("submit_verification", json.dumps(data, ensure_ascii=False))


@pytest.mark.parametrize("status", ["VERIFIED", "PARTIALLY_VERIFIED", "DISPUTED", "REJECTED", "UNVERIFIED"])
def test_valid_status_submission_binds_target_without_materializing_evidence(status):
    context = input_context()
    before = context.model_dump_json()
    checker = FactChecker(context)
    result = checker.submit_verification(terminal(payload(status)))
    assert isinstance(result, VerificationSubmission) and not isinstance(result, VerificationResult)
    assert result.research_fact_id == context.target_fact.fact_id
    assert result.claim_snapshot == context.target_fact.claim
    assert result.status.value == status
    for selection in result.verification_evidence + result.contradiction_evidence:
        assert isinstance(selection, VerificationEvidenceSelection)
        assert not isinstance(selection, VerificationEvidence)
        assert "excerpt" not in selection.model_dump()
    assert context.model_dump_json() == before
    assert "verification_id" not in result.model_dump()


@pytest.mark.parametrize("status,changed", [
    ("VERIFIED", {"verification_evidence": []}),
    ("PARTIALLY_VERIFIED", {"verification_evidence": []}),
    ("PARTIALLY_VERIFIED", {"unresolved_issues": [], "contradiction_evidence": []}),
    ("DISPUTED", {"verification_evidence": []}),
    ("DISPUTED", {"contradiction_evidence": []}),
    ("REJECTED", {"contradiction_evidence": []}),
    ("UNVERIFIED", {"unresolved_issues": []}),
])
def test_status_structure_rejects_invalid_proposals(status, changed):
    data = payload(status)
    data.update(changed)
    with pytest.raises(ValidationError):
        FactChecker(input_context()).submit_verification(terminal(data))


def test_partial_with_contradiction_and_no_unresolved_issue_is_valid():
    data = payload("PARTIALLY_VERIFIED")
    data.update(unresolved_issues=[], contradiction_evidence=[dict(source_id="SRC-A")])
    assert FactChecker(input_context()).submit_verification(terminal(data)).status.value == "PARTIALLY_VERIFIED"


@pytest.mark.parametrize("issue", ["", " \n "])
def test_unverified_requires_nonblank_unresolved_issue(issue):
    data = payload()
    data["unresolved_issues"] = [issue]
    provider = SequenceProvider([decision(terminal(data))])
    tools = RecordingTools()
    with pytest.raises(ValidationError):
        FactChecker(input_context(), tools=tools, provider=provider).investigate()
    assert len(provider.requests) == 1 and tools.calls == []


@pytest.mark.parametrize("arguments", ["{bad", "[]", "null", '"text"'])
def test_malformed_terminal_json_fails_without_tool_execution(arguments):
    provider = SequenceProvider([decision(native("submit_verification", arguments))])
    tools = RecordingTools()
    with pytest.raises(ValueError):
        FactChecker(input_context(), tools=tools, provider=provider).investigate()
    assert len(provider.requests) == 1 and tools.calls == []


@pytest.mark.parametrize("field", ["status", "independence_note", "rationale"])
def test_missing_semantic_fields_rejected(field):
    data = payload()
    del data[field]
    with pytest.raises(ValidationError):
        FactChecker(input_context()).submit_verification(terminal(data))


@pytest.mark.parametrize("field,value", [
    ("research_fact_id", "RF-foreign"), ("claim_snapshot", "Changed target claim"),
    ("verification_id", "V-invented"), ("status", "MADE_UP"),
    ("independence_note", " "), ("rationale", 123),
])
def test_identity_override_and_malformed_semantic_fields_rejected(field, value):
    data = payload()
    data[field] = value
    with pytest.raises(ValidationError):
        FactChecker(input_context()).submit_verification(terminal(data))


@pytest.mark.parametrize("override", [dict(excerpt="invented quote"), dict(span_id="SPAN-invented"),
                                     dict(source_version="VER-invented"), dict(source_id="../bad")])
def test_model_cannot_supply_canonical_evidence(override):
    data = payload("VERIFIED")
    data["verification_evidence"][0].update(override)
    with pytest.raises(ValidationError):
        FactChecker(input_context()).submit_verification(terminal(data))


def test_unknown_source_selection_rejected_without_evidence_authorization():
    with pytest.raises(ValueError, match="unknown source"):
        FactChecker(input_context()).submit_verification(terminal(payload("VERIFIED", "SRC-unknown")))


def test_terminal_stops_at_last_step_without_tool_or_provider_round_trip(tmp_path, monkeypatch):
    monkeypatch.chdir(tmp_path)
    final = decision(terminal(payload("VERIFIED", "SRC-new")))
    provider = SequenceProvider([decision(native()),
        decision(native("read_source", '{"source_id":"SRC-new"}')), final])
    tools = RecordingTools()
    outcome = FactChecker(input_context(), tools=tools, provider=provider).investigate(max_steps=3)
    assert outcome.stop_reason.value == "SUBMITTED" and outcome.steps == 3
    assert outcome.final_response is final and len(provider.requests) == 3
    assert len(tools.calls) == 2 and outcome.observations == [tools.search_result, tools.read_result]
    assert isinstance(outcome.submission, VerificationSubmission)
    assert not isinstance(outcome.submission, VerificationResult)
    assert outcome.submission.verification_evidence[0].source_id == "SRC-new"
    assert outcome.submission.claim_snapshot == input_context().target_fact.claim
    assert list(tmp_path.iterdir()) == []


def test_immediate_submission_executes_zero_external_tools_and_preserves_unverified_as_explicit_judgment():
    final = decision(terminal(payload()))
    provider = SequenceProvider([final])
    tools = RecordingTools()
    outcome = FactChecker(input_context(), tools=tools, provider=provider).investigate()
    assert len(provider.requests) == 1 and tools.calls == []
    assert outcome.final_response is final and outcome.submission.status.value == "UNVERIFIED"
    assert outcome.stop_reason.value == "SUBMITTED" and outcome.observations == []


def test_multiple_actions_including_submit_rejected_before_execution():
    provider = SequenceProvider([decision(native(), terminal(payload()))])
    tools = RecordingTools()
    with pytest.raises(ValueError, match="exactly one"):
        FactChecker(input_context(), tools=tools, provider=provider).investigate()
    assert len(provider.requests) == 1 and tools.calls == []


def test_terminal_is_not_external_dispatch_and_other_runtime_stops_have_no_submission():
    tools = RecordingTools()
    checker = FactChecker(input_context(), tools=tools)
    with pytest.raises(ValueError, match="Unsupported"):
        checker.execute_tool_call(terminal(payload()))
    assert tools.calls == []
    for response, reason in [(decision(native()), "LIMIT_REACHED"),
                              (decision(text="VERIFIED"), "MODEL_TEXT"), (decision(), "NO_TOOL_CALL")]:
        provider = SequenceProvider([response])
        outcome = FactChecker(input_context(), tools=tools, provider=provider).investigate(max_steps=1)
        assert outcome.stop_reason.value == reason and outcome.submission is None


def test_invalid_terminal_structure_in_loop_is_not_converted_to_unverified():
    data = payload("VERIFIED")
    data["verification_evidence"] = []
    provider = SequenceProvider([decision(terminal(data))])
    tools = RecordingTools()
    with pytest.raises(ValidationError):
        FactChecker(input_context(), tools=tools, provider=provider).investigate()
    assert len(provider.requests) == 1 and tools.calls == []


def test_final_model_request_exposes_strict_terminal_schema_and_no_runtime_identity():
    provider = SequenceProvider([decision(terminal(payload()))])
    checker = FactChecker(input_context(), tools=RecordingTools(), provider=provider)
    checker.investigate()
    request = provider.requests[0]
    body = OpenAIModelProvider(None, "fake", 0, 0).request_body(request)
    assert [t["name"] for t in body["tools"]] == ["search_web", "read_source", "submit_verification"]
    assert body["tool_choice"] == "required"
    schema = body["tools"][-1]["parameters"]
    assert set(schema["properties"]) == set(VerificationSubmissionInput.model_fields)
    assert not {"claim_snapshot", "research_fact_id", "verification_id"} & set(schema["properties"])
    def inspect(node):
        if isinstance(node, dict):
            if "$ref" in node:
                assert len(node) == 1
            if node.get("type") == "object":
                assert node["additionalProperties"] is False
                assert set(node["required"]) == set(node["properties"])
            for value in node.values():
                inspect(value)
        elif isinstance(node, list):
            for value in node:
                inspect(value)
    inspect(schema)
    assert '"excerpt":' not in json.dumps(schema)
    assert "submit_verification" in request.instructions
