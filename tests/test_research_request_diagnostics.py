import json
from types import SimpleNamespace

import httpx
import pytest
from openai import OpenAI

from history_studio.cli import main
from history_studio.research.agent import ResearchAgent
from history_studio.research.diagnostics import RequestDiagnostic, request_diagnostic
from history_studio.research.openai_provider import OpenAIConfiguration, OpenAIResearchProvider
from history_studio.storage import ArtifactStore
from test_openai_provider import response
from test_research_agent import FakeProvider, FakeTools, settings, setup_run, ledger, calls, update

SECRET = "sk-secret-authorization-private-reasoning"


@pytest.mark.parametrize("mode, category, code, error_class", [
    ("timeout", "connectivity", "request_timeout", "APITimeoutError"),
    ("connect", "connectivity", "connection_failed", "APIConnectionError"),
    ("400", "api", "provider_api_error", "BadRequestError"),
    ("401", "api", "provider_api_error", "AuthenticationError"),
    ("429", "api", "provider_api_error", "RateLimitError"),
    ("500", "api", "provider_api_error", "InternalServerError"),
    ("incomplete", "model_response", "incomplete_model_response", "ModelResponseError"),
    ("no_call", "model_response", "expected_one_native_function_call", "ModelResponseError"),
    ("json", "model_response", "invalid_function_arguments_json", "ModelResponseError"),
    ("shape", "model_response", "invalid_model_reply", "ModelResponseError"),
])
def test_request_failures_are_diagnosed_without_secrets(tmp_path, capsys, mode, category, code, error_class):
    def handler(request):
        if mode == "timeout":
            raise httpx.ReadTimeout(SECRET, request=request)
        if mode == "connect":
            raise httpx.ConnectError(SECRET, request=request)
        if mode.isdigit():
            return httpx.Response(int(mode), json={"error": {
                "message": SECRET, "type": SECRET, "param": SECRET,
                "code": "invalid_json_schema" if mode == "400" else SECRET}},
                headers={"x-request-id": SECRET})
        raw = response([])
        if mode == "incomplete":
            raw.update(status="incomplete", incomplete_details={"reason": "max_output_tokens"})
        elif mode in ("json", "shape"):
            raw["output"] = [dict(type="function_call", id="fc", call_id="c", name="checkpoint_research",
                arguments=SECRET if mode == "json" else json.dumps([SECRET]), status="completed")]
        return httpx.Response(200, json=raw)

    project, store = setup_run(tmp_path)
    events = []
    with OpenAI(api_key=SECRET, http_client=httpx.Client(transport=httpx.MockTransport(handler)),
                max_retries=0) as client:
        package = ResearchAgent(OpenAIResearchProvider(client, OpenAIConfiguration()), FakeTools(),
                                settings(), events.append).run(project, store)
    assert package.progress.stop_reason == "research_model_request_failed"
    diagnostic = ArtifactStore(store.project_dir / ".runtime").load_latest("diagnostics", RequestDiagnostic)
    assert (diagnostic.category, diagnostic.code, diagnostic.error_class) == (category, code, error_class)
    assert diagnostic.iteration == 1 and diagnostic.turn == 1
    assert diagnostic.pending_request
    if mode.isdigit():
        assert diagnostic.http_status == int(mode)
        assert diagnostic.provider_code == ("invalid_json_schema" if mode == "400" else None)
    if mode == "incomplete":
        assert diagnostic.response_status == "incomplete"
        assert diagnostic.incomplete_reason == "max_output_tokens"
    usage = ledger(store)
    assert usage.pending_request and usage.unknown_usage
    assert usage.model_calls == 1 and usage.committed_budget_usd > 0
    assert usage.estimated_total_cost_usd == 0  # Accounting semantics unchanged, even on invalid responses.
    assert code in str(events)
    assert main(["--projects-dir", str(tmp_path), "status", "test"]) == 0
    output = capsys.readouterr().out
    assert code in output and error_class in output
    assert SECRET not in output + str(events)
    assert all(SECRET not in p.read_text(encoding="utf-8") for p in store.project_dir.rglob("*.json"))


def test_latest_request_diagnostic_supersedes_old_validation_and_budget_survives_resume(tmp_path, capsys):
    project, store = setup_run(tmp_path)
    bad = update()
    bad.arguments["facts"][0]["claim"] = "one; two"
    ResearchAgent(FakeProvider(calls()[:2] + [bad, RuntimeError(SECRET)]), FakeTools(), settings()).run(project, store)
    runtime = ArtifactStore(store.project_dir / ".runtime")
    assert runtime.list_versions("diagnostics") == [1, 2]
    assert runtime.load_latest("diagnostics", RequestDiagnostic).turn == 4
    before = ledger(store)
    assert main(["--projects-dir", str(tmp_path), "status", "test"]) == 0
    output = capsys.readouterr().out
    assert "unexpected_exception" in output
    assert "Checkpoint validation rejected" not in output
    # Accumulated reservation disallows the next request after resume, without calling the provider.
    provider = FakeProvider([], cost=0.001)
    result = ResearchAgent(provider, FakeTools(), settings(hard_budget_usd=before.committed_budget_usd,
        soft_budget_usd=before.committed_budget_usd / 2)).run(project, store)
    after = ledger(store)
    assert result.progress.stop_reason == "hard_budget"
    assert provider.calls == 0
    assert after.run_id == before.run_id
    assert after.committed_budget_usd == before.committed_budget_usd
    assert after.model_calls == before.model_calls
    assert after.unknown_usage


def test_arbitrary_exception_class_name_is_not_persisted():
    exc = type(SECRET, (Exception,), {})(SECRET)
    diagnostic = request_diagnostic(exc, run_id="test", iteration=1, turn=1, pending_request=True)
    assert diagnostic.error_class == "Exception"
    assert SECRET not in diagnostic.model_dump_json()



def bad_request(body):
    from openai import BadRequestError
    request = httpx.Request("POST", "https://api.openai.com/v1/responses",
        headers={"Authorization": "Bearer header-only-secret", "X-Private": "private-header"},
        json={"input": "private-request-body", "reasoning": "private-reasoning"})
    response = httpx.Response(400, request=request, json={"error": body},
                             headers={"X-Private": "private-response-header"})
    return BadRequestError("full exception body must not survive", response=response, body=body)


def test_structured_bad_request_details_reach_saved_diagnostics_and_status(tmp_path, capsys):
    message = ("Invalid schema for function 'checkpoint_research': In context "
               "('$defs', 'HistoricalTime', 'properties', 'precision'), '$ref' is not permitted.")
    body = dict(type="invalid_request_error", code="new_schema_error", param="tools[2].parameters",
                message=message, reasoning="hidden-body-reasoning", request="body-echo")
    exc = bad_request(body)
    assert exc.type == body["type"] and exc.param == body["param"]
    project, store = setup_run(tmp_path)
    events = []
    ResearchAgent(FakeProvider([exc]), FakeTools(), settings(), events.append).run(project, store)
    diagnostic = ArtifactStore(store.project_dir / ".runtime").load_latest("diagnostics", RequestDiagnostic)
    assert diagnostic.http_status == 400
    assert diagnostic.provider_type == "invalid_request_error"
    assert diagnostic.provider_code == "new_schema_error"
    assert diagnostic.provider_param == "tools[2].parameters"
    assert diagnostic.provider_message == message
    assert main(["--projects-dir", str(tmp_path), "status", "test"]) == 0
    output = capsys.readouterr().out + str(events) + diagnostic.model_dump_json()
    for text in (message, "new_schema_error", "tools[2].parameters", "invalid_request_error"):
        assert text in output
    for secret in ("header-only-secret", "private-header", "private-request-body", "private-reasoning",
                   "private-response-header", "hidden-body-reasoning", "body-echo", "full exception body"):
        assert secret not in output


@pytest.mark.parametrize("suffix", [
    " Authorization: Bearer private-leak", " request_headers: private-leak",
    " request body: private-leak", " response_body: private-leak",
    " hidden reasoning: private-leak", " <think>private-leak</think>",
    ' {"input": "private-leak"}', " api_key=private-leak", " Bearer private-leak",
])
def test_provider_message_redacts_payloads_and_credentials(suffix):
    exc = bad_request(dict(message="Invalid schema." + suffix, type="invalid_request_error"))
    diagnostic = request_diagnostic(exc, run_id="test", iteration=1, turn=1, pending_request=True)
    assert "Invalid schema." in diagnostic.provider_message
    assert "private-leak" not in diagnostic.model_dump_json()


def test_provider_message_bounds_control_characters_and_key_redaction():
    message = "Invalid schema.\x1b[31m\r\n\u202e " + SECRET + " " + "details " * 2000
    diagnostic = request_diagnostic(bad_request(dict(message=message)), run_id="test",
                                    iteration=1, turn=1, pending_request=True)
    assert len(diagnostic.provider_message) == 1000
    assert diagnostic.provider_message.endswith("...")
    assert SECRET not in diagnostic.provider_message
    assert not any(c in diagnostic.provider_message for c in ("\x1b", "\r", "\n", "\u202e"))


@pytest.mark.parametrize("body", [None, "raw response secret", {"message": {"secret": "value"}}, {}])
def test_missing_structured_message_never_falls_back_to_raw_exception(body):
    diagnostic = request_diagnostic(bad_request(body), run_id="test", iteration=1, turn=1, pending_request=True)
    assert diagnostic.provider_message is None
    assert "secret" not in diagnostic.model_dump_json()


def test_unsafe_or_unbounded_metadata_is_omitted():
    body = dict(type=SECRET, code="x" * 129, param="Authorization: Bearer private-leak", message="Invalid schema")
    diagnostic = request_diagnostic(bad_request(body), run_id="test", iteration=1, turn=1, pending_request=True)
    assert diagnostic.provider_type is diagnostic.provider_code is diagnostic.provider_param is None
    assert "private-leak" not in diagnostic.model_dump_json()
