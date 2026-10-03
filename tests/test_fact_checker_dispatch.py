"""Explicit fake tool dispatch; no model round trip or evidence acceptance."""
import pytest
from pydantic import ValidationError

from history_studio.model_io import NativeToolCall
from history_studio.models import build_verification_context
from history_studio.research.boundaries import ToolObservation
from history_studio.verification import FactChecker
from test_fact_checker_decision import FakeModelProvider
from test_verification_context import package_data, source_data
from history_studio.models.sources import SourceReference


class RecordingTools:
    def __init__(self):
        self.calls = []
        source = SourceReference(**source_data("SRC-new"))
        self.search_result = ToolObservation(kind="search", sources=[source])
        self.read_result = None

    def search_reserve_cost(self, query):
        pytest.fail("Dispatch must not add cost reservation")

    def search_web(self, query):
        self.calls.append(("search", query))
        return self.search_result

    def read_source(self, source, max_chars):
        self.calls.append(("read", source.model_dump(mode="json"), max_chars))
        self.read_result = ToolObservation(kind="source", source_id=source.source_id,
            sources=[source.model_copy(deep=True)], text="Candidate source text.", truncated=True)
        return self.read_result


def native(name="search_web", arguments=None, call_id="call-correlation"):
    return NativeToolCall(call_id=call_id, name=name,
                          arguments=arguments if arguments is not None else '{"query":" independent date "}')


def checker_with_tools():
    context = build_verification_context(package_data(), "RF-target")
    tools = RecordingTools()
    provider = FakeModelProvider(RuntimeError("Dispatch must not call the model"))
    return FactChecker(context, tools=tools, provider=provider), tools, provider


def test_search_validates_once_and_returns_same_observation_without_model_or_files(tmp_path, monkeypatch):
    monkeypatch.chdir(tmp_path)
    checker, tools, provider = checker_with_tools()
    prepared = checker.prepare()
    call = native(call_id="SPAN-not-provenance")
    before = call.model_dump_json()
    result = checker.execute_tool_call(call)
    assert tools.calls == [("search", "independent date")]
    assert result is tools.search_result
    assert provider.requests == []
    assert call.model_dump_json() == before
    assert checker.prepare() == prepared
    assert list(tmp_path.iterdir()) == []
    assert "verification_evidence" not in result.model_dump() and "verification_id" not in result.model_dump()


def test_read_validates_once_resolves_metadata_and_preserves_observation():
    checker, tools, provider = checker_with_tools()
    prepared = checker.prepare()
    result = checker.execute_tool_call(native("read_source", '{"source_id":"SRC-A"}', "SRC-B"), max_chars=321)
    assert len(tools.calls) == 1
    kind, source, limit = tools.calls[0]
    assert kind == "read" and source["source_id"] == "SRC-A" and limit == 321
    expected = build_verification_context(package_data(), "RF-target")
    assert source == next(s.model_dump(mode="json") for s in expected.sources if s.source_id == "SRC-A")
    assert result is tools.read_result and result.text == "Candidate source text." and result.truncated
    assert provider.requests == [] and checker.prepare() == prepared
    assert "verification_evidence" not in result.model_dump() and "verification_id" not in result.model_dump()


def test_explicit_search_metadata_can_be_read_without_becoming_model_context_or_evidence():
    checker, tools, provider = checker_with_tools()
    prepared = checker.prepare()
    checker.execute_tool_call(native())
    # A second explicit caller invocation, not a loop or model round trip.
    result = checker.execute_tool_call(native("read_source", '{"source_id":"SRC-new"}'))
    assert [c[0] for c in tools.calls] == ["search", "read"]
    assert result.source_id == "SRC-new"
    assert provider.requests == [] and checker.prepare() == prepared


@pytest.mark.parametrize("name, arguments, error, message", [
    ("checkpoint_research", "{}", ValueError, "Unsupported"),
    ("execute_shell", "{}", ValueError, "Unsupported"),
    ("search_web", "{broken", ValueError, "Invalid tool arguments JSON"),
    *[("search_web", arg, ValueError, "JSON object") for arg in ("[]", "null", '"text"', "1", "true")],
    ("search_web", "{}", ValidationError, "query"),
    ("search_web", '{"query":5}', ValidationError, "query"),
    ("search_web", '{"query":" "}', ValidationError, "query"),
    ("search_web", '{"query":"q","extra":true}', ValidationError, "extra"),
    ("read_source", "{}", ValidationError, "source_id"),
    ("read_source", '{"source_id":5}', ValidationError, "source_id"),
    ("read_source", '{"source_id":"../bad"}', ValidationError, "source_id"),
    ("read_source", '{"source_id":"SRC-A","extra":true}', ValidationError, "extra"),
    ("read_source", '{"source_id":"SRC-unknown"}', ValueError, "original or discovered"),
])
def test_invalid_calls_never_execute_tools_or_provider(name, arguments, error, message):
    checker, tools, provider = checker_with_tools()
    with pytest.raises(error, match=message):
        checker.execute_tool_call(native(name, arguments))
    assert tools.calls == [] and provider.requests == []


def test_no_tools_and_multiple_call_input_fail_clearly():
    context = build_verification_context(package_data(), "RF-target")
    with pytest.raises(RuntimeError, match="configured ResearchTools"):
        FactChecker(context).execute_tool_call(native())
    checker, tools, provider = checker_with_tools()
    with pytest.raises(TypeError, match="one NativeToolCall"):
        checker.execute_tool_call([native(), native()])
    assert tools.calls == [] and provider.requests == []


@pytest.mark.parametrize("max_chars", [0, -1, True, "100"])
def test_invalid_read_bound_prevents_execution(max_chars):
    checker, tools, provider = checker_with_tools()
    with pytest.raises(ValueError, match="positive integer"):
        checker.execute_tool_call(native("read_source", '{"source_id":"SRC-A"}'), max_chars=max_chars)
    assert tools.calls == [] and provider.requests == []


def test_tool_failure_is_propagated_without_retry_or_historical_verdict():
    checker, tools, provider = checker_with_tools()
    failure = RuntimeError("offline tool failure")
    def failed(query):
        tools.calls.append(("search", query))
        raise failure
    tools.search_web = failed
    with pytest.raises(RuntimeError) as caught:
        checker.execute_tool_call(native())
    assert caught.value is failure
    assert len(tools.calls) == 1 and provider.requests == []
