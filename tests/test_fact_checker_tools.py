"""Offline dependency/schema boundary, not an autonomous investigation loop."""
import json

import pytest
from pydantic import ValidationError

from history_studio.models import VerificationContext, build_verification_context
from history_studio.models.verification import VerificationEvidence, VerificationResult
from history_studio.research.actions import ReadRequest, SearchRequest
from history_studio.research.boundaries import ToolObservation
from history_studio.research.openai_provider import native_tools
from history_studio.verification import FactChecker
from history_studio.verification.context import INSTRUCTIONS
from test_research_agent import FakeTools, SOURCE
from test_verification_context import package_data


def context_data():
    # Only the already-projected context crosses the Agent boundary.
    return VerificationContext.model_validate_json(
        build_verification_context(package_data(), "RF-target").model_dump_json())


def test_shared_fake_dependencies_are_injected_without_execution_or_model_calls(monkeypatch):
    from openai.resources.responses import Responses
    def forbidden(*args, **kwargs):
        pytest.fail("Tool-boundary preparation must not request a model")
    monkeypatch.setattr(Responses, "create", forbidden)
    context = context_data()
    before = context.model_dump_json()
    tools = FakeTools()
    checker = FactChecker(context, tools=tools)
    assert checker.tools is tools
    definitions = checker.tool_definitions()
    assert [d["name"] for d in definitions] == ["search_web", "read_source"]
    assert tools.queries == [] and tools.reads == []
    assert checker.prepare() == FactChecker(context).prepare()
    assert checker.prepare() == checker.prepare()
    assert context.model_dump_json() == before
    state = json.loads(checker.prepare()[len(INSTRUCTIONS):])
    assert state == context.model_dump(mode="json")
    assert "RF-other" not in checker.prepare()
    assert "RESEARCH_INPUT_NOT_INDEPENDENT_VERIFICATION" in checker.prepare()
    assert all(not hasattr(checker, name) for name in ("run", "verify", "complete", "provider"))


def test_search_read_fake_outputs_remain_shared_observations_not_verification():
    context = context_data()
    before = context.model_dump_json()
    checker = FactChecker(context, tools=FakeTools())
    # Invoke only the injected fake directly: there is no FactChecker dispatch path.
    search = checker.tools.search_web(SearchRequest(query="Target date contradiction").query)
    source_id = ReadRequest(source_id=search.sources[0].source_id).source_id
    source = next(s for s in search.sources if s.source_id == source_id)
    read = checker.tools.read_source(source, 1000)
    assert isinstance(search, ToolObservation) and search.kind == "search"
    assert isinstance(read, ToolObservation) and read.kind == "source"
    assert read.source_id == SOURCE.source_id and read.text
    for result in (search, read):
        assert not isinstance(result, (VerificationEvidence, VerificationResult))
        assert "verification_evidence" not in result.model_dump()
    assert context.model_dump_json() == before
    assert json.loads(checker.prepare()[len(INSTRUCTIONS):]) == context.model_dump(mode="json")


def test_definitions_reuse_existing_request_schemas_and_normalization():
    definitions = FactChecker(context_data(), tools=FakeTools()).tool_definitions()
    research = {d["name"]: d for d in native_tools()}
    for definition, contract in zip(definitions, (SearchRequest, ReadRequest)):
        assert definition["parameters"] == research[definition["name"]]["parameters"]
        schema = definition["parameters"]
        assert definition["strict"] is True
        assert schema["additionalProperties"] is False
        assert set(schema["required"]) == set(contract.model_fields)
        assert set(schema["properties"]) == set(contract.model_fields)
    assert all(d["name"] != "checkpoint_research" for d in definitions)


def test_tool_descriptions_keep_independence_and_claim_boundary_explicit():
    definitions = FactChecker(context_data(), tools=FakeTools()).tool_definitions()
    search, read = [d["description"] for d in definitions]
    assert "independent evidence" in search and "do not research the whole topic" in search
    for description in (search, read):
        assert "target atomic claim only" in description
        assert "contradiction or qualification" in description or "contradiction, or qualification" in description
        assert "VerificationEvidence" in description
        assert "checkpoint_research" not in description
    assert "not automatically accepted" in read
    assert "Agent will select evidence" in read and "Python will later validate/extract" in read


def test_definition_results_are_detached_and_no_tools_is_backward_compatible():
    context = context_data()
    checker = FactChecker(context)
    assert checker.tools is None
    assert checker.tool_definitions() == []
    configured = FactChecker(context, tools=FakeTools())
    first = configured.tool_definitions()
    first[0]["parameters"]["properties"].clear()
    assert "query" in configured.tool_definitions()[0]["parameters"]["properties"]
    assert configured.prepare() == checker.prepare()


def test_tools_do_not_allow_package_input():
    with pytest.raises(TypeError, match="one VerificationContext"):
        FactChecker(package_data(), tools=FakeTools())


@pytest.mark.parametrize("contract, arguments", [
    (SearchRequest, {"query": " "}),
    (SearchRequest, {"query": "x" * 501}),
    (ReadRequest, {"source_id": "../invented"}),
])
def test_shared_input_contracts_still_reject_malformed_arguments(contract, arguments):
    with pytest.raises(ValidationError):
        contract.model_validate(arguments)
