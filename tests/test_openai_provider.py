import json
from types import SimpleNamespace

import httpx
import pytest
from openai import OpenAI
from pydantic import ValidationError

from history_studio.models.historical_time import HistoricalTime
from history_studio.research.boundaries import ToolCall
from history_studio.research.openai_provider import (
    OpenAIConfiguration, OpenAIResearchProvider, OpenAIWebTools, RunConfiguration,
    create_client, native_tools, strict_schema,
)


def response(output: list, usage: dict | None = None, status="completed") -> dict:
    return dict(id="resp_test", object="response", created_at=1, model="gpt-4.1-mini", status=status,
                output=output, parallel_tool_calls=False, tool_choice="required", tools=[],
                usage=usage or {"input_tokens": 50, "output_tokens": 10, "total_tokens": 60})


def test_actual_sdk_native_function_request_and_result_roundtrip() -> None:
    requests = []
    def handler(request: httpx.Request) -> httpx.Response:
        requests.append(json.loads(request.content))
        return httpx.Response(200, json=response([dict(type="function_call", id="fc1", call_id="call1",
            name="search_web", arguments='{"query":"agent-selected query"}', status="completed")]))
    client = OpenAI(api_key="test-key", http_client=httpx.Client(transport=httpx.MockTransport(handler)), max_retries=0)
    provider = OpenAIResearchProvider(client, OpenAIConfiguration())
    try:
        reply = provider.decide("bounded context", None, 1000)
        assert reply.call.name == "search_web"
        assert reply.call.arguments == {"query": "agent-selected query"}
        provider.decide("new context", (reply.call, '{"sources":[]}'), 1000)
    finally:
        client.close()
    checkpoint = next(t for t in requests[0]["tools"] if t["name"] == "checkpoint_research")
    assert_historical_time_guidance(checkpoint["parameters"])
    assert_no_ref_siblings(checkpoint["parameters"])
    for tool in requests[0]["tools"]:
        assert_no_ref_siblings(tool["parameters"])
    assert requests[0]["store"] is False
    assert requests[0]["parallel_tool_calls"] is False
    assert requests[0]["tool_choice"] == "required"
    assert {tool["name"] for tool in requests[0]["tools"]} == {"search_web", "read_source", "checkpoint_research"}
    assert requests[1]["input"][-1] == {"type": "function_call_output", "call_id": "call1", "output": '{"sources":[]}'}
    assert "previous_response_id" not in requests[1]
    assert "test-key" not in json.dumps(requests)


def test_native_schemas_are_strict_and_exclude_legacy_embedded_sources() -> None:
    def check(value):
        if isinstance(value, list):
            for item in value:
                check(item)
        elif isinstance(value, dict):
            if value.get("type") == "object":
                assert value["additionalProperties"] is False
                assert set(value["required"]) == set(value["properties"])
            assert "default" not in value
            for child in value.values():
                check(child)
    for tool in native_tools():
        assert_no_ref_siblings(tool["parameters"])
        assert tool["strict"]
        check(tool["parameters"])
    fact_schema = native_tools()[-1]["parameters"]["$defs"]["ResearchFactProposal"]["properties"]
    assert "sources" not in fact_schema
    assert "evidence" in fact_schema


def test_hosted_search_is_bounded_and_only_exposes_real_source_urls() -> None:
    requests = []
    def handler(request):
        requests.append(json.loads(request.content))
        return httpx.Response(200, json=response([
            dict(type="web_search_call", id="ws1", status="completed", action={"type": "search",
                "query": "chosen query", "sources": [{"type": "url", "url": "https://example.org/a"}]}),
            dict(type="message", id="msg1", role="assistant", status="completed", content=[
                {"type": "output_text", "text": "Untrusted synthesized summary, NOT evidence", "annotations": [
                    {"type": "url_citation", "url": "https://example.org/a", "title": "Source title", "start_index": 0, "end_index": 10}]}]),
        ]))
    client = OpenAI(api_key="test", http_client=httpx.Client(transport=httpx.MockTransport(handler)))
    tools = OpenAIWebTools(client, OpenAIConfiguration())
    try:
        result = tools.search_web("chosen query")
        assert len(result.sources) == 1
        assert result.text == ""
        assert result.sources[0].title == "Source title"
        assert result.usage.estimated_tool_cost_usd == 0.01
        assert tools.search_reserve_cost("chosen query") >= result.usage.estimated_model_cost_usd + 0.01
    finally:
        client.close()
    assert requests[0]["max_tool_calls"] == 1
    assert requests[0]["include"] == ["web_search_call.action.sources"]
    assert requests[0]["tools"][0]["type"] == "web_search"


def test_missing_key_and_configuration_secret_rejection(monkeypatch) -> None:
    monkeypatch.delenv("OPENAI_API_KEY", raising=False)
    with pytest.raises(ValueError, match="OPENAI_API_KEY"):
        create_client(OpenAIConfiguration())
    monkeypatch.setenv("OPENAI_API_KEY", "test-secret")
    monkeypatch.setenv("OPENAI_BASE_URL", "https://untrusted.example/")
    client = create_client(OpenAIConfiguration())
    try:
        assert str(client.base_url) == "https://api.openai.com/v1/"
        assert client.max_retries == 0
        assert "test-secret" not in RunConfiguration().model_dump_json()
    finally:
        client.close()
    with pytest.raises(ValidationError):
        OpenAIConfiguration(api_key="bad")
    with pytest.raises(ValidationError, match="pricing_model"):
        OpenAIConfiguration(model="different-model")
    alternate = OpenAIConfiguration(model="different-model", pricing_model="different-model",
                                    input_usd_per_million=2, output_usd_per_million=8)
    assert alternate.model == "different-model"


def test_missing_usage_stays_unknown_and_reservation_covers_schema() -> None:
    raw = SimpleNamespace(status="completed", model="model", usage=None, output=[
        SimpleNamespace(type="function_call", call_id="c1", name="search_web", arguments='{"query":"q"}')])
    client = SimpleNamespace(responses=SimpleNamespace(create=lambda **kwargs: raw))
    provider = OpenAIResearchProvider(client, OpenAIConfiguration())
    reply = provider.decide("context", None, 1000)
    assert reply.usage.input_tokens is None
    assert reply.usage.estimated_model_cost_usd is None
    assert provider.reserve_cost("context", None, 1000) > 1000 * 1.6 / 1_000_000


@pytest.mark.parametrize("status, output", [("incomplete", []), ("completed", [])])
def test_malformed_or_incomplete_response_fails_closed(status, output) -> None:
    raw = SimpleNamespace(status=status, output=output)
    client = SimpleNamespace(responses=SimpleNamespace(create=lambda **kwargs: raw))
    with pytest.raises(RuntimeError):
        OpenAIResearchProvider(client, OpenAIConfiguration()).decide("context", None, 1000)


def test_alternate_model_cannot_accidentally_inherit_default_rates() -> None:
    with pytest.raises(ValidationError, match="both rates"):
        OpenAIConfiguration(model="different-model", pricing_model="different-model")


@pytest.mark.parametrize("query, expected_usd", [("x", 0.016444), ("李白", 0.016446)])
def test_search_reservation_counts_one_fixed_block(query: str, expected_usd: float, monkeypatch) -> None:
    tools = OpenAIWebTools(SimpleNamespace(), OpenAIConfiguration())
    monkeypatch.setattr(tools, "_request", lambda query: {"input": query})
    # 14 / 19 UTF-8 request bytes + 4096 framing + one 8000-token block,
    # at $0.40/M input; 1000 max output tokens at $1.60/M; one $0.01 tool fee.
    assert tools.search_reserve_cost(query) == pytest.approx(expected_usd)


def test_search_reservation_preserves_configured_cost_components(monkeypatch) -> None:
    config = OpenAIConfiguration(request_overhead_tokens=8192, search_output_tokens=2000,
        search_input_usd_per_million=2, search_output_usd_per_million=7, search_call_usd=0.03)
    tools = OpenAIWebTools(SimpleNamespace(), config)
    monkeypatch.setattr(tools, "_request", lambda query: {"input": query})
    # 14 bytes + 8192 framing + 8000 search tokens at $2/M, 2000 output at $7/M,
    # and the configured $0.03 fee. No network or usage response is required.
    assert tools.search_reserve_cost("x") == pytest.approx(0.076412)


def assert_historical_time_guidance(parameters: dict) -> None:
    definitions = parameters["$defs"]
    evidence = definitions["EvidenceSelection"]
    assert set(evidence["properties"]) == {"source_id", "span_id"}
    assert "Python extracts canonical text" in evidence["description"]
    assert "accepted evidence on this same fact or a current read" in evidence["properties"]["source_id"]["description"]
    assert "do not transcribe or override excerpts" in evidence["properties"]["span_id"]["description"]
    schema = definitions["HistoricalTime"]
    properties = schema["properties"]
    assert set(schema["required"]) == {"display", "start_year", "end_year", "precision"}
    for guidance in ("use null for unavailable bounds", "end_year requires start_year",
                     "Equal bounds are permitted", "era/textual dates"):
        assert guidance in schema["description"]
    assert "original historical date expression" in properties["display"]["description"]
    assert "0 = 1 BCE" in properties["start_year"]["description"]
    assert "If null, end_year must also be null" in properties["start_year"]["description"]
    assert ">= start_year" in properties["end_year"]["description"]
    assert "YEAR use null or the same year" in properties["end_year"]["description"]
    assert "RANGE/CENTURY: both bounds required" in properties["precision"]["description"]
    assert "APPROXIMATE/UNKNOWN: both null, start alone, or ordered bounds" in properties["precision"]["description"]
    assert "$ref" not in properties["precision"]
    assert properties["precision"]["type"] == "string"
    assert properties["precision"]["enum"] == definitions["TimePrecision"]["enum"]
    for precision in definitions["TimePrecision"]["enum"]:
        assert precision in definitions["TimePrecision"]["description"]
    for field in ("start_year", "end_year"):
        assert {item["type"] for item in properties[field]["anyOf"]} == {"integer", "null"}
    examples = [json.loads(line.strip()) for line in schema["description"].splitlines()
                if line.strip().startswith('{"display"')]
    assert len(examples) == 8
    assert {e["precision"] for e in examples} == set(definitions["TimePrecision"]["enum"])
    for example in examples:
        assert set(example) == set(schema["required"])
        HistoricalTime.model_validate(example)


def test_checkpoint_schema_communicates_valid_historical_time_examples() -> None:
    checkpoint = next(t for t in native_tools() if t["name"] == "checkpoint_research")
    assert_historical_time_guidance(checkpoint["parameters"])
    assert_no_ref_siblings(checkpoint["parameters"])



def assert_no_ref_siblings(value):
    if isinstance(value, dict):
        if "$ref" in value:
            assert set(value) == {"$ref"}, value
        for child in value.values():
            assert_no_ref_siblings(child)
    elif isinstance(value, list):
        for child in value:
            assert_no_ref_siblings(child)


def test_generic_ref_siblings_match_sdk_and_preserve_input():
    from copy import deepcopy
    from openai.lib._pydantic import _ensure_strict_json_schema
    schema = {"type": "object", "$defs": {
        "Choice": {"type": "string", "enum": ["a", "b"], "description": "shared"},
        "Record": {"type": "object", "properties": {"choice": {
            "$ref": "#/$defs/Choice", "description": "local semantics"}}}},
        "properties": {"record": {"$ref": "#/$defs/Record", "description": "record guidance"},
                       "choice": {"anyOf": [{"$ref": "#/$defs/Choice", "description": "union guidance"},
                                            {"type": "null"}]},
                       "plain": {"$ref": "#/$defs/Choice"}}}
    original = deepcopy(schema)
    expected = deepcopy(schema)
    _ensure_strict_json_schema(expected, path=(), root=expected)
    actual = strict_schema(schema)
    assert actual == expected
    assert schema == original
    assert_no_ref_siblings(actual)
    record = actual["properties"]["record"]
    assert record["additionalProperties"] is False
    assert record["required"] == ["choice"]
    assert record["properties"]["choice"]["description"] == "local semantics"
    assert actual["properties"]["plain"] == {"$ref": "#/$defs/Choice"}


@pytest.mark.parametrize("ref", ["https://example.org/schema", "#/$defs/missing", "#/$defs/Cycle"])
def test_ref_expansion_fails_explicitly_for_unresolvable_targets(ref):
    with pytest.raises(ValueError):
        definitions = ({"Cycle": {"$ref": "#/$defs/Cycle", "description": "cycle"}}
                       if ref == "#/$defs/Cycle" else {})
        strict_schema({"$defs": definitions, "$ref": ref, "description": "local"})


def goal_checkpoint_arguments(status):
    goal = dict(gap_id="G1", question="Which date is supported?", status=status,
                completion_criteria=["Compare dated records"], critical=True, fact_ids=[])
    if status in ("COVERED", "RESEARCHED_UNRESOLVED"):
        goal["fact_ids"] = ["F1"]
        goal["coverage_assessment"] = dict(
            criteria=[dict(criterion="Compare dated records", addressed=True)],
            supporting_fact_ids=["F1"],
            unresolved_issues=["Records disagree"] if status == "RESEARCHED_UNRESOLVED" else [],
            rationale="Dated records were compared.")
    return dict(plan=dict(gaps=[goal], coverage_summary="Records under review"),
                facts=[], recommendation="CONTINUE")


@pytest.mark.parametrize("status", ["OPEN", "INVESTIGATING", "COVERED", "RESEARCHED_UNRESOLVED"])
def test_goal_coverage_action_uses_canonical_models_and_roundtrips(status):
    from history_studio.models.research_package import CoverageAssessment, ResearchGoal, ResearchPlan
    from history_studio.research.actions import ResearchSelectionUpdate, ResearchUpdate, action_contracts

    contract = action_contracts()["checkpoint_research"]
    assert contract is ResearchSelectionUpdate
    assert contract.model_fields["plan"].annotation is ResearchPlan
    assert ResearchUpdate.model_fields["plan"].annotation is ResearchPlan
    arguments = goal_checkpoint_arguments(status)
    proposal = contract.model_validate_json(json.dumps(arguments))
    goal = proposal.plan.gaps[0]
    assert type(goal) is ResearchGoal
    assert goal.status == status
    assert goal.completion_criteria == ["Compare dated records"]
    if status in ("OPEN", "INVESTIGATING"):
        assert "coverage_assessment" not in arguments["plan"]["gaps"][0]
        assert goal.coverage_assessment is None
        # Strict native calls explicitly use null for optional fields.
        arguments["plan"]["gaps"][0]["coverage_assessment"] = None
        assert contract.model_validate(arguments) == proposal
    else:
        assert type(goal.coverage_assessment) is CoverageAssessment
        assert goal.coverage_assessment.model_dump() == arguments["plan"]["gaps"][0]["coverage_assessment"]
    assert contract.model_validate_json(proposal.model_dump_json()) == proposal
    assert ResearchUpdate.model_validate(proposal.model_dump()).plan == proposal.plan


@pytest.mark.parametrize("changes", [
    {"criteria": []},
    {"criteria": [{"criterion": "Compare dated records", "addressed": "yes"}]},
    {"supporting_fact_ids": ["../invalid"]},
    {"unresolved_issues": [" "]},
    {"rationale": " "},
    {"unexpected": "field"},
])
def test_goal_coverage_action_rejects_malformed_assessment(changes):
    from history_studio.research.actions import action_contracts

    arguments = goal_checkpoint_arguments("RESEARCHED_UNRESOLVED")
    arguments["plan"]["gaps"][0]["coverage_assessment"].update(changes)
    with pytest.raises(ValidationError):
        action_contracts()["checkpoint_research"].model_validate(arguments)


@pytest.mark.parametrize("status", ["OPEN", "INVESTIGATING", "COVERED", "RESEARCHED_UNRESOLVED"])
def test_goal_coverage_final_native_request_and_mocked_reply(status):
    from history_studio.research.actions import action_contracts

    contract = action_contracts()["checkpoint_research"]
    # Dump includes defaults/nulls required by the strict native schema.
    arguments = contract.model_validate(goal_checkpoint_arguments(status)).model_dump(mode="json")
    requests = []

    def handler(request):
        requests.append(json.loads(request.content))
        return httpx.Response(200, json=response([dict(type="function_call", id="fc1", call_id="call1",
            name="checkpoint_research", arguments=json.dumps(arguments), status="completed")]))

    with OpenAI(api_key="test", max_retries=0,
                http_client=httpx.Client(transport=httpx.MockTransport(handler))) as client:
        reply = OpenAIResearchProvider(client, OpenAIConfiguration()).decide("context", None, 1000)
    assert contract.model_validate(reply.call.arguments).model_dump(mode="json") == arguments
    tool = next(t for t in requests[0]["tools"] if t["name"] == "checkpoint_research")
    assert tool["strict"] is True
    schema = tool["parameters"]
    definitions = schema["$defs"]
    # Follow the actual root-to-goal references to ensure these are reachable fields.
    plan = definitions[schema["properties"]["plan"]["$ref"].split("/")[-1]]
    goal = definitions[plan["properties"]["gaps"]["items"]["$ref"].split("/")[-1]]
    properties = goal["properties"]
    assert properties["completion_criteria"]["type"] == "array"
    assert properties["completion_criteria"]["items"]["type"] == "string"
    assert set(definitions[properties["status"]["$ref"].split("/")[-1]]["enum"]) == {
        "OPEN", "INVESTIGATING", "COVERED", "RESEARCHED_UNRESOLVED"}
    variants = properties["coverage_assessment"]["anyOf"]
    assert {"type": "null"} in variants
    assessment = definitions[next(v["$ref"] for v in variants if "$ref" in v).split("/")[-1]]
    assert set(assessment["properties"]) == {"criteria", "supporting_fact_ids", "unresolved_issues", "rationale"}
    criterion = definitions[assessment["properties"]["criteria"]["items"]["$ref"].split("/")[-1]]
    assert criterion["properties"]["addressed"]["type"] == "boolean"
    assert criterion["properties"]["criterion"]["type"] == "string"

    def check_strict(node):
        if isinstance(node, dict):
            assert "default" not in node
            if node.get("type") == "object":
                assert node["additionalProperties"] is False
                assert set(node["required"]) == set(node["properties"])
            for child in node.values():
                check_strict(child)
        elif isinstance(node, list):
            for child in node:
                check_strict(child)

    check_strict(schema)
    assert_no_ref_siblings(schema)
