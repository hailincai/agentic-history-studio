import json
import unicodedata

import pytest

from history_studio.research.evidence_diagnostics import evidence_mismatch
from history_studio.research.web_tools import normalize_text, read_public_source
from history_studio.research.agent import ResearchAgent
from history_studio.research.context import INSTRUCTIONS
from history_studio.research.diagnostics import ValidationDiagnostic
from history_studio.storage import ArtifactStore
from test_research_agent import FakeProvider, FakeTools, SOURCE, setup_run, settings, calls, update


@pytest.mark.parametrize("page, quote, accepted, category", [
    ("他於某年出生，後來遷居。", "他於某年出生，", True, "exact"),
    ("他於某年出生，後來遷居。", "他於某年出生,", False, "punctuation_difference"),
    ("他於某年 出生，後來遷居。", "他於某年\\n  出生，", True, "whitespace_only"),
    ("他於某年出生，後來遷居。", "他於某年 出生，", False, "unknown"),
    ("姓名為Café，後來遷居。", "姓名為Cafe\u0301，", False, "unicode_normalization_difference"),
    ("他於某年出生，後來遷居。", "他於某年出生，遷居。", False, "unknown"),
    ("他於某年出生，後來遷居。", "他於某年出生，隨即遷居。", False, "unknown"),
    ("他於某年出生，後來遷居。", "He was born that year.", False, "unknown"),
])
def test_chinese_checkpoint_serialization_and_mismatch(tmp_path, page, quote, accepted, category):
    quote = quote.replace("\\n", "\n")
    from history_studio.research.actions import ResearchUpdate
    from history_studio.models.research_package import ResearchPackage
    from history_studio.research.diagnostics import ArtifactValidationError
    project, store = setup_run(tmp_path)
    data = update().arguments
    data["facts"][0]["evidence"] = [{"source_id": SOURCE.source_id, "excerpt": quote}]
    data = json.loads(json.dumps(data, ensure_ascii=True))
    assert data["facts"][0]["evidence"][0]["excerpt"] == quote
    canonical = ResearchUpdate.model_validate(data)
    package = ResearchPackage(project_id=project.project_id, topic=project.topic, sources=[SOURCE],
                              progress={"run_id": "test"})
    agent = ResearchAgent(FakeProvider([]), FakeTools(), settings())
    mismatch = evidence_mismatch(SOURCE.source_id, quote, {SOURCE.source_id: normalize_text(page)},
                                 raw_text=page, truncated=False)
    assert mismatch.category == category
    assert mismatch.occurs_after_normalization is accepted
    if accepted:
        assert agent._apply_update(package, canonical, {SOURCE.source_id: normalize_text(page)}).facts
    else:
        with pytest.raises(ArtifactValidationError) as caught:
            agent._apply_update(package, canonical, {SOURCE.source_id: normalize_text(page)},
                                {SOURCE.source_id: (page, False)})
        assert caught.value.issue.evidence_mismatch == mismatch


def test_html_entities_truncation_and_wrong_source(monkeypatch):
    monkeypatch.setattr("history_studio.research.web_tools._fetch", lambda url:
        (url, "<p>他說：&quot;甲&amp;乙&quot;。然後離開。</p>"))
    result = read_public_source(SOURCE, 14)
    assert result.text == '他說："甲&乙"。然後離開。'[:14]
    full = '他說："甲&乙"。然後離開。'
    result = read_public_source(SOURCE, 10)
    assert result.truncated
    passages = {SOURCE.source_id: normalize_text(result.text)}
    mismatch = evidence_mismatch(SOURCE.source_id, full, passages, raw_text=result.text, truncated=True)
    assert not mismatch.occurs_after_normalization
    assert mismatch.source_truncated and mismatch.boundary_prefix_match
    assert mismatch.category == "unknown"  # Missing tail cannot be verified from retained text.
    assert evidence_mismatch("other", full, passages).category == "source_id_mismatch"
    assert evidence_mismatch(SOURCE.source_id, '他說：&quot;甲&amp;乙&quot;。', passages).category == "unknown"


def test_mismatch_details_remain_bounded_and_redacted():
    page = "prefix " + "甲乙丙丁" * 2000 + " sk-secret123 Authorization: Bearer private-secret"
    quote = "甲乙丙丁。 sk-secret123 Authorization: Bearer private-secret"
    mismatch = evidence_mismatch(SOURCE.source_id, quote, {SOURCE.source_id: page})
    output = mismatch.model_dump_json()
    assert "sk-secret123" not in output and "private-secret" not in output
    assert len(mismatch.proposed_preview) <= 1000
    assert mismatch.nearest_span_end - mismatch.nearest_span_start <= 240
    assert len(output) < 2500 and page not in output


def test_diagnostic_feedback_keeps_read_context_for_correction(tmp_path):
    project, store = setup_run(tmp_path)
    provider = FakeProvider(calls()[:2] + [update(quote="A paraphrase"), update()])
    package = ResearchAgent(provider, FakeTools(), settings()).run(project, store)
    assert package.progress.status == "COMPLETE"
    feedback = json.loads(provider.observations[3][1])
    mismatch = feedback["errors"][0]["span_selection"]
    context = json.loads(provider.contexts[3][len(INSTRUCTIONS):])
    assert mismatch["source_was_read"] is True
    assert feedback["errors"][0]["type"] == "span_not_found"
    assert context["evidence_context"]["latest_read_source"]["source_id"] == SOURCE.source_id
    assert context["evidence_context"]["latest_read_source"]["spans"][0]["text"]


@pytest.mark.parametrize("escape_unicode", [True, False])
def test_mocked_sdk_checkpoint_arguments_reach_validator_unchanged(tmp_path, monkeypatch, escape_unicode):
    import httpx
    from openai import OpenAI
    from history_studio.research.openai_provider import OpenAIResearchProvider, OpenAIConfiguration
    from test_openai_provider import response

    page = "他於某年出生，後來遷居。"
    from history_studio.research.spans import make_spans
    exact = page
    proposed = ["SPAN-invented", next(iter(make_spans(SOURCE.source_id, page).spans))]
    requests = []
    validator_inputs = []
    project, store = setup_run(tmp_path)
    # Retrieval is lexical: align the pre-plan scope with this Chinese transport fixture.
    project.research_scope = "出生與遷居"

    class PageTools(FakeTools):
        def read_source(self, source, max_chars):
            result = super().read_source(source, max_chars)
            result.text = page
            return result

    def handler(request):
        requests.append(json.loads(request.content))
        arguments = update().arguments
        arguments["facts"][0]["evidence"] = [{"source_id": SOURCE.source_id,
                                               "span_id": proposed[len(requests) - 1]}]
        return httpx.Response(200, json=response([dict(type="function_call", id="fc",
            call_id="checkpoint", name="checkpoint_research", status="completed",
            arguments=json.dumps(arguments, ensure_ascii=escape_unicode))]))

    original = ResearchAgent._apply_update
    def inspect_validation(self, package, update, passages, read_observations=None):
        validator_inputs.append(update.facts[0].evidence[0].excerpt)
        assert passages[SOURCE.source_id] == normalize_text(page)
        assert read_observations[SOURCE.source_id][0] == page
        return original(self, package, update, passages, read_observations)
    monkeypatch.setattr(ResearchAgent, "_apply_update", inspect_validation)

    with OpenAI(api_key="offline-test-key", max_retries=0,
                http_client=httpx.Client(transport=httpx.MockTransport(handler))) as client:
        sdk = OpenAIResearchProvider(client, OpenAIConfiguration())
        initial = FakeProvider(calls()[:2])
        class RoutedProvider:
            def reserve_cost(self, context, observation, max_output_tokens):
                return 0.001
            def decide(self, context, observation, max_output_tokens):
                if initial.calls < 2:
                    return initial.decide(context, observation, max_output_tokens)
                return sdk.decide(context, observation, max_output_tokens)
        tools = PageTools()
        package = ResearchAgent(RoutedProvider(), tools, settings()).run(project, store)

    assert package.progress.status == "COMPLETE"
    assert validator_inputs == [page]
    assert package.facts[0].evidence[0].excerpt == exact
    assert tools.reads == [SOURCE.source_id]
    assert json.loads(requests[0]["input"][-1]["output"])["spans"][0]["text"] == page
    feedback = json.loads(requests[1]["input"][-1]["output"])
    mismatch = feedback["errors"][0]["span_selection"]
    assert feedback["errors"][0]["type"] == "span_not_found"
    assert mismatch["span_id"] == proposed[0]
    schema = next(t for t in requests[1]["tools"] if t["name"] == "checkpoint_research")["parameters"]
    assert set(schema["$defs"]["EvidenceSelection"]["properties"]) == {"source_id", "span_id"}
    assert "EvidenceReference" not in schema["$defs"]
    prior_call = json.loads(requests[1]["input"][-2]["arguments"])
    assert prior_call["facts"][0]["evidence"][0]["span_id"] == proposed[0]
    context = json.loads(requests[1]["input"][0]["content"][len(INSTRUCTIONS):])
    assert context["evidence_context"]["latest_read_source"]["spans"][0]["text"] == page
