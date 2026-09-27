import json

import pytest
from pydantic import ValidationError

from history_studio.models.research import EvidenceReference
from history_studio.models.research_package import ResearchPackage
from history_studio.research.actions import EvidenceSelection
from history_studio.research.agent import ResearchAgent
from history_studio.research.context import INSTRUCTIONS
from history_studio.research.diagnostics import ArtifactValidationError, ValidationDiagnostic
from history_studio.research.spans import make_spans, resolve_selection
from history_studio.research.web_tools import normalize_text, source_reference
from history_studio.storage import ArtifactStore
from test_research_agent import FakeProvider, FakeTools, SOURCE, TEXT, setup_run, settings, calls, update, action


def index(read):
    return {key: (read.source_id, read.source_version) for key in read.spans}


@pytest.mark.parametrize("text", [
    "他於某年出生。後來遷居！事蹟見於記載？" * 70,
    "First sentence. Second sentence! Third sentence? " * 70,
    "無標點" * 1000,
    "word " * 1000,
])
def test_stable_bounded_spans_cover_normalized_source(text):
    read = make_spans(SOURCE.source_id, text)
    assert read == make_spans(SOURCE.source_id, text)
    assert read.text == normalize_text(text)
    cursor = 0
    for key, (start, end) in read.spans.items():
        assert not read.text[cursor:start].strip()
        assert 0 < end - start <= 600
        canonical = resolve_selection(SOURCE.source_id, key, {SOURCE.source_id: read}, index(read), [])
        assert canonical.excerpt == read.text[start:end]
        assert canonical.span_id == key and canonical.source_version == read.source_version
        cursor = end
    assert not read.text[cursor:].strip()
    assert read.source_version != make_spans(SOURCE.source_id, text + "new content").source_version
    assert set(read.spans).isdisjoint(make_spans("different_source", text).spans)
    assert read.source_version != make_spans(SOURCE.source_id, text, True).source_version


@pytest.mark.parametrize("mode, code", [
    ("invented", "span_not_found"), ("unread", "source_not_read"),
    ("mismatched", "span_source_mismatch"), ("stale", "stale_source_span"),
])
def test_selection_identity_rejections(mode, code):
    old = make_spans(SOURCE.source_id, TEXT)
    current = make_spans(SOURCE.source_id, TEXT + " Changed content.")
    other = make_spans("other_source", TEXT)
    selected = next(iter(old.spans))
    reads = {SOURCE.source_id: old}
    seen = index(old) | index(other)
    if mode == "invented":
        selected = "SPAN-invented"
    elif mode == "unread":
        reads = {}
    elif mode == "mismatched":
        selected = next(iter(other.spans))
    else:
        reads[SOURCE.source_id] = current
        seen |= index(current)
    with pytest.raises(ArtifactValidationError) as caught:
        resolve_selection(SOURCE.source_id, selected, reads, seen, ["facts", 0, "evidence", 0, "span_id"])
    assert caught.value.issue.type == code
    assert caught.value.issue.span_selection["span_id"] == selected


@pytest.mark.parametrize("override", [
    {"excerpt": "manufactured"}, {"locator": "fake"}, {"source_version": "fake"},
    {"start": 0, "end": 2},
])
def test_agent_cannot_override_canonical_evidence(override):
    with pytest.raises(ValidationError):
        EvidenceSelection.model_validate({"source_id": SOURCE.source_id, "span_id": "SPAN-test", **override})


def test_existing_artifact_without_span_metadata_remains_readable():
    evidence = EvidenceReference(source_id=SOURCE.source_id, excerpt=TEXT)
    assert evidence.source_version is evidence.span_id is None
    data = update().arguments["facts"][0]
    data["evidence"] = [{"source_id": SOURCE.source_id, "excerpt": TEXT, "locator": None}]
    package = ResearchPackage(project_id="test", topic="subject", facts=[data], sources=[SOURCE],
                              progress={"run_id": "test"})
    legacy = json.loads(package.model_dump_json())
    for e in legacy["facts"][0]["evidence"]:
        e.pop("source_version")
        e.pop("span_id")
    assert ResearchPackage.model_validate(legacy).facts[0].evidence[0].excerpt == TEXT
    with pytest.raises(ValidationError, match="together"):
        EvidenceReference(source_id=SOURCE.source_id, excerpt=TEXT, span_id="SPAN-alone")


def test_changed_read_rejects_old_span_then_corrects_from_retained_observation(tmp_path):
    project, store = setup_run(tmp_path)
    old = make_spans(SOURCE.source_id, TEXT)
    fresh = make_spans(SOURCE.source_id, TEXT + " Changed content.")
    class ChangingTools(FakeTools):
        def read_source(self, source, max_chars):
            result = super().read_source(source, max_chars)
            if len(self.reads) > 1:
                result.text += " Changed content."
            return result
    correct = update()
    correct.arguments["facts"][0]["evidence"][0]["span_id"] = next(iter(fresh.spans))
    provider = FakeProvider(calls()[:2] + [action("read_source", source_id=SOURCE.source_id), update(), correct])
    events = []
    package = ResearchAgent(provider, ChangingTools(), settings(), events.append).run(project, store)
    assert package.progress.status == "COMPLETE"
    evidence = package.facts[0].evidence[0]
    assert evidence.excerpt == fresh.text
    assert evidence.source_version == fresh.source_version
    persisted = store.load_latest("research", ResearchPackage).facts[0].evidence[0]
    assert persisted == evidence
    saved = ArtifactStore(store.project_dir / ".runtime").load_latest("diagnostics", ValidationDiagnostic)
    assert saved.errors[0].type == "stale_source_span"
    assert saved.errors[0].span_selection["selected_source_version"] == old.source_version
    assert any("stale_source_span" in e for e in events)
    retained = json.loads(provider.contexts[4][len(INSTRUCTIONS):])["evidence_context"]["latest_read_source"]
    assert retained["source_version"] == fresh.source_version
    assert retained["spans"][0]["span_id"] == evidence.span_id


def test_observation_budget_bounds_spans_and_marks_truncation(tmp_path):
    project, store = setup_run(tmp_path)
    class LongTools(FakeTools):
        def read_source(self, source, max_chars):
            result = super().read_source(source, max_chars)
            result.text = "A sentence with evidence. " * 300
            return result
    provider = FakeProvider(calls())
    original = provider.decide
    selected = {}
    def choose_visible(context, observation, max_output_tokens):
        if provider.calls == 2:
            page = json.loads(observation[1])
            assert page["truncated"] is True
            assert "text" not in page  # Avoid a duplicate page alongside span text.
            assert len(observation[1]) + len(json.dumps(observation[0].model_dump(), ensure_ascii=False)) <= 3000
            selected.update(page)
            reply = original(context, observation, max_output_tokens)
            reply.call.arguments["facts"][0]["evidence"] = [{"source_id": page["source_id"],
                "span_id": page["spans"][-1]["span_id"]}]
            return reply
        return original(context, observation, max_output_tokens)
    provider.decide = choose_visible
    package = ResearchAgent(provider, LongTools(), settings(max_observation_chars=3000)).run(project, store)
    assert package.progress.status == "COMPLETE"
    evidence = package.facts[0].evidence[0]
    assert evidence.excerpt == selected["spans"][-1]["text"]
    assert evidence.source_version == selected["source_version"]
    assert all(len(c) <= 21000 for c in provider.contexts)
    assert all(e.source_id == SOURCE.source_id for e in package.facts[0].evidence)
