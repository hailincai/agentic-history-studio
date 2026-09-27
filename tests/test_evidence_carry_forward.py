from copy import deepcopy

import pytest

from history_studio.models.research_package import ResearchPackage
from history_studio.research.agent import ResearchAgent
from history_studio.research.diagnostics import ValidationDiagnostic
from history_studio.research.spans import make_spans
from history_studio.storage import ArtifactStore
from test_research_agent import FakeProvider, FakeTools, SOURCE, TEXT, setup_run, settings, calls, update, action, ledger


def saved_evidence(store, version=2):
    return store.load("research", version, ResearchPackage).facts[0].evidence[0]


def test_unchanged_evidence_survives_next_iteration_without_read(tmp_path):
    project, store = setup_run(tmp_path)
    tools = FakeTools()
    provider = FakeProvider(calls(complete=False) + [update()])
    package = ResearchAgent(provider, tools, settings()).run(project, store)
    assert package.progress.status == "COMPLETE"
    assert package.progress.iterations == 2
    assert package.facts[0].evidence[0].model_dump() == saved_evidence(store).model_dump()
    assert tools.reads == [SOURCE.source_id]
    assert ArtifactStore(store.project_dir / ".runtime").list_versions("diagnostics") == []


@pytest.mark.parametrize("change, code", [
    ({"span_id": "SPAN-modified"}, "source_not_read"),
    ({"source_id": "SRC-other"}, "source_not_read"),
    ({"excerpt": "manufactured"}, "extra_forbidden"),
    ({"source_version": "VER-modified"}, "extra_forbidden"),
])
def test_old_evidence_overrides_rejected_then_carry_forward_corrects(tmp_path, change, code):
    project, store = setup_run(tmp_path)
    bad = update()
    bad.arguments["facts"][0]["evidence"][0].update(change)
    tools = FakeTools()
    provider = FakeProvider(calls(complete=False) + [bad, update()])
    package = ResearchAgent(provider, tools, settings()).run(project, store)
    assert package.progress.status == "COMPLETE"
    diagnostic = ArtifactStore(store.project_dir / ".runtime").load_latest("diagnostics", ValidationDiagnostic)
    assert diagnostic.errors[0].type == code
    assert diagnostic.iteration == 2 and diagnostic.turn == 1
    assert package.facts[0].evidence[0] == saved_evidence(store)
    assert tools.reads == [SOURCE.source_id]


def test_new_fact_cannot_borrow_old_trust_without_read(tmp_path):
    project, store = setup_run(tmp_path)
    new = update()
    new.arguments["facts"][0].update(fact_id="RF-new", claim="A new candidate claim")
    provider = FakeProvider(calls(complete=False) + [new, update()])
    package = ResearchAgent(provider, FakeTools(), settings()).run(project, store)
    assert package.progress.status == "COMPLETE"
    assert [f.fact_id for f in package.facts] == ["RF-1"]
    diagnostic = ArtifactStore(store.project_dir / ".runtime").load_latest("diagnostics", ValidationDiagnostic)
    assert diagnostic.errors[0].type == "source_not_read"


def test_accepted_evidence_and_new_read_can_mix_after_correction(tmp_path):
    project, store = setup_run(tmp_path)
    changed_text = TEXT + " Another recorded event."
    fresh = make_spans(SOURCE.source_id, changed_text)
    class ChangingTools(FakeTools):
        def read_source(self, source, max_chars):
            result = super().read_source(source, max_chars)
            if len(self.reads) == 2:
                result.text = changed_text
            return result
    mixed = update()
    mixed.arguments["facts"][0]["evidence"].append({"source_id": SOURCE.source_id,
        "span_id": next(iter(fresh.spans))})
    bad = deepcopy(mixed)
    bad.arguments["facts"][0]["evidence"][1]["span_id"] = "SPAN-invented"
    provider = FakeProvider(calls(complete=False) + [action("read_source", source_id=SOURCE.source_id), bad, mixed])
    tools = ChangingTools()
    package = ResearchAgent(provider, tools, settings()).run(project, store)
    assert package.progress.status == "COMPLETE"
    prior, new = package.facts[0].evidence
    assert prior.model_dump() == saved_evidence(store).model_dump()
    assert new.excerpt == changed_text
    assert new.source_version == fresh.source_version != prior.source_version
    assert new.span_id == next(iter(fresh.spans))
    diagnostic = ArtifactStore(store.project_dir / ".runtime").load_latest("diagnostics", ValidationDiagnostic)
    assert diagnostic.errors[0].type == "span_not_found"
    assert diagnostic.errors[0].location == ["facts", 0, "evidence", 1, "span_id"]
    assert len(tools.reads) == 2
    assert store.load_latest("research", ResearchPackage).facts[0].evidence == [prior, new]


def test_resume_uses_persisted_evidence_without_reread_or_budget_reset(tmp_path):
    project, store = setup_run(tmp_path)
    ResearchAgent(FakeProvider(calls(complete=False) + [RuntimeError("offline failure")]),
                  FakeTools(), settings()).run(project, store)
    before = ledger(store)
    original = saved_evidence(store)
    # A fresh agent/provider simulates another process. No in-memory read registry survives.
    tools = FakeTools()
    package = ResearchAgent(FakeProvider([update()]), tools, settings()).run(project, store)
    assert package.progress.status == "COMPLETE"
    assert tools.reads == [] and tools.queries == []
    assert package.facts[0].evidence == [original]
    assert ledger(store).run_id == before.run_id
    assert ledger(store).committed_budget_usd > before.committed_budget_usd
    assert ledger(store).source_reads == before.source_reads
