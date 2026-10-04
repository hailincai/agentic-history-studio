"""Durable Phase 3 closure across real orchestration, with external boundaries fake."""
from datetime import datetime, timezone
import json

import pytest

from history_studio.cli import main, read_project
from history_studio.models import ArtifactReference, VerificationPackage
from history_studio.models.research_package import ResearchPackage, ResearchRunStatus
from history_studio.research.agent import ResearchAgent
from history_studio.research.spans import make_spans
from history_studio.storage.artifact_store import ArtifactStore, write_json
from history_studio.verification import FactCheckingRunner
from history_studio.workflow import ApprovalRecord, ProjectState as S, RuntimeState
from history_studio.workflow.fact_checking import FactCheckingWorkflow
from history_studio.workflow.fact_review import load_fact_review, apply_fact_review
from test_research_agent import setup_run, FakeProvider, FakeTools, settings, action, update, SOURCE, TEXT
from test_fact_checker_dispatch import native
from test_fact_checker_investigation import SequenceProvider, decision
from test_verification_evidence import TextTools
from test_verification_submission import payload, terminal


def research_to_gate(tmp_path):
    project, store = setup_run(tmp_path)
    checkpoint = update()
    second = dict(checkpoint.arguments["facts"][0], fact_id="RF-2",
        claim="A different source suggests 702.",
        historical_time=dict(display="702", start_year=702, precision="YEAR"))
    checkpoint.arguments["facts"].append(second)
    goal = checkpoint.arguments["plan"]["gaps"][0]
    goal["fact_ids"].append("RF-2")
    goal["coverage_assessment"]["supporting_fact_ids"].append("RF-2")
    research = ResearchAgent(FakeProvider([
        action("search_web", query="Candidate birth chronology"),
        action("read_source", source_id=SOURCE.source_id), checkpoint]),
        FakeTools(), settings(min_facts=2)).run(project, store)
    assert research.progress.status == ResearchRunStatus.COMPLETE
    assert [fact.fact_id for fact in research.facts] == ["RF-1", "RF-2"]
    # ResearchAgent itself published and bound the accepted checkpoint.
    project, before = read_project(store.project_dir)
    reference = ArtifactReference(project_id="test", artifact_type="research", version=2)
    assert before.current_state == S.RESEARCH_COMPLETE
    assert before.artifacts.research == reference
    assert before.artifacts.verification is None and before.artifacts.approved_verification is None
    assert store.load("research", reference.version, ResearchPackage) == research

    contexts, providers, tools = [], [], []
    canonical = make_spans("SRC-new", TEXT)
    selection = dict(source_id="SRC-new", span_id=next(iter(canonical.spans)))
    def provider_factory(context):
        contexts.append(context)
        submission = payload("VERIFIED", source_id="SRC-new")
        submission["verification_evidence"] = [selection]
        provider = SequenceProvider([
            decision(native("search_web", json.dumps(dict(query="Independent claim-specific chronology")))),
            decision(native("read_source", json.dumps(dict(source_id="SRC-new")))),
            decision(terminal(submission))])
        providers.append(provider)
        return provider
    def tools_factory(context):
        tool = TextTools(TEXT)
        tools.append(tool)
        return tool
    runner = FactCheckingRunner(provider_factory=provider_factory, tools_factory=tools_factory)
    outcome = FactCheckingWorkflow(runner).run(project, store)
    assert outcome.state.current_state == S.WAITING_FACT_APPROVAL
    assert outcome.stage.completed_count == 2 and outcome.stage.verification_version == 2
    assert [context.target_fact.fact_id for context in contexts] == ["RF-1", "RF-2"]
    assert all(context.research_input_ref == reference for context in contexts)
    assert all(len(provider.requests) == 3 for provider in providers)
    assert all([call[0] for call in tool.calls] == ["search", "read"] for tool in tools)
    for version in (1, 2):
        package = store.load("verification", version, VerificationPackage)
        assert package.completed_fact_ids == ["RF-1", "RF-2"][:version]
        assert package.pending_fact_ids == ["RF-1", "RF-2"][version:]
        assert package.is_complete == (version == 2)
        assert package.research_input_ref == reference
        assert len(package.results) == version
        for result in package.results:
            assert result.verification_evidence[0].excerpt == canonical.text
    return store.project_dir, reference


def human_decision(state):
    return ApprovalRecord(project_id="test", stage="facts", artifact_type="verification",
        artifact_version=state.artifacts.verification.version, decision="APPROVED",
        decided_by="Integration human reviewer", decision_source="human",
        decided_at=datetime(2026, 10, 4, 12, tzinfo=timezone.utc))


@pytest.mark.parametrize("version_drift", [False, True])
def test_durable_research_verification_review_approval_after_restart(tmp_path, capsys, version_drift):
    path, research_ref = research_to_gate(tmp_path)
    # Fresh store/project/state/services simulate restart; no in-memory outcome is trusted.
    store = ArtifactStore(path)
    project, waiting = read_project(path)
    verification_ref = ArtifactReference(project_id=project.project_id, artifact_type="verification", version=2)
    assert waiting.artifacts.research == research_ref
    assert waiting.artifacts.verification == verification_ref
    assert waiting.artifacts.approved_verification is None
    target = store.load("verification", 2, VerificationPackage)
    if version_drift:
        assert store.save("verification", target) == 3
    assert load_fact_review(waiting, ArtifactStore(path)) == target
    assert target.is_complete and target.completed_fact_ids == ["RF-1", "RF-2"]
    args = ["--projects-dir", str(tmp_path), "review", "test", "facts"]
    capsys.readouterr()
    assert main(args) == 0
    output = capsys.readouterr().out
    assert "test/verification:v2" in output and "verification:v3" not in output
    assert "RF-1" in output and "RF-2" in output and "VERIFIED" in output
    assert read_project(path)[1] == waiting
    attestation = human_decision(waiting)
    if version_drift:
        decision_path = tmp_path / "human-decision.json"
        write_json(decision_path, attestation)
        assert main(args + ["--decision-file", str(decision_path)]) == 0
        assert "FACTS_APPROVED" in capsys.readouterr().out
    else:
        apply_fact_review(path, attestation)
    _, approved = read_project(path)
    assert approved.current_state == S.FACTS_APPROVED
    assert approved.last_successful_state == S.FACTS_APPROVED
    assert approved.artifacts.research == research_ref
    assert approved.artifacts.verification == verification_ref
    assert approved.artifacts.approved_verification == verification_ref
    assert ArtifactStore(path).load("approvals", 1, ApprovalRecord) == attestation
    assert ArtifactStore(path).list_versions("verification") == ([1, 2, 3] if version_drift else [1, 2])
    assert RuntimeState.model_validate_json((path / ".runtime/state.json").read_text(encoding="utf-8")) == approved
    assert not store.list_versions("story")


def test_persisted_corrupted_research_lineage_fails_closed_without_latest_fallback(tmp_path, capsys):
    path, _ = research_to_gate(tmp_path)
    store = ArtifactStore(path)
    _, waiting = read_project(path)
    valid = store.load("verification", 2, VerificationPackage)
    data = valid.model_dump(mode="json")
    data["research_input_ref"]["version"] = 1
    for result in data["results"]:
        result["research_input_ref"]["version"] = 1
    # Structurally valid package; deliberately corrupt the persisted workflow lineage.
    corrupted = VerificationPackage.model_validate(data)
    assert store.save("verification", corrupted) == 3
    assert store.save("verification", valid) == 4  # A valid latest artifact cannot repair the bound target.
    state_data = waiting.model_dump(mode="json")
    state_data["artifacts"]["verification"]["version"] = 3
    write_json(path / ".runtime/state.json", RuntimeState.model_validate(state_data), replace=True)
    _, restarted = read_project(path)
    before = (path / ".runtime/state.json").read_bytes()
    with pytest.raises(ValueError, match="research snapshot"):
        load_fact_review(restarted, ArtifactStore(path))
    with pytest.raises(ValueError, match="research snapshot"):
        apply_fact_review(path, human_decision(restarted))
    capsys.readouterr()
    assert main(["--projects-dir", str(tmp_path), "review", "test", "facts"]) == 1
    assert "research snapshot" in capsys.readouterr().err
    assert (path / ".runtime/state.json").read_bytes() == before
    after = read_project(path)[1]
    assert after.current_state == S.WAITING_FACT_APPROVAL
    assert after.artifacts.verification.version == 3
    assert after.artifacts.approved_verification is None
    assert not store.list_versions("approvals")
