"""Exact human review/decision boundaries, entirely offline."""
from datetime import datetime, timezone
import json

import pytest
from pydantic import ValidationError

from history_studio.cli import main
from history_studio.models import ArtifactReference, VerificationPackage
from history_studio.storage.artifact_store import write_json
from history_studio.workflow import ApprovalRecord, RuntimeState, ProjectState as S, ProjectStateMachine, WorkflowArtifactBindings
from history_studio.workflow.fact_review import load_fact_review, fact_review_lines, apply_fact_review
from history_studio.workflow.fact_checking import FactCheckingWorkflow
from test_fact_checking_workflow import prepare, read_state
from test_fact_checking_runner import Dependencies
from test_verification_package import result_for


def ready(tmp_path):
    project, store, research = prepare(tmp_path)
    outcome = FactCheckingWorkflow(Dependencies().runner()).run(project, store)
    return project, store, research, outcome.state, outcome.stage.package


def decision(state, verdict="APPROVED", **changes):
    return ApprovalRecord(**(dict(project_id="test", stage="facts", artifact_type="verification",
        artifact_version=state.artifacts.verification.version, decision=verdict,
        decided_by="Human reviewer", decision_source="human", decided_at=datetime.now(timezone.utc)) | changes))


def test_exact_bound_review_and_approval_ignore_newer_versions(tmp_path, monkeypatch):
    _, store, _, state, package = ready(tmp_path)
    assert store.save("verification", package) == 4
    def forbidden(*args, **kwargs):
        raise AssertionError("No semantic latest discovery")
    monkeypatch.setattr(store, "load_latest", forbidden)
    versions = store.list_versions
    monkeypatch.setattr(store, "list_versions", lambda kind: forbidden() if kind == "verification" else versions(kind))
    assert load_fact_review(state, store) == package
    machine = ProjectStateMachine(state)
    machine.apply_human_decision(decision(state), store)
    assert machine.state.current_state == S.FACTS_APPROVED
    assert machine.state.artifacts.approved_verification == state.artifacts.verification
    assert machine.state.artifacts.verification.version == 3
    assert versions("verification") == [1, 2, 3, 4]
    assert versions("approvals") == [1]


def test_atomic_publication_reload_and_failure_recovery_preserve_approval(tmp_path, monkeypatch):
    import history_studio.workflow.fact_review as module
    _, store, _, state, _ = ready(tmp_path)
    write = module.write_json
    publications = []
    def observed(path, value, **kwargs):
        publications.append(value)
        assert value.current_state == S.FACTS_APPROVED
        assert value.artifacts.approved_verification == value.artifacts.verification
        return write(path, value, **kwargs)
    monkeypatch.setattr(module, "write_json", observed)
    approved = apply_fact_review(store.project_dir, decision(state))
    assert len(publications) == 1 and read_state(store) == approved
    store.save("verification", store.load("verification", 3, VerificationPackage))
    assert read_state(store).artifacts.approved_verification.version == 3
    machine = ProjectStateMachine(read_state(store))
    machine.transition(S.STORY_GENERATING)
    machine.fail("interrupted")
    machine = ProjectStateMachine(RuntimeState.model_validate_json(machine.state.model_dump_json()))
    machine.recover()
    assert machine.state.artifacts.approved_verification == state.artifacts.verification


@pytest.mark.parametrize("verdict", ["REJECTED", "REVISION_REQUESTED"])
def test_rejection_preserves_target_and_existing_stage_transition(tmp_path, verdict):
    _, store, _, state, package = ready(tmp_path)
    rejected = apply_fact_review(store.project_dir, decision(state, verdict))
    assert rejected.current_state == S.FACT_CHECKING
    assert rejected.artifacts.verification == state.artifacts.verification
    assert rejected.artifacts.approved_verification is None
    assert store.load("verification", 3, VerificationPackage) == package
    assert store.list_versions("verification") == [1, 2, 3]
    assert store.load("approvals", 1, ApprovalRecord).decision == verdict


@pytest.mark.parametrize("failure", ["state", "missing_binding", "foreign", "wrong_type", "missing_artifact",
    "malformed", "incomplete", "research_ref", "membership", "claim", "foreign_binding"])
def test_invalid_review_target_fails_closed_without_approval(tmp_path, failure):
    _, store, research, state, package = ready(tmp_path)
    data = state.model_dump(mode="json")
    package_data = package.model_dump(mode="json")
    if failure == "state":
        data.update(current_state="FACT_CHECKING", last_successful_state="RESEARCH_COMPLETE")
    elif failure == "missing_binding":
        data["artifacts"]["verification"] = None
    elif failure == "foreign":
        for kind in ("research", "verification"):
            data["artifacts"][kind]["project_id"] = "foreign"
    elif failure == "wrong_type":
        data["artifacts"]["verification"]["artifact_type"] = "research"
    elif failure == "missing_artifact":
        data["artifacts"]["verification"]["version"] = 99
    elif failure == "foreign_binding":
        data["artifacts"]["story"] = dict(project_id="foreign", artifact_type="story", version=1)
    elif failure == "malformed":
        package_data = {"provider_request": "must-not-display"}
    elif failure == "incomplete":
        package_data["results"].pop()
    elif failure == "research_ref":
        package_data["research_input_ref"]["version"] = 3
        for result in package_data["results"]:
            result["research_input_ref"]["version"] = 3
    elif failure == "membership":
        package_data["research_facts"].pop()
        package_data["results"].pop()
    elif failure == "claim":
        package_data["research_facts"][0]["claim_snapshot"] = "Changed claim"
        package_data["results"][0]["claim_snapshot"] = "Changed claim"
    if failure in {"malformed", "incomplete", "research_ref", "membership", "claim"}:
        (store.project_dir / "verification/verification_v3.json").write_text(json.dumps(package_data), encoding="utf-8")
    before = (store.project_dir / ".runtime/state.json").read_bytes()
    with pytest.raises((ValueError, FileNotFoundError)):
        invalid = RuntimeState.model_validate(data)
        ProjectStateMachine(invalid).apply_fact_review_decision(decision(state), store)
    assert (store.project_dir / ".runtime/state.json").read_bytes() == before
    assert not store.list_versions("approvals")


@pytest.mark.parametrize("changes", [{"artifact_version": 4}, {"project_id": "other"}, {"artifact_type": "facts"},
    {"stage": "story", "artifact_type": "story"}, {"decision_source": "agent"}])
def test_invalid_human_decision_cannot_change_state(tmp_path, changes):
    _, store, _, state, _ = ready(tmp_path)
    before = read_state(store)
    with pytest.raises(ValueError):
        apply_fact_review(store.project_dir, decision(state, **changes))
    assert read_state(store) == before and not store.list_versions("approvals")


def test_review_content_is_accepted_fields_only_and_bounded(tmp_path):
    _, store, _, state, package = ready(tmp_path)
    data = package.model_dump(mode="json")
    from test_verification_models import evidence_data
    result = result_for(package, 0, "DISPUTED", contradiction_evidence=[evidence_data("A conflicting record gives another date.")])
    data["results"][0] = result.model_dump(mode="json")
    data["results"][0]["rationale"] = "A" * 2000
    data["results"][0]["unresolved_issues"] = ["Uncertain historical date"]
    package = VerificationPackage.model_validate(data)
    output = "\n".join(fact_review_lines(state, package))
    for fact in package.research_facts:
        assert fact.research_fact_id in output and fact.claim_snapshot in output
    for term in ("DISPUTED", "verification_evidence", "contradiction_evidence", "Uncertain historical date",
                 "rationale", "independence_note", "truncated"):
        assert term in output
    assert result.verification_evidence[0].excerpt in output
    assert result.contradiction_evidence[0].excerpt in output
    assert "A" * 1001 not in output
    for excluded in ("ToolObservation", "provider_request", "provider_response", "transcript", "read_ledger"):
        assert excluded not in output


def test_cli_review_and_external_decision_file_use_exact_target(tmp_path, capsys):
    _, store, _, state, package = ready(tmp_path)
    store.save("verification", package)
    args = ["--projects-dir", str(tmp_path), "review", "test", "facts"]
    assert main(args) == 0
    output = capsys.readouterr().out
    assert "verification:v3" in output and "verification:v4" not in output
    assert read_state(store) == state and not store.list_versions("approvals")
    path = tmp_path / "human-decision.json"
    write_json(path, decision(state))
    assert main(args + ["--decision-file", str(path)]) == 0
    assert "FACTS_APPROVED" in capsys.readouterr().out
    assert read_state(store).artifacts.approved_verification == state.artifacts.verification
    assert store.list_versions("verification") == [1, 2, 3, 4]


def test_legacy_states_readable_but_fact_review_cannot_infer_missing_identity(tmp_path, capsys):
    _, store, _, state, _ = ready(tmp_path)
    legacy = RuntimeState(current_state=S.WAITING_FACT_APPROVAL, last_successful_state=S.WAITING_FACT_APPROVAL,
        research_input_ref=state.artifacts.research)
    write_json(store.project_dir / ".runtime/state.json", legacy, replace=True)
    assert main(["--projects-dir", str(tmp_path), "review", "test", "facts"]) == 1
    assert "binding is missing" in capsys.readouterr().err
    assert read_state(store).artifacts.verification is None
    with pytest.raises(ValueError, match="binding is missing"):
        ProjectStateMachine(legacy).apply_human_decision(decision(state), store)
    assert RuntimeState(current_state=S.FACTS_APPROVED, last_successful_state=S.FACTS_APPROVED).artifacts.approved_verification is None


def test_state_write_failure_leaves_waiting_state_without_approved_binding(tmp_path, monkeypatch):
    _, store, _, state, _ = ready(tmp_path)
    def failed(*args, **kwargs):
        raise OSError("state publication failed")
    monkeypatch.setattr("history_studio.workflow.fact_review.write_json", failed)
    with pytest.raises(OSError):
        apply_fact_review(store.project_dir, decision(state))
    assert read_state(store) == state and read_state(store).artifacts.approved_verification is None
    assert store.list_versions("approvals") == [1]  # Audit/state are separate files; fail closed.


def test_duplicate_exact_human_decision_rejected_without_new_artifact(tmp_path):
    _, store, _, state, _ = ready(tmp_path)
    record = decision(state)
    ProjectStateMachine(state).apply_fact_review_decision(record, store)
    with pytest.raises(ValueError, match="already has a decision"):
        ProjectStateMachine(state).apply_fact_review_decision(record, store)
    assert store.list_versions("verification") == [1, 2, 3]
