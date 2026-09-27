from datetime import datetime, timezone
from pathlib import Path

import pytest
from pydantic import ValidationError

from history_studio.models import VerifiedFact, StoryPlan, Script, Storyboard
from history_studio.storage import ArtifactStore
from history_studio.workflow import (
    ApprovalRecord, InvalidTransitionError, ProjectState as S, ProjectStateMachine, RuntimeState,
)


def record(stage: str = "facts", version: int = 1, decision: str = "APPROVED") -> ApprovalRecord:
    return ApprovalRecord(project_id="li_bai", stage=stage, artifact_type=stage,
                          artifact_version=version, decision=decision, decided_by="Human reviewer",
                          decision_source="human", decided_at=datetime.now(timezone.utc))


def artifact() -> VerifiedFact:
    return VerifiedFact(fact_id="f1", claim="claim", status="VERIFIED", confidence=1,
                        reasoning="checked")


def machine_at(state: S) -> ProjectStateMachine:
    return ProjectStateMachine(RuntimeState(current_state=state, last_successful_state=state))


def test_full_workflow_with_persisted_human_gates(tmp_path: Path) -> None:
    store = ArtifactStore(tmp_path / "li_bai")
    machine = ProjectStateMachine()
    for state in (S.RESEARCHING, S.RESEARCH_COMPLETE, S.FACT_CHECKING, S.WAITING_FACT_APPROVAL):
        machine.transition(state)
    gates = [
        ("facts", S.FACTS_APPROVED, S.STORY_GENERATING, S.WAITING_STORY_APPROVAL),
        ("story", S.STORY_APPROVED, S.SCRIPT_GENERATING, S.WAITING_SCRIPT_APPROVAL),
        ("script", S.SCRIPT_APPROVED, S.STORYBOARD_GENERATING, S.WAITING_STORYBOARD_APPROVAL),
        ("storyboard", S.STORYBOARD_APPROVED, S.GENERATING_MEDIA, S.ASSEMBLING),
    ]
    for stage, approved, generating, waiting in gates:
        with pytest.raises(InvalidTransitionError):
            machine.transition(approved)
        artifacts = {
            "facts": artifact(),
            "story": StoryPlan(title="title", thesis="thesis", central_question="why", hook="hook",
                beats=[dict(sequence=1, title="title", time_period="Tang", purpose="purpose",
                            fact_ids=["f1"], target_seconds=10)], ending="ending", target_duration_seconds=10),
            "script": Script(title="title", scenes=[dict(scene_id="s1", sequence=1,
                narration="narration", fact_ids=["f1"], duration_seconds=10)], target_duration_seconds=10),
            "storyboard": Storyboard(shots=[dict(shot_id="shot1", scene_id="s1", sequence=1,
                start_seconds=0, duration_seconds=10, visual_description="river", location="China",
                period="Tang", generation_method="STATIC_IMAGE", camera_motion="none", prompt="river")],
                estimated_media_cost_usd=0),
        }
        store.save(stage, artifacts[stage])
        assert machine.apply_human_decision(record(stage), store).current_state == approved
        machine.transition(generating)
        machine.transition(waiting)
    machine.transition(S.COMPLETE)
    assert store.list_versions("approvals") == [1, 2, 3, 4]
    with pytest.raises(InvalidTransitionError):
        machine.fail("error")


def test_invalid_jumps_and_failure_recovery() -> None:
    machine = ProjectStateMachine()
    with pytest.raises(InvalidTransitionError):
        machine.transition(S.SCRIPT_GENERATING)
    machine.transition(S.RESEARCHING)
    machine.fail("Network error")
    restored = ProjectStateMachine(RuntimeState.model_validate_json(machine.state.model_dump_json()))
    assert restored.state.last_successful_state == S.CREATED
    assert restored.state.failed_state == S.RESEARCHING
    assert restored.state.latest_error == "Network error"
    with pytest.raises(InvalidTransitionError):
        restored.transition(S.RESEARCH_COMPLETE)
    with pytest.raises(InvalidTransitionError):
        restored.fail("another error")
    assert restored.recover().current_state == S.RESEARCHING
    assert restored.state.latest_error is None
    assert restored.state.failed_state is None
    assert restored.state.last_successful_state == S.CREATED
    restored.transition(S.RESEARCH_COMPLETE)
    with pytest.raises(InvalidTransitionError):
        restored.recover()


@pytest.mark.parametrize("decision", ["REJECTED", "REVISION_REQUESTED"])
def test_rejection_preserves_version(tmp_path: Path, decision: str) -> None:
    store = ArtifactStore(tmp_path / "li_bai")
    store.save("facts", artifact())
    machine = machine_at(S.WAITING_FACT_APPROVAL)
    assert machine.apply_human_decision(record(decision=decision), store).current_state == S.FACT_CHECKING
    assert store.load("facts", 1, VerifiedFact) == artifact()
    assert store.load_latest("approvals", ApprovalRecord).decision == decision
    machine.transition(S.WAITING_FACT_APPROVAL)
    with pytest.raises(InvalidTransitionError):
        machine.apply_human_decision(record(), store)
    store.save("facts", artifact())
    assert machine.apply_human_decision(record(version=2), store).current_state == S.FACTS_APPROVED


def test_gate_rejects_missing_stale_or_other_project(tmp_path: Path) -> None:
    store = ArtifactStore(tmp_path / "li_bai")
    machine = machine_at(S.WAITING_FACT_APPROVAL)
    with pytest.raises(InvalidTransitionError):
        machine.apply_human_decision(record(), store)
    store.save("facts", artifact())
    store.save("facts", artifact())
    with pytest.raises(InvalidTransitionError):
        machine.apply_human_decision(record(), store)
    other = record(version=2).model_copy(update={"project_id": "other"})
    with pytest.raises(InvalidTransitionError):
        machine.apply_human_decision(other, store)
    assert machine.state.current_state == S.WAITING_FACT_APPROVAL
    assert store.list_versions("approvals") == []


def test_failed_approval_write_does_not_advance(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    store = ArtifactStore(tmp_path / "li_bai")
    store.save("facts", artifact())
    machine = machine_at(S.WAITING_FACT_APPROVAL)
    def fail(*args: object) -> None:
        raise OSError("disk full")
    monkeypatch.setattr(store, "save", fail)
    with pytest.raises(OSError):
        machine.apply_human_decision(record(), store)
    assert machine.state.current_state == S.WAITING_FACT_APPROVAL


def test_inconsistent_runtime_rejected() -> None:
    with pytest.raises(ValidationError):
        RuntimeState(current_state=S.FAILED)
    with pytest.raises(ValidationError):
        RuntimeState(current_state=S.RESEARCHING, last_successful_state=S.RESEARCHING)


def test_gate_validates_persisted_artifact_contract(tmp_path: Path) -> None:
    from history_studio.models import ProjectConfig
    store = ArtifactStore(tmp_path / "li_bai")
    store.save("facts", ProjectConfig(project_id="li_bai", topic="topic"))
    machine = machine_at(S.WAITING_FACT_APPROVAL)
    with pytest.raises(ValidationError):
        machine.apply_human_decision(record(), store)
    assert machine.state.current_state == S.WAITING_FACT_APPROVAL
    assert store.list_versions("approvals") == []


@pytest.mark.parametrize("checkpoint, running, next_checkpoint", [
    (S.CREATED, S.RESEARCHING, S.RESEARCH_COMPLETE),
    (S.RESEARCH_COMPLETE, S.FACT_CHECKING, S.WAITING_FACT_APPROVAL),
    (S.FACTS_APPROVED, S.STORY_GENERATING, S.WAITING_STORY_APPROVAL),
    (S.STORY_APPROVED, S.SCRIPT_GENERATING, S.WAITING_SCRIPT_APPROVAL),
    (S.SCRIPT_APPROVED, S.STORYBOARD_GENERATING, S.WAITING_STORYBOARD_APPROVAL),
])
def test_failure_preserves_checkpoint_and_retry_stage(checkpoint: S, running: S,
                                                     next_checkpoint: S) -> None:
    machine = machine_at(checkpoint)
    machine.transition(running)
    assert machine.state.last_successful_state == checkpoint
    assert machine.resume_state == running
    machine.fail("interrupted")
    restored = ProjectStateMachine(RuntimeState.model_validate_json(machine.state.model_dump_json()))
    assert restored.state.current_state == S.FAILED
    assert restored.state.last_successful_state == checkpoint
    assert restored.state.failed_state == running
    assert restored.resume_state == running
    restored.recover()
    assert restored.state.current_state == running
    assert restored.state.last_successful_state == checkpoint
    assert restored.state.failed_state is None
    assert restored.state.latest_error is None
    restored.fail("retry interrupted")
    assert restored.state.last_successful_state == checkpoint
    assert restored.state.failed_state == running
    restored.recover()
    restored.transition(next_checkpoint)
    assert restored.state.last_successful_state == next_checkpoint


def test_media_and_assembly_retain_checkpoint() -> None:
    machine = machine_at(S.STORYBOARD_APPROVED)
    machine.transition(S.GENERATING_MEDIA)
    machine.transition(S.ASSEMBLING)
    machine.fail("assembly interrupted")
    assert machine.state.last_successful_state == S.STORYBOARD_APPROVED
    assert machine.state.failed_state == S.ASSEMBLING
    machine.recover()
    machine.transition(S.COMPLETE)
    assert machine.state.last_successful_state == S.COMPLETE


def test_failure_at_approval_gate_recovers_same_checkpoint() -> None:
    machine = machine_at(S.WAITING_SCRIPT_APPROVAL)
    machine.fail("interrupted")
    assert machine.state.last_successful_state == S.WAITING_SCRIPT_APPROVAL
    assert machine.state.failed_state == S.WAITING_SCRIPT_APPROVAL
    assert machine.recover().current_state == S.WAITING_SCRIPT_APPROVAL


@pytest.mark.parametrize("changes", [
    {"failed_state": None}, {"failed_state": S.FAILED}, {"failed_state": S.COMPLETE},
    {"latest_error": None}, {"last_successful_state": S.SCRIPT_GENERATING},
    {"current_state": S.SCRIPT_GENERATING},
])
def test_invalid_failure_metadata(changes: dict) -> None:
    data = dict(current_state=S.FAILED, last_successful_state=S.STORY_APPROVED,
                failed_state=S.SCRIPT_GENERATING, latest_error="interrupted")
    with pytest.raises(ValidationError):
        RuntimeState(**(data | changes))


@pytest.mark.parametrize("previous_project, is_duplicate", [("li_bai", True), ("other", False)])
def test_approval_identity_includes_project(tmp_path: Path, previous_project: str,
                                            is_duplicate: bool) -> None:
    store = ArtifactStore(tmp_path / "li_bai")
    store.save("facts", artifact())
    previous = record().model_copy(update={"project_id": previous_project})
    store.save("approvals", previous)
    machine = machine_at(S.WAITING_FACT_APPROVAL)
    if is_duplicate:
        with pytest.raises(InvalidTransitionError, match="already has a decision"):
            machine.apply_human_decision(record(), store)
        assert machine.state.current_state == S.WAITING_FACT_APPROVAL
        assert store.list_versions("approvals") == [1]
    else:
        assert machine.apply_human_decision(record(), store).current_state == S.FACTS_APPROVED
        assert store.list_versions("approvals") == [1, 2]
