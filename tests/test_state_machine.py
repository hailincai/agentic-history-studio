from datetime import datetime, timezone
from pathlib import Path

import pytest
from pydantic import ValidationError

from history_studio.models import (Storyboard, ArtifactReference, VerificationPackage,
    create_verification_package, add_verification_result)
from history_studio.storage import ArtifactStore
from history_studio.workflow import (
    ApprovalRecord, InvalidTransitionError, ProjectState as S, ProjectStateMachine, RuntimeState, WorkflowArtifactBindings,
)


def record(stage: str = "facts", version: int = 1, decision: str = "APPROVED") -> ApprovalRecord:
    return ApprovalRecord(project_id="li_bai", stage=stage, artifact_type="verification" if stage == "facts" else stage,
                          artifact_version=version, decision=decision, decided_by="Human reviewer",
                          decision_source="human", decided_at=datetime.now(timezone.utc))


def research_artifact():
    from test_verification_package import research_data
    from history_studio.models.research_package import ResearchPlan, ResearchPackage
    research = research_data()
    research.project_id = "li_bai"
    ids = [fact.fact_id for fact in research.facts]
    research.plan = ResearchPlan(gaps=[dict(gap_id="G1", question="What happened?",
        completion_criteria=["Assess records"], status="COVERED", fact_ids=ids,
        coverage_assessment=dict(criteria=[dict(criterion="Assess records", addressed=True)],
            supporting_fact_ids=ids, rationale="Records assessed"))])
    research.progress.status = "COMPLETE"
    return ResearchPackage.model_validate(research.model_dump())


def ref(kind, version=1):
    return ArtifactReference(project_id="li_bai", artifact_type=kind, version=version)


def artifact() -> VerificationPackage:
    from test_verification_package import result_for
    package = create_verification_package(research_artifact(), research_input_ref=ref("research"))
    for index in range(len(package.research_facts)):
        package = add_verification_result(package, result_for(package, index))
    return package


def machine_at(state: S, store=None, version=1) -> ProjectStateMachine:
    bindings = WorkflowArtifactBindings()
    if state == S.WAITING_FACT_APPROVAL and store is not None:
        if not store.list_versions("research"):
            store.save("research", research_artifact())
        bindings = WorkflowArtifactBindings(research=ref("research"), verification=ref("verification", version))
    return ProjectStateMachine(RuntimeState(current_state=state, last_successful_state=state, artifacts=bindings))


def test_full_workflow_with_persisted_human_gates(tmp_path: Path) -> None:
    store = ArtifactStore(tmp_path / "li_bai")
    machine = ProjectStateMachine()
    for state in (S.RESEARCHING, S.RESEARCH_COMPLETE, S.FACT_CHECKING, S.WAITING_FACT_APPROVAL):
        machine.transition(state)
    machine = machine_at(S.WAITING_FACT_APPROVAL, store)
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
            "storyboard": Storyboard(shots=[dict(shot_id="shot1", scene_id="s1", sequence=1,
                start_seconds=0, duration_seconds=10, visual_description="river", location="China",
                period="Tang", generation_method="STATIC_IMAGE", camera_motion="none", prompt="river")],
                estimated_media_cost_usd=0),
        }
        decision_version = 1
        if stage == "story":
            from history_studio.story import build_story_context, StorySubmission, finalize_story_submission
            context = build_story_context(store, verification_input_ref=ref("verification"))
            proposal = StorySubmission(title="title", narrative_thesis="thesis", sections=[dict(
                section_id="section_01", purpose="purpose", beats=[dict(beat_id="beat_01",
                    narrative_role="opening", summary="Grounded opening", fact_proposals=[dict(
                        research_fact_id=context.eligible_facts[0].research_fact_id, use="AFFIRMATIVE")])])])
            version = store.save("story", finalize_story_submission(context, proposal))
            bindings = machine.state.artifacts.with_story(ref("story", version))
            machine = ProjectStateMachine(RuntimeState(current_state=S.WAITING_STORY_APPROVAL,
                last_successful_state=S.WAITING_STORY_APPROVAL, artifacts=bindings))
        elif stage == "script":
            from history_studio.script import build_script_context, ScriptSubmission, finalize_script_submission
            context = build_script_context(store, story_input_ref=machine.state.artifacts.approved_story)
            section = context.sections[0]
            beat = section.beats[0]
            proposal = ScriptSubmission(title="title", sections=[dict(
                section_id=section.section_id, title="opening", segments=[dict(
                    segment_id="s1", kind="HISTORICAL", narration="narration", grounding=dict(
                        story_beat_id=beat.beat_id, research_fact_ids=[beat.fact_refs[0].research_fact_id]))])])
            decision_version = store.save("script", finalize_script_submission(context, proposal))
            bindings = machine.state.artifacts.with_script(ref("script", decision_version))
            machine = ProjectStateMachine(RuntimeState(current_state=S.WAITING_SCRIPT_APPROVAL,
                last_successful_state=S.WAITING_SCRIPT_APPROVAL, artifacts=bindings))
        else:
            store.save("verification" if stage == "facts" else stage, artifacts[stage])
        assert machine.apply_human_decision(record(stage, version=decision_version), store).current_state == approved
        machine.transition(generating)
        machine.transition(waiting)
    machine = ProjectStateMachine(RuntimeState(current_state=S.ASSEMBLING, last_successful_state=S.STORYBOARD_APPROVED,
        artifacts=machine.state.artifacts.with_media(ref("media"))))
    machine.complete_assembly(ref("assembly"), project_id="li_bai")
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
    store.save("verification", artifact())
    machine = machine_at(S.WAITING_FACT_APPROVAL, store)
    assert machine.apply_human_decision(record(decision=decision), store).current_state == S.FACT_CHECKING
    assert store.load("verification", 1, VerificationPackage) == artifact()
    assert store.load_latest("approvals", ApprovalRecord).decision == decision
    machine.transition(S.WAITING_FACT_APPROVAL)
    with pytest.raises(InvalidTransitionError):
        machine.apply_human_decision(record(), store)
    store.save("verification", artifact())
    machine = machine_at(S.WAITING_FACT_APPROVAL, store, version=2)
    assert machine.apply_human_decision(record(version=2), store).current_state == S.FACTS_APPROVED


def test_gate_rejects_missing_stale_or_other_project(tmp_path: Path) -> None:
    store = ArtifactStore(tmp_path / "li_bai")
    machine = machine_at(S.WAITING_FACT_APPROVAL, store, version=2)
    with pytest.raises(FileNotFoundError):
        machine.apply_human_decision(record(), store)
    store.save("verification", artifact())
    store.save("verification", artifact())
    with pytest.raises(InvalidTransitionError):
        machine.apply_human_decision(record(), store)
    other = record(version=2).model_copy(update={"project_id": "other"})
    with pytest.raises(InvalidTransitionError):
        machine.apply_human_decision(other, store)
    assert machine.state.current_state == S.WAITING_FACT_APPROVAL
    assert store.list_versions("approvals") == []


def test_failed_approval_write_does_not_advance(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    store = ArtifactStore(tmp_path / "li_bai")
    store.save("verification", artifact())
    machine = machine_at(S.WAITING_FACT_APPROVAL, store)
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
    store.save("verification", ProjectConfig(project_id="li_bai", topic="topic"))
    machine = machine_at(S.WAITING_FACT_APPROVAL, store)
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
    machine = ProjectStateMachine(RuntimeState(current_state=S.ASSEMBLING, last_successful_state=S.STORYBOARD_APPROVED,
        artifacts=WorkflowArtifactBindings(media=ref("media"))))
    machine.complete_assembly(ref("assembly"), project_id="li_bai")
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
    store.save("verification", artifact())
    previous = record().model_copy(update={"project_id": previous_project})
    store.save("approvals", previous)
    machine = machine_at(S.WAITING_FACT_APPROVAL, store)
    if is_duplicate:
        with pytest.raises(InvalidTransitionError, match="already has a decision"):
            machine.apply_human_decision(record(), store)
        assert machine.state.current_state == S.WAITING_FACT_APPROVAL
        assert store.list_versions("approvals") == [1]
    else:
        assert machine.apply_human_decision(record(), store).current_state == S.FACTS_APPROVED
        assert store.list_versions("approvals") == [1, 2]
