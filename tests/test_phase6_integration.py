"""Phase 6 durable integration: real production boundaries, only providers fake.

Written for manual execution by the user; no production changes accompany these tests.
"""
import json
from datetime import datetime, timezone

import pytest

from history_studio.cli import read_project
from history_studio.model_io import ModelRequest, ModelResponse, NativeToolCall
from history_studio.models import ScriptPackage, StoryboardPackage
from history_studio.storage.artifact_store import ArtifactStore, write_json
from history_studio.visual_director import VisualDirector, build_visual_director_context, validate_storyboard_integrity
from history_studio.visual_director.preparation import INSTRUCTIONS
from history_studio.workflow import ApprovalRecord, ProjectState as S, RuntimeState
from history_studio.workflow.storyboard import StoryboardWorkflow
from history_studio.workflow.storyboard_review import (
    apply_storyboard_review, load_storyboard_review, storyboard_review_lines,
)
from history_studio.workflow.script_review import apply_script_review
from test_phase5_integration import approved_story_start, generate_script, decision as script_decision


@pytest.fixture(autouse=True)
def no_paid_api(monkeypatch):
    from openai.resources.responses import Responses

    def forbidden(*args, **kwargs):
        pytest.fail("Phase 6 integration uses fake providers only")

    monkeypatch.setattr(Responses, "create", forbidden)


def approved_script_start(tmp_path):
    """Reach SCRIPT_APPROVED through the real preceding durable workflows and gates."""
    project, store, _ = approved_story_start(tmp_path)
    waiting = generate_script(project, store).state
    approved = apply_script_review(store.project_dir, script_decision(waiting))
    assert approved.current_state == S.SCRIPT_APPROVED
    assert approved.artifacts.approved_script == approved.artifacts.script
    return project, store, approved.artifacts.approved_script


def protect_phase6_boundary(monkeypatch):
    """After setup, Phase 6 may load only immediate authority, candidate and audits."""
    load = ArtifactStore.load

    def bounded_load(self, kind, version, model):
        assert kind in ("script", "storyboard", "approvals"), "Phase 6 must not restart upstream semantics"
        return load(self, kind, version, model)

    def no_latest(*args, **kwargs):
        pytest.fail("Latest artifact lookup cannot establish Phase 6 lineage")

    monkeypatch.setattr(ArtifactStore, "load", bounded_load)
    monkeypatch.setattr(ArtifactStore, "load_latest", no_latest)


class VisualProvider:
    """Return raw native arguments through real parsing, finalization and validation."""

    def __init__(self, context, treatment="First visual treatment"):
        self.expected_context = context.model_dump(mode="json")
        self.treatment = treatment
        self.requests = []

    def decide(self, request: ModelRequest) -> ModelResponse:
        payload = json.loads(request.input)
        assert payload == self.expected_context
        assert request.instructions == INSTRUCTIONS
        assert [tool["name"] for tool in request.tools] == ["submit_storyboard"]
        assert request.tool_choice == "required"
        self.requests.append(request)
        sections = []
        for section in payload["sections"]:
            shots = []
            for segment in section["segments"]:
                # Historical narration gets two shots; structural narration gets one.
                for index in range(2 if segment["kind"] == "HISTORICAL" else 1):
                    shots.append(dict(shot_id=f"{segment['segment_id']}_shot_{index + 1}",
                        kind=segment["kind"], source_segment_id=segment["segment_id"],
                        visual_description=f"{self.treatment}: restrained atmospheric landscape",
                        generation_prompt="Generic landscape with cinematic lighting; artistic interpretation only",
                        generation_method="STATIC_IMAGE", framing="WIDE", camera_motion="NONE",
                        estimated_duration_seconds=3.5))
            sections.append(dict(section_id=section["section_id"], title="Proposed section title", shots=shots))
        proposal = dict(title="Proposed package title", sections=sections)
        return ModelResponse(status="completed", usage={}, tool_calls=[NativeToolCall(
            call_id="storyboard_submission", name="submit_storyboard", arguments=json.dumps(proposal))])


def generate_storyboard(project, store, treatment="First visual treatment"):
    providers = []

    def factory(context):
        _, active = read_project(store.project_dir)
        assert active.current_state == S.STORYBOARD_GENERATING
        assert context.script_input_ref == active.artifacts.approved_script
        provider = VisualProvider(context, treatment)
        providers.append(provider)
        return provider

    outcome = StoryboardWorkflow(provider_factory=factory).run(project, store)
    assert outcome.state.current_state == S.WAITING_STORYBOARD_APPROVAL
    assert outcome.stage.stop_reason == "SUBMITTED" and outcome.stage.package is not None
    assert len(providers) == 1 and len(providers[0].requests) == 1
    return outcome, providers[0]


def decision(state, kind="APPROVED"):
    ref = state.artifacts.storyboard
    return ApprovalRecord(project_id=ref.project_id, stage="storyboard", artifact_type="storyboard",
        artifact_version=ref.version, decision=kind, feedback="Human reviewed exact visual plan",
        decided_by="Integration reviewer", decision_source="human",
        decided_at=datetime(2026, 10, 5, 15, tzinfo=timezone.utc))


def assert_upstream_preserved(before, after):
    for field in ("research", "verification", "approved_verification", "story", "approved_story", "script", "approved_script"):
        assert getattr(after.artifacts, field) == getattr(before.artifacts, field)


def test_phase6_happy_path_restart_review_and_exact_approval(tmp_path, monkeypatch):
    project, store, approved_script = approved_script_start(tmp_path)
    start = read_project(store.project_dir)[1]
    protect_phase6_boundary(monkeypatch)
    outcome, _ = generate_storyboard(project, store)
    waiting = outcome.state
    ref = waiting.artifacts.storyboard
    assert ref.project_id == project.project_id and ref.artifact_type == "storyboard"
    assert waiting.artifacts.approved_storyboard is None
    assert_upstream_preserved(start, waiting)
    durable = store.load("storyboard", ref.version, StoryboardPackage)
    assert durable == outcome.stage.package and durable.script_input_ref == approved_script
    context = build_visual_director_context(store, script_input_ref=approved_script)
    assert validate_storyboard_integrity(context, durable).is_valid
    assert [shot.source_segment_id for shot in durable.sections[0].shots] == [
        "narration_01", "narration_01", "transition_01"]
    assert [shot.kind for shot in durable.sections[0].shots] == ["HISTORICAL", "HISTORICAL", "STRUCTURAL"]
    path = store.project_dir
    fresh = ArtifactStore(path)
    restarted = read_project(path)[1]
    assert restarted == waiting
    reviewed = load_storyboard_review(restarted, fresh)
    assert reviewed == durable
    lines = storyboard_review_lines(restarted, reviewed, context)
    assert lines[0] == f"Review {project.project_id}/storyboard:v{ref.version}"
    assert any("exact date remains uncertain" in line for line in lines)
    versions = fresh.list_versions("storyboard")
    approved = apply_storyboard_review(path, decision(restarted))
    assert approved.current_state == S.STORYBOARD_APPROVED
    assert approved.artifacts.approved_storyboard == approved.artifacts.storyboard == ref
    assert approved.artifacts.approved_script == approved_script
    assert_upstream_preserved(start, approved)
    assert read_project(path)[1] == approved
    assert fresh.list_versions("storyboard") == versions
    assert fresh.load("storyboard", ref.version, StoryboardPackage) == durable
    assert fresh.load("approvals", fresh.list_versions("approvals")[-1], ApprovalRecord) == decision(restarted)


def test_phase6_preexisting_orphan_is_never_adopted(tmp_path, monkeypatch):
    project, store, approved_script = approved_script_start(tmp_path)
    protect_phase6_boundary(monkeypatch)
    context = build_visual_director_context(store, script_input_ref=approved_script)
    orphan = VisualDirector(context, provider=VisualProvider(context, "Orphan treatment")).generate().package
    assert orphan is not None and validate_storyboard_integrity(context, orphan).is_valid
    orphan_version = store.save("storyboard", orphan)
    start = read_project(store.project_dir)[1]
    assert start.current_state == S.SCRIPT_APPROVED and start.artifacts.storyboard is None
    outcome, provider = generate_storyboard(project, ArtifactStore(store.project_dir), "Official treatment")
    ref = outcome.state.artifacts.storyboard
    assert provider.requests and ref.version > orphan_version
    official = store.load("storyboard", ref.version, StoryboardPackage)
    assert official != orphan and official.script_input_ref == approved_script
    assert store.load("storyboard", orphan_version, StoryboardPackage) == orphan
    assert load_storyboard_review(outcome.state, store) == official
    approved = apply_storyboard_review(store.project_dir, decision(outcome.state))
    assert approved.artifacts.approved_storyboard == ref
    assert_upstream_preserved(start, approved)


def test_phase6_newer_unapproved_script_cannot_drive_generation(tmp_path, monkeypatch):
    project, store, approved_script = approved_script_start(tmp_path)
    original = store.load("script", approved_script.version, ScriptPackage)
    newer = original.model_copy(deep=True)
    newer.title = "Unapproved replacement narration"
    newer.sections[0].segments[0].segment_id = "unapproved_segment"
    newer.sections[0].segments[0].narration = "Unapproved replacement text"
    version = store.save("script", newer)
    assert version > approved_script.version
    state = read_project(store.project_dir)[1]
    data = state.model_dump(mode="json")
    data["artifacts"]["script"]["version"] = version
    write_json(store.project_dir / ".runtime/state.json", RuntimeState.model_validate(data), replace=True)
    start = read_project(store.project_dir)[1]
    protect_phase6_boundary(monkeypatch)
    outcome, provider = generate_storyboard(project, ArtifactStore(store.project_dir))
    assert json.loads(provider.requests[0].input)["script_input_ref"] == approved_script.model_dump(mode="json")
    assert "unapproved_segment" not in provider.requests[0].input
    durable = store.load("storyboard", outcome.state.artifacts.storyboard.version, StoryboardPackage)
    assert durable.script_input_ref == approved_script != start.artifacts.script
    assert durable.title == original.title
    approved = apply_storyboard_review(store.project_dir, decision(outcome.state))
    assert_upstream_preserved(start, approved)


def test_phase6_tampered_provenance_blocks_review_and_approval(tmp_path, monkeypatch):
    project, store, approved_script = approved_script_start(tmp_path)
    alternative = store.save("script", store.load("script", approved_script.version, ScriptPackage))
    protect_phase6_boundary(monkeypatch)
    waiting = generate_storyboard(project, store)[0].state
    ref = waiting.artifacts.storyboard
    original = store.load("storyboard", ref.version, StoryboardPackage)
    data = original.model_dump(mode="json")
    data["script_input_ref"]["version"] = alternative
    assert StoryboardPackage.model_validate(data).script_input_ref != approved_script
    path = store.project_dir / "storyboard" / f"storyboard_v{ref.version}.json"
    path.write_text(json.dumps(data), encoding="utf-8")
    fresh = ArtifactStore(store.project_dir)
    restarted = read_project(store.project_dir)[1]
    audits = fresh.list_versions("approvals")
    with pytest.raises(ValueError, match="SCRIPT_PROVENANCE_MISMATCH"):
        load_storyboard_review(restarted, fresh)
    with pytest.raises(ValueError, match="SCRIPT_PROVENANCE_MISMATCH"):
        apply_storyboard_review(store.project_dir, decision(restarted))
    after = read_project(store.project_dir)[1]
    assert after == waiting and after.current_state == S.WAITING_STORYBOARD_APPROVAL
    assert after.artifacts.approved_storyboard is None
    assert fresh.list_versions("approvals") == audits


@pytest.mark.parametrize("kind", ["REVISION_REQUESTED", "REJECTED"])
def test_phase6_revision_creates_new_immutable_lineage(tmp_path, monkeypatch, kind):
    project, store, approved_script = approved_script_start(tmp_path)
    start = read_project(store.project_dir)[1]
    protect_phase6_boundary(monkeypatch)
    first = generate_storyboard(project, store)[0].state
    old_ref = first.artifacts.storyboard
    old = load_storyboard_review(first, store)
    old_path = store.project_dir / "storyboard" / f"storyboard_v{old_ref.version}.json"
    old_bytes = old_path.read_bytes()
    revised = apply_storyboard_review(store.project_dir, decision(first, kind))
    assert revised.current_state == S.STORYBOARD_GENERATING
    assert revised.artifacts.storyboard == old_ref and revised.artifacts.approved_storyboard is None
    assert_upstream_preserved(start, revised)
    second = generate_storyboard(project, ArtifactStore(store.project_dir), "Revised treatment")[0].state
    new_ref = second.artifacts.storyboard
    assert new_ref.version > old_ref.version and new_ref != old_ref
    assert old_path.read_bytes() == old_bytes
    assert store.load("storyboard", old_ref.version, StoryboardPackage) == old
    assert store.load("storyboard", new_ref.version, StoryboardPackage) != old
    assert second.artifacts.approved_script == approved_script
    approved = apply_storyboard_review(store.project_dir, decision(second))
    assert approved.current_state == S.STORYBOARD_APPROVED
    assert approved.artifacts.approved_storyboard == approved.artifacts.storyboard == new_ref
    assert_upstream_preserved(start, approved)


def test_phase6_stale_human_intent_cannot_approve_regenerated_candidate(tmp_path, monkeypatch):
    project, store, approved_script = approved_script_start(tmp_path)
    protect_phase6_boundary(monkeypatch)
    first = generate_storyboard(project, store)[0].state
    load_storyboard_review(first, store)
    stale = decision(first)
    apply_storyboard_review(store.project_dir, decision(first, "REVISION_REQUESTED"))
    second = generate_storyboard(project, ArtifactStore(store.project_dir), "New candidate treatment")[0].state
    assert second.artifacts.storyboard.version > stale.artifact_version
    audits = store.list_versions("approvals")
    with pytest.raises(ValueError, match="exact bound"):
        apply_storyboard_review(store.project_dir, stale)
    assert read_project(store.project_dir)[1] == second
    assert second.artifacts.approved_storyboard is None
    assert store.list_versions("approvals") == audits
    approved = apply_storyboard_review(store.project_dir, decision(second))
    assert approved.current_state == S.STORYBOARD_APPROVED
    assert approved.artifacts.approved_storyboard == approved.artifacts.storyboard == second.artifacts.storyboard
    assert approved.artifacts.approved_script == approved_script
    assert_upstream_preserved(first, approved)
