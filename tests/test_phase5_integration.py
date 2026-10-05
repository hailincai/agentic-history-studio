"""Phase 5 durable integration: real production boundaries, only model providers fake.

Written for manual execution by the user; no production changes accompany these tests.
"""
import json
from datetime import datetime, timezone

import pytest

from history_studio.cli import read_project
from history_studio.model_io import ModelRequest, ModelResponse, NativeToolCall
from history_studio.models import ScriptPackage, StoryPackage
from history_studio.script import ScriptWriter, build_script_context, validate_script_grounding
from history_studio.script.preparation import INSTRUCTIONS
from history_studio.storage.artifact_store import ArtifactStore, write_json
from history_studio.workflow import ApprovalRecord, ProjectState as S, ProjectStateMachine
from history_studio.workflow.script import ScriptWorkflow
from history_studio.workflow.script_review import apply_script_review, load_script_review, script_review_lines
from history_studio.workflow.story_review import apply_story_review
from test_phase4_integration import phase3_start, generate as generate_story, story_decision


def approved_story_start(tmp_path):
    """Reach the real Story Human Gate from persisted approved verification knowledge."""
    project, store = phase3_start(tmp_path)
    outcome, _ = generate_story(project, store)
    approved = apply_story_review(store.project_dir, story_decision(outcome.state))
    assert approved.current_state == S.STORY_APPROVED
    assert approved.artifacts.story == approved.artifacts.approved_story
    return project, store, approved.artifacts.approved_story


class ScriptProvider:
    """External provider fake; raw native arguments go through real parsing/finalization."""

    def __init__(self, context):
        self.expected_context = context.model_dump(mode="json")
        self.requests = []

    def decide(self, request: ModelRequest) -> ModelResponse:
        payload = json.loads(request.input)
        assert payload == self.expected_context
        assert request.instructions == INSTRUCTIONS
        assert [tool["name"] for tool in request.tools] == ["submit_script"]
        assert request.tool_choice == "required"
        self.requests.append(request)
        section = payload["sections"][0]
        beat = next(item for item in section["beats"] if item["kind"] == "historical")
        proposal = dict(title="A grounded life journey in narration", sections=[dict(
                    section_id=section["section_id"], title="The early journey", segments=[dict(
                        segment_id="narration_01", kind="HISTORICAL",
                        narration="The approved accounts describe an early journey; its exact date remains uncertain.",
                        grounding=dict(story_beat_id=beat["beat_id"],
                                       research_fact_ids=[ref["research_fact_id"] for ref in beat["fact_refs"]])),
                        dict(segment_id="transition_01", kind="STRUCTURAL",
                             narration="Now we turn to the next part of the story.", grounding=None)])])
        return ModelResponse(status="completed", usage={}, tool_calls=[NativeToolCall(
            call_id="script_submission", name="submit_script", arguments=json.dumps(proposal))])


def generate_script(project, store):
    providers = []

    def factory(context):
        _, active = read_project(store.project_dir)
        assert active.current_state == S.SCRIPT_GENERATING
        assert context.story_input_ref == active.artifacts.approved_story
        provider = ScriptProvider(context)
        providers.append(provider)
        return provider

    outcome = ScriptWorkflow(provider_factory=factory).run(project, store)
    assert outcome.state.current_state == S.WAITING_SCRIPT_APPROVAL
    assert outcome.stage.stop_reason == "SUBMITTED"
    assert outcome.stage.package is not None
    assert providers and all(provider.requests for provider in providers)
    return outcome


def decision(state, kind="APPROVED"):
    ref = state.artifacts.script
    return ApprovalRecord(project_id=ref.project_id, stage="script", artifact_type="script",
        artifact_version=ref.version, decision=kind, feedback="Exact narration reviewed by a human",
        decided_by="Integration reviewer", decision_source="human",
        decided_at=datetime(2026, 10, 5, 14, tzinfo=timezone.utc))


def test_phase5_generation_persistence_restart_and_exact_human_approval(tmp_path):
    project, store, approved_story = approved_story_start(tmp_path)
    outcome = generate_script(project, store)
    waiting = outcome.state
    script_ref = waiting.artifacts.script
    assert waiting.artifacts.approved_story == approved_story
    assert waiting.artifacts.approved_script is None
    durable = store.load("script", script_ref.version, ScriptPackage)
    assert durable == outcome.stage.package and durable.story_input_ref == approved_story
    context = build_script_context(store, story_input_ref=approved_story)
    assert validate_script_grounding(context, durable).is_valid
    historical, structural = durable.sections[0].segments
    assert durable.sections[0].section_id == "section_01"
    assert historical.segment_id == "narration_01" and historical.kind == "HISTORICAL"
    assert historical.grounding.story_beat_id == "beat_01"
    assert historical.grounding.research_fact_ids == ("RF-other", "RF-target")
    assert "exact date remains uncertain" in historical.narration
    assert structural.kind == "STRUCTURAL" and structural.grounding is None
    path = store.project_dir
    del project, store, outcome, waiting, context
    fresh_store = ArtifactStore(path)
    _, reloaded = read_project(path)
    assert reloaded.artifacts.approved_story == approved_story
    assert reloaded.artifacts.script == script_ref
    target = load_script_review(reloaded, fresh_store)
    assert target == durable
    assert script_review_lines(reloaded, target)[0] == f"Review li_bai/script:v{script_ref.version}"
    versions = fresh_store.list_versions("script")
    approved = apply_script_review(path, decision(reloaded))
    assert approved.current_state == S.SCRIPT_APPROVED
    assert approved.artifacts.approved_script == approved.artifacts.script == script_ref
    assert approved.artifacts.approved_story == approved_story
    assert read_project(path)[1] == approved
    assert fresh_store.list_versions("script") == versions
    assert fresh_store.load("script", script_ref.version, ScriptPackage) == durable
    audit_versions = fresh_store.list_versions("approvals")
    assert fresh_store.load("approvals", audit_versions[-1], ApprovalRecord) == decision(reloaded)


def test_phase5_newer_unbound_script_does_not_replace_review_target(tmp_path):
    project, store, _ = approved_story_start(tmp_path)
    outcome = generate_script(project, store)
    bound = outcome.state.artifacts.script
    original = store.load("script", bound.version, ScriptPackage)
    newer = original.model_copy(deep=True)
    newer.title = "Newer unbound narration"
    newer_version = store.save("script", newer)
    assert newer_version > bound.version
    fresh_store = ArtifactStore(store.project_dir)
    _, reloaded = read_project(store.project_dir)
    target = load_script_review(reloaded, fresh_store)
    assert target == original and target != newer
    assert "Newer unbound narration" not in "\n".join(script_review_lines(reloaded, target))
    with pytest.raises(ValueError, match="exact bound"):
        apply_script_review(store.project_dir, decision(reloaded).model_copy(update={"artifact_version": newer_version}))
    approved = apply_script_review(store.project_dir, decision(reloaded))
    assert approved.artifacts.approved_script == bound
    assert fresh_store.load("script", newer_version, ScriptPackage) == newer


def test_phase5_newer_unapproved_story_does_not_change_script_authority(tmp_path):
    project, store, approved_ref = approved_story_start(tmp_path)
    original = store.load("story", approved_ref.version, StoryPackage)
    newer = original.model_copy(deep=True)
    newer.plan.title = "Unapproved replacement blueprint"
    newer.plan.sections[0].beats[0].beat_id = "unapproved_beat"
    assert store.save("story", newer) > approved_ref.version
    # Neither store contents nor the ordinary Story output binding may replace approval.
    outcome = generate_script(project, ArtifactStore(store.project_dir))
    durable = store.load("script", outcome.state.artifacts.script.version, ScriptPackage)
    assert durable.story_input_ref == approved_ref
    assert durable.sections[0].segments[0].grounding.story_beat_id == "beat_01"
    assert outcome.state.artifacts.approved_story == approved_ref
    assert validate_script_grounding(build_script_context(store, story_input_ref=approved_ref), durable).is_valid


def test_phase5_tampered_persisted_provenance_fails_closed_on_review(tmp_path):
    project, store, approved_ref = approved_story_start(tmp_path)
    waiting = generate_script(project, store).state
    original = store.load("script", waiting.artifacts.script.version, ScriptPackage)
    story = store.load("story", approved_ref.version, StoryPackage)
    alternative_version = store.save("story", story)
    data = original.model_dump(mode="json")
    data["story_input_ref"]["version"] = alternative_version
    # Test-only tampering preserves individual schema validity but breaks exact lineage.
    assert ScriptPackage.model_validate(data).story_input_ref != approved_ref
    path = store.project_dir / "script" / f"script_v{waiting.artifacts.script.version}.json"
    path.write_text(json.dumps(data), encoding="utf-8")
    fresh_store = ArtifactStore(store.project_dir)
    _, reloaded = read_project(store.project_dir)
    before_audits = fresh_store.list_versions("approvals")
    with pytest.raises(ValueError, match="STORY_PROVENANCE_MISMATCH"):
        load_script_review(reloaded, fresh_store)
    with pytest.raises(ValueError, match="STORY_PROVENANCE_MISMATCH"):
        apply_script_review(store.project_dir, decision(reloaded))
    assert read_project(store.project_dir)[1] == waiting
    assert fresh_store.list_versions("approvals") == before_audits


def test_phase5_orphan_recovery_regenerates_from_durable_approved_story(tmp_path):
    project, store, approved_ref = approved_story_start(tmp_path)
    _, start = read_project(store.project_dir)
    machine = ProjectStateMachine(start)
    machine.transition(S.SCRIPT_GENERATING)
    context = build_script_context(store, story_input_ref=approved_ref)
    orphan = ScriptWriter(context, provider=ScriptProvider(context)).generate().package
    assert orphan is not None
    orphan_version = store.save("script", orphan)
    machine.fail("script_binding_publication_failed")
    write_json(store.project_dir / ".runtime/state.json", machine.state, replace=True)
    path = store.project_dir
    del store, context, machine, project
    fresh_store = ArtifactStore(path)
    project, interrupted = read_project(path)
    assert interrupted.failed_state == S.SCRIPT_GENERATING
    assert interrupted.artifacts.script is None
    assert interrupted.artifacts.approved_story == approved_ref
    recovered = generate_script(project, fresh_store)
    bound = recovered.state.artifacts.script
    assert bound.version > orphan_version
    assert fresh_store.load("script", orphan_version, ScriptPackage) == orphan
    assert fresh_store.load("script", bound.version, ScriptPackage).story_input_ref == approved_ref
    approved = apply_script_review(path, decision(recovered.state))
    assert approved.artifacts.approved_script == bound


def test_phase5_revision_preserves_reviewed_script_and_approved_story(tmp_path):
    project, store, approved_ref = approved_story_start(tmp_path)
    waiting = generate_script(project, store).state
    _, restarted = read_project(store.project_dir)
    versions = store.list_versions("script")
    reviewed = load_script_review(restarted, ArtifactStore(store.project_dir))
    revised = apply_script_review(store.project_dir, decision(restarted, "REVISION_REQUESTED"))
    assert revised.current_state == S.SCRIPT_GENERATING
    assert revised.artifacts.script == waiting.artifacts.script
    assert revised.artifacts.approved_script is None
    assert revised.artifacts.approved_story == approved_ref
    assert revised.artifacts.storyboard is revised.artifacts.approved_storyboard is None
    assert store.list_versions("script") == versions
    assert store.load("script", revised.artifacts.script.version, ScriptPackage) == reviewed
    assert read_project(store.project_dir)[1] == revised
