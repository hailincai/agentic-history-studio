"""Phase 4 durable integration: real domain boundaries, only the LLM is fake.

Constructed for manual execution by the user; no production changes are needed.
"""
import json
from datetime import datetime, timezone

import pytest

from history_studio.cli import read_project
from history_studio.model_io import ModelRequest, ModelResponse, NativeToolCall
from history_studio.models import (
    ArtifactReference, HistoricalTime, ProjectConfig, StoryPackage, VerificationPackage,
    add_verification_result, create_verification_package,
)
from history_studio.storage.artifact_store import ArtifactStore, write_json
from history_studio.story import StoryArchitect, StorySubmission, build_story_context
from history_studio.story.preparation import INSTRUCTIONS
from history_studio.workflow import ApprovalRecord, ProjectState as S, ProjectStateMachine, RuntimeState
from history_studio.workflow.story import StoryWorkflow
from history_studio.workflow.story_review import apply_story_review, load_story_review, story_review_lines
from test_state_machine import research_artifact
from test_verification_package import result_for


def exact_ref(kind, version):
    return ArtifactReference(project_id="li_bai", artifact_type=kind, version=version)


def phase3_start(tmp_path):
    """Minimum complete P3 knowledge, approved through the real existing Fact Gate."""
    research = research_artifact()
    research.facts[0].historical_time = HistoricalTime(display="701", start_year=701, precision="YEAR")
    research.facts[1].historical_time = HistoricalTime(display="circa 725", start_year=725,
                                                     precision="APPROXIMATE")
    project = ProjectConfig(project_id="li_bai", topic=research.topic, research_scope=research.research_scope)
    store = ArtifactStore(tmp_path / project.project_id)
    for _ in range(4):
        store.save("research", research)
    research_ref = exact_ref("research", 4)
    package = create_verification_package(research, research_input_ref=research_ref)
    for index in range(len(package.research_facts)):
        changes = {"unresolved_issues": ["The exact date remains uncertain"]} if index == 1 else {}
        package = add_verification_result(package, result_for(package, index,
            status="PARTIALLY_VERIFIED" if index == 1 else "VERIFIED", **changes))
    assert package.is_complete
    for _ in range(2):
        store.save("verification", package)
    verification_ref = exact_ref("verification", 2)
    machine = ProjectStateMachine()
    machine.transition(S.RESEARCHING)
    machine.complete_research(research_ref, project_id=project.project_id)
    machine.transition(S.FACT_CHECKING)
    machine.complete_verification(verification_ref, project_id=project.project_id)
    machine.apply_fact_review_decision(ApprovalRecord(project_id=project.project_id, stage="facts",
        artifact_type="verification", artifact_version=2, decision="APPROVED",
        decided_by="Integration reviewer", decision_source="human",
        decided_at=datetime(2026, 10, 5, 12, tzinfo=timezone.utc)), store)
    write_json(store.project_dir / "project.json", project)
    write_json(store.project_dir / ".runtime/state.json", machine.state)
    assert machine.state.current_state == S.FACTS_APPROVED
    assert machine.state.artifacts.approved_verification == verification_ref
    return project, store


class NarrativeProvider:
    """Generic provider fake returning only an Agent-owned terminal proposal."""
    def __init__(self, expected_context):
        self.expected_context = expected_context.model_dump(mode="json")
        self.requests = []

    def decide(self, request: ModelRequest) -> ModelResponse:
        assert json.loads(request.input) == self.expected_context
        assert request.instructions == INSTRUCTIONS
        assert [tool["name"] for tool in request.tools] == ["submit_story"]
        assert request.tool_choice == "required"
        self.requests.append(request)
        payload = dict(title="A grounded life journey", narrative_thesis="Trace change while preserving uncertainty",
            sections=[dict(section_id="section_01", purpose="Introduce the journey", beats=[
                dict(beat_id="beat_01", kind="historical", narrative_role="development",
                    summary="Organize the supported accounts with their chronological uncertainty",
                    fact_proposals=[dict(research_fact_id="RF-other", use="AFFIRMATIVE", qualification=None),
                                    dict(research_fact_id="RF-target", use="QUALIFIED",
                                         qualification="The exact date remains uncertain")],
                    uncertainty_notes=["The second account has an approximate date"]),
                dict(beat_id="transition_01", kind="structural", narrative_role="transition",
                    summary="Move to the next period", fact_proposals=[], uncertainty_notes=[]),
            ])])
        # The contract validates the fake's proposal, but never creates the durable output.
        proposal = StorySubmission.model_validate(payload)
        return ModelResponse(status="completed", usage={}, tool_calls=[NativeToolCall(
            call_id="story_submission", name="submit_story", arguments=proposal.model_dump_json())])


def generate(project, store):
    providers, contexts = [], []
    def factory(context):
        _, active = read_project(store.project_dir)
        assert active.current_state == S.STORY_GENERATING
        assert context.verification_input_ref == active.artifacts.approved_verification
        contexts.append(context)
        provider = NarrativeProvider(context)
        providers.append(provider)
        return provider
    outcome = StoryWorkflow(provider_factory=factory).run(project, store)
    assert outcome.state.current_state == S.WAITING_STORY_APPROVAL
    assert outcome.stage.stop_reason == "SUBMITTED" and outcome.stage.package is not None
    assert providers and all(provider.requests for provider in providers)
    return outcome, contexts


def story_decision(state):
    target = state.artifacts.story
    return ApprovalRecord(project_id=target.project_id, stage="story", artifact_type="story",
        artifact_version=target.version, decision="APPROVED", feedback="Exact blueprint reviewed",
        decided_by="Integration reviewer", decision_source="human",
        decided_at=datetime(2026, 10, 5, 13, tzinfo=timezone.utc))


def forbid_latest(monkeypatch):
    def forbidden(*args, **kwargs):
        pytest.fail("Global latest discovery cannot establish Phase 4 lineage")
    monkeypatch.setattr(ArtifactStore, "load_latest", forbidden)


def test_phase4_happy_path_complete_durable_lineage(tmp_path, monkeypatch):
    project, store = phase3_start(tmp_path)
    forbid_latest(monkeypatch)
    outcome, contexts = generate(project, store)
    waiting = outcome.state
    bound = waiting.artifacts.story
    durable = store.load("story", bound.version, StoryPackage)
    verification = store.load("verification", 2, VerificationPackage)
    assert verification.research_input_ref == waiting.artifacts.research == exact_ref("research", 4)
    assert waiting.artifacts.verification == waiting.artifacts.approved_verification == exact_ref("verification", 2)
    assert contexts[0].research_input_ref == verification.research_input_ref
    assert contexts[0].verification_input_ref == durable.verification_input_ref == exact_ref("verification", 2)
    assert verification.schema_version != 2 and durable.verification_input_ref.version != durable.schema_version
    assert durable == outcome.stage.package
    beat = durable.plan.sections[0].beats[0]
    assert [fact.status.value for fact in beat.fact_refs] == ["VERIFIED", "PARTIALLY_VERIFIED"]
    assert beat.fact_refs[1].qualification == "The exact date remains uncertain"
    assert [entry.historical_time for entry in beat.fact_chronology] == [
        contexts[0].fact("RF-other").historical_time, contexts[0].fact("RF-target").historical_time]
    assert [entry.historical_time.start_year for entry in beat.fact_chronology] == [701, 725]
    assert "historical_time" not in beat.model_dump()
    assert load_story_review(waiting, store) == durable
    versions_before = store.list_versions("story")
    approved = apply_story_review(store.project_dir, story_decision(waiting))
    assert approved.current_state == S.STORY_APPROVED
    assert approved.artifacts.approved_story == approved.artifacts.story == bound
    assert store.list_versions("story") == versions_before
    assert read_project(store.project_dir)[1] == approved


def test_phase4_restart_reload_then_exact_human_approval(tmp_path, monkeypatch):
    project, store = phase3_start(tmp_path)
    forbid_latest(monkeypatch)
    outcome, _ = generate(project, store)
    path, expected_story = store.project_dir, outcome.state.artifacts.story
    del outcome, store, project
    fresh_store = ArtifactStore(path)
    _, reloaded = read_project(path)
    assert reloaded.current_state == S.WAITING_STORY_APPROVAL
    assert reloaded.artifacts.story == expected_story
    assert reloaded.artifacts.approved_verification == exact_ref("verification", 2)
    assert load_story_review(reloaded, fresh_store).verification_input_ref == reloaded.artifacts.approved_verification
    apply_story_review(path, story_decision(reloaded))
    _, approved = read_project(path)
    assert approved.current_state == S.STORY_APPROVED
    assert approved.artifacts.approved_story == expected_story


def test_phase4_newer_artifacts_do_not_redefine_review_or_generation(tmp_path, monkeypatch):
    project, store = phase3_start(tmp_path)
    forbid_latest(monkeypatch)
    verification = store.load("verification", 2, VerificationPackage)
    assert store.save("verification", verification) == 3
    # Unbound prior Story publications make the actual workflow output story:v4.
    context = build_story_context(store, verification_input_ref=exact_ref("verification", 2))
    prior = StoryArchitect(context, provider=NarrativeProvider(context)).generate().package
    for _ in range(3):
        store.save("story", prior)
    outcome, contexts = generate(project, store)
    assert contexts[0].verification_input_ref == exact_ref("verification", 2)
    assert outcome.state.artifacts.story == exact_ref("story", 4)
    bound = store.load("story", 4, StoryPackage)
    newer = StoryPackage.model_validate(bound.model_dump(mode="json") | {
        "plan": bound.plan.model_dump(mode="json") | {"title": "Unbound newer blueprint"}})
    assert store.save("story", newer) == 5
    _, waiting = read_project(store.project_dir)
    target = load_story_review(waiting, store)
    assert target == bound and target != newer
    lines = story_review_lines(waiting, target)
    assert lines[0] == "Review li_bai/story:v4"
    assert "Unbound newer blueprint" not in "\n".join(lines)
    approved = apply_story_review(store.project_dir, story_decision(waiting))
    assert approved.artifacts.approved_story == exact_ref("story", 4)
    assert store.list_versions("story") == [1, 2, 3, 4, 5]


def test_phase4_corrupt_approved_version_fails_without_fallback(tmp_path, monkeypatch):
    project, store = phase3_start(tmp_path)
    forbid_latest(monkeypatch)
    _, before = read_project(store.project_dir)
    data = before.model_dump(mode="json")
    data["artifacts"]["approved_verification"]["version"] = 99
    corrupted = RuntimeState.model_validate(data)
    write_json(store.project_dir / ".runtime/state.json", corrupted, replace=True)
    def forbidden_provider(context):
        pytest.fail("Corrupt lineage must fail before any model execution")
    with pytest.raises(FileNotFoundError):
        StoryWorkflow(provider_factory=forbidden_provider).run(project, store)
    _, unchanged = read_project(store.project_dir)
    assert unchanged == corrupted and unchanged.current_state == S.FACTS_APPROVED
    assert unchanged.artifacts.story is None and store.list_versions("story") == []


def test_phase4_orphan_recovery_regenerates_from_durable_input(tmp_path, monkeypatch):
    project, store = phase3_start(tmp_path)
    forbid_latest(monkeypatch)
    _, start = read_project(store.project_dir)
    machine = ProjectStateMachine(start)
    machine.transition(S.STORY_GENERATING)
    context = build_story_context(store, verification_input_ref=start.artifacts.approved_verification)
    orphan = StoryArchitect(context, provider=NarrativeProvider(context)).generate().package
    assert orphan is not None and store.save("story", orphan) == 1
    # Represent the durable aftermath of publication without a successful binding;
    # low-level state-write fault injection is covered separately by P4-G tests.
    machine.fail("story_binding_publication_failed")
    write_json(store.project_dir / ".runtime/state.json", machine.state, replace=True)
    path = store.project_dir
    del machine, store, context
    fresh_store = ArtifactStore(path)
    project, interrupted = read_project(path)
    assert interrupted.failed_state == S.STORY_GENERATING and interrupted.artifacts.story is None
    assert interrupted.artifacts.approved_verification == exact_ref("verification", 2)
    outcome, contexts = generate(project, fresh_store)
    assert contexts[0].verification_input_ref == interrupted.artifacts.approved_verification
    assert outcome.state.artifacts.story == exact_ref("story", 2)
    assert fresh_store.load("story", 1, StoryPackage) == orphan
    approved = apply_story_review(path, story_decision(outcome.state))
    assert approved.current_state == S.STORY_APPROVED
    assert approved.artifacts.approved_story == exact_ref("story", 2)
    assert fresh_store.list_versions("story") == [1, 2]
