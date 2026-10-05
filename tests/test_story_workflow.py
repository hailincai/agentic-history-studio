import pytest

from history_studio.models import ArtifactReference, ProjectConfig, StoryPackage
from history_studio.storage.artifact_store import write_json
from history_studio.workflow import RuntimeState, ProjectState as S, WorkflowArtifactBindings, ProjectStateMachine
from history_studio.workflow.story import StoryWorkflow
from test_story_context import setup
from test_story_generation import FakeProvider, response, call
from test_story_submission import with_beats, beat, selection


def prepare(tmp_path):
    store, _, research, verification = setup(tmp_path)
    store.save("verification", verification)
    store.save("verification", verification)
    ref = ArtifactReference(project_id="test", artifact_type="verification", version=2)
    state = RuntimeState(current_state=S.FACTS_APPROVED, last_successful_state=S.FACTS_APPROVED,
        artifacts=WorkflowArtifactBindings(research=verification.research_input_ref,
            verification=ref, approved_verification=ref))
    project = ProjectConfig(project_id="test", topic=research.topic)
    write_json(store.project_dir / "project.json", project)
    write_json(store.project_dir / ".runtime/state.json", state)
    return project, store, ref


def read(store):
    return RuntimeState.model_validate_json((store.project_dir / ".runtime/state.json").read_text())


def terminal(fact_id="RF-target"):
    return response([call(with_beats([beat(fact_proposals=[selection(fact_id)])]).model_dump_json())])


def test_exact_input_success_binding_and_waiting_noop(tmp_path, monkeypatch):
    project, store, ref = prepare(tmp_path)
    contexts = []
    def factory(ctx):
        assert read(store).current_state == S.STORY_GENERATING
        contexts.append(ctx)
        return FakeProvider([terminal()])
    monkeypatch.setattr(store, "load_latest", lambda *args: pytest.fail("No latest lookup"))
    outcome = StoryWorkflow(provider_factory=factory).run(project, store)
    assert outcome.state.current_state == S.WAITING_STORY_APPROVAL
    assert contexts[0].verification_input_ref == ref
    bound = outcome.state.artifacts.story
    assert bound.version == 1 and bound.artifact_type == "story"
    assert store.load("story", bound.version, StoryPackage).verification_input_ref == ref
    assert read(store) == outcome.state
    assert outcome.state.artifacts.approved_story is None
    assert StoryWorkflow(provider_factory=factory).run(project, store).stage is None
    assert len(contexts) == 1


@pytest.mark.parametrize("failure", ["limit", "provider", "grounding", "persistence"])
def test_failure_and_regeneration_preserve_exact_input(tmp_path, monkeypatch, failure):
    project, store, ref = prepare(tmp_path)
    provider = FakeProvider([response()] if failure == "limit" else
        [RuntimeError("failure")] if failure == "provider" else
        [terminal("unknown")] if failure == "grounding" else [terminal()])
    save = store.save
    if failure == "persistence":
        monkeypatch.setattr(store, "save", lambda *args: (_ for _ in ()).throw(OSError("save failed")))
    result = StoryWorkflow(provider_factory=lambda ctx: provider).run(project, store, max_steps=1)
    assert result.state.current_state == S.FAILED and result.state.failed_state == S.STORY_GENERATING
    assert result.state.artifacts.approved_verification == ref and result.state.artifacts.story is None
    monkeypatch.setattr(store, "save", save)
    resumed = StoryWorkflow(provider_factory=lambda ctx: FakeProvider([terminal()])).run(project, store)
    assert resumed.state.current_state == S.WAITING_STORY_APPROVAL
    assert resumed.state.artifacts.approved_verification == ref


@pytest.mark.parametrize("crash", [False, True])
def test_orphan_publication_never_adopted_recovery_regenerates(tmp_path, monkeypatch, crash):
    import history_studio.workflow.story as module
    project, store, ref = prepare(tmp_path)
    write = module.write_json
    def broken(path, state, **kwargs):
        if state.current_state == S.WAITING_STORY_APPROVAL:
            assert state.artifacts.story.version == 1
            if crash:
                raise KeyboardInterrupt("crash")
            raise OSError("binding publication failed")
        return write(path, state, **kwargs)
    monkeypatch.setattr(module, "write_json", broken)
    workflow = StoryWorkflow(provider_factory=lambda ctx: FakeProvider([terminal()]))
    if crash:
        with pytest.raises(KeyboardInterrupt):
            workflow.run(project, store)
    else:
        assert workflow.run(project, store).state.current_state == S.FAILED
    assert read(store).artifacts.story is None
    assert read(store).artifacts.approved_verification == ref
    orphan = store.load("story", 1, StoryPackage)
    monkeypatch.setattr(module, "write_json", write)
    calls = []
    def regenerate(ctx):
        assert ctx.verification_input_ref == ref
        calls.append(ctx)
        return FakeProvider([terminal()])
    recovered = StoryWorkflow(provider_factory=regenerate).run(project, store)
    assert len(calls) == 1
    assert recovered.state.artifacts.story.version == 2
    assert recovered.state.current_state == S.WAITING_STORY_APPROVAL
    assert store.load("story", 1, StoryPackage) == orphan


@pytest.mark.parametrize("failure", ["missing", "type", "project", "version", "state"])
def test_invalid_input_before_generation(tmp_path, failure):
    project, store, ref = prepare(tmp_path)
    data = read(store).model_dump(mode="json")
    if failure == "missing":
        data["artifacts"]["approved_verification"] = None
    elif failure == "state":
        data["current_state"] = data["last_successful_state"] = "CREATED"
    else:
        field, value = {"type": ("artifact_type", "story"), "project": ("project_id", "foreign"),
                        "version": ("version", 99)}[failure]
        data["artifacts"]["approved_verification"][field] = value
    path = store.project_dir / ".runtime/state.json"
    import json
    path.write_text(json.dumps(data))
    with pytest.raises((ValueError, OSError)):
        StoryWorkflow(provider_factory=lambda ctx: pytest.fail("No provider")).run(project, store)


def test_replacement_invalidates_only_downstream_and_requires_active_stage():
    def ref(kind, version=1):
        return ArtifactReference(project_id="test", artifact_type=kind, version=version)
    bindings = WorkflowArtifactBindings(research=ref("research"), verification=ref("verification"),
        approved_verification=ref("verification"), story=ref("story"), approved_story=ref("story"),
        script=ref("script"), approved_script=ref("script"), storyboard=ref("storyboard"),
        approved_storyboard=ref("storyboard"))
    assert bindings.with_story(ref("story")) == bindings
    changed = bindings.with_story(ref("story", 2))
    assert changed.research == bindings.research and changed.verification == bindings.verification
    assert changed.approved_verification == bindings.approved_verification
    assert all(getattr(changed, key) is None for key in
        ("approved_story", "script", "approved_script", "storyboard", "approved_storyboard"))
    with pytest.raises(ValueError):
        ProjectStateMachine().complete_story(ref("story"), project_id="test")


def test_cli_delegates_to_same_workflow(tmp_path, monkeypatch):
    from history_studio.cli import main
    from history_studio.workflow.story import StoryWorkflowOutcome
    project, store, _ = prepare(tmp_path)
    calls = []
    def run(self, supplied_project, supplied_store):
        calls.append(supplied_project)
        machine = ProjectStateMachine(read(supplied_store))
        machine.transition(S.STORY_GENERATING)
        machine.fail("story_limit_reached")
        return StoryWorkflowOutcome(state=machine.state)
    monkeypatch.setattr(StoryWorkflow, "run", run)
    assert main(["--projects-dir", str(tmp_path), "story", "test"]) == 1
    assert calls == [project]
