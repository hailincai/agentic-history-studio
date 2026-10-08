import json

import pytest

from history_studio.models import ArtifactReference, ProjectConfig, ScriptPackage, StoryboardPackage
from history_studio.storage.artifact_store import ArtifactStore, write_json
from history_studio.workflow import RuntimeState, ProjectState as S, WorkflowArtifactBindings, ProjectStateMachine
from history_studio.workflow.storyboard import StoryboardWorkflow, StoryboardWorkflowOutcome
from history_studio.visual_director import StoryboardIntegrityReport, StoryboardIntegrityIssue
from test_visual_director_preparation import context
from test_visual_director_generation import FakeProvider, response, call


def prepare(tmp_path):
    store = ArtifactStore(tmp_path / "project")
    data = context().model_dump(mode="json")
    data.pop("script_input_ref")
    script = ScriptPackage(**data, story_input_ref=ArtifactReference(
        project_id="project", artifact_type="story", version=99))
    assert store.save("script", script) == 1
    newer = script.model_copy(deep=True)
    newer.title = "Newer unapproved Script"
    store.save("script", newer)
    ref = ArtifactReference(project_id="project", artifact_type="script", version=1)
    state = RuntimeState(current_state=S.SCRIPT_APPROVED, last_successful_state=S.SCRIPT_APPROVED,
        artifacts=WorkflowArtifactBindings(script=ref.model_copy(update={"version": 2}), approved_script=ref,
            approved_story=ArtifactReference(project_id="project", artifact_type="story", version=99),
            approved_verification=ArtifactReference(project_id="project", artifact_type="verification", version=42)))
    project = ProjectConfig(project_id="project", topic="Test")
    write_json(store.project_dir / "project.json", project)
    write_json(store.project_dir / ".runtime/state.json", state)
    return project, store, ref


def read(store):
    return RuntimeState.model_validate_json((store.project_dir / ".runtime/state.json").read_text())


def workflow():
    return StoryboardWorkflow(provider_factory=lambda ctx: FakeProvider([response([call()])]))


def test_success_exact_authority_validate_persist_reload_bind_and_waiting_noop(tmp_path, monkeypatch):
    import history_studio.workflow.storyboard as module
    project, store, ref = prepare(tmp_path)
    upstream = read(store).artifacts
    contexts, events = [], []
    validate, save, load = module.validate_storyboard_integrity, store.save, store.load

    def check(ctx, package):
        events.append("validate")
        return validate(ctx, package)

    def publish(kind, package):
        assert events[-1] == "validate" and kind == "storyboard"
        events.append("save")
        return save(kind, package)

    def exact_load(kind, version, model):
        assert kind in ("script", "storyboard")
        if kind == "script":
            assert version == 1
        else:
            assert version == 1
            events.append("reload")
        return load(kind, version, model)

    def factory(ctx):
        assert read(store).current_state == S.STORYBOARD_GENERATING
        assert ctx.script_input_ref == ref and ctx.title == "Approved title"
        contexts.append(ctx)
        ctx.title = "Factory copy mutation"
        return FakeProvider([response([call()])])

    monkeypatch.setattr(module, "validate_storyboard_integrity", check)
    monkeypatch.setattr(store, "save", publish)
    monkeypatch.setattr(store, "load", exact_load)
    monkeypatch.setattr(store, "load_latest", lambda *args: pytest.fail("No latest lookup"))
    runner = StoryboardWorkflow(provider_factory=factory)
    outcome = runner.run(project, store)
    assert events == ["validate", "save", "reload", "validate"]
    assert outcome.state.current_state == S.WAITING_STORYBOARD_APPROVAL
    bound = outcome.state.artifacts.storyboard
    assert bound == ArtifactReference(project_id="project", artifact_type="storyboard", version=1)
    assert outcome.state.artifacts.approved_storyboard is None
    for field in ("script", "approved_script", "approved_story", "approved_verification"):
        assert getattr(outcome.state.artifacts, field) == getattr(upstream, field)
    assert read(store) == outcome.state
    assert load("storyboard", 1, StoryboardPackage).title == "Approved title"
    assert runner.run(project, store).stage is None and len(contexts) == 1


@pytest.mark.parametrize("failure", ["limit", "provider", "finalization", "validation", "save", "reload"])
def test_failure_and_retry_preserve_exact_input(tmp_path, monkeypatch, failure):
    import history_studio.workflow.storyboard as module
    from test_storyboard_submission import submission
    project, store, ref = prepare(tmp_path)
    proposal = submission().model_dump(mode="json")
    proposal["sections"][0]["shots"].pop()
    provider = FakeProvider([response()] if failure == "limit" else
        [RuntimeError("provider failed")] if failure == "provider" else
        [response([call(json.dumps(proposal))])] if failure == "finalization" else [response([call()])])
    save, load, validate = store.save, store.load, module.validate_storyboard_integrity
    if failure == "validation":
        monkeypatch.setattr(module, "validate_storyboard_integrity", lambda *args: StoryboardIntegrityReport(
            issues=[StoryboardIntegrityIssue(code="SEGMENT_UNCOVERED", message="Missing", source_segment_id="a")]))
    if failure == "save":
        monkeypatch.setattr(store, "save", lambda *args: (_ for _ in ()).throw(OSError("save failed")))
    if failure == "reload":
        def broken_load(kind, version, model):
            if kind == "storyboard":
                raise OSError("reload failed")
            return load(kind, version, model)
        monkeypatch.setattr(store, "load", broken_load)
    outcome = StoryboardWorkflow(provider_factory=lambda ctx: provider).run(project, store, max_steps=1)
    assert outcome.state.current_state == S.FAILED and outcome.state.failed_state == S.STORYBOARD_GENERATING
    assert read(store) == outcome.state
    assert outcome.state.artifacts.approved_script == ref and outcome.state.artifacts.storyboard is None
    assert store.list_versions("storyboard") == ([1] if failure == "reload" else [])
    if failure == "validation":
        assert not outcome.validation_report.is_valid
    monkeypatch.setattr(store, "save", save)
    monkeypatch.setattr(store, "load", load)
    monkeypatch.setattr(module, "validate_storyboard_integrity", validate)
    resumed = workflow().run(project, store)
    assert resumed.state.current_state == S.WAITING_STORYBOARD_APPROVAL
    assert resumed.state.artifacts.approved_script == ref
    assert resumed.state.artifacts.storyboard.version == (2 if failure == "reload" else 1)


@pytest.mark.parametrize("crash", [False, True])
def test_orphan_after_publication_never_adopted(tmp_path, monkeypatch, crash):
    import history_studio.workflow.storyboard as module
    project, store, ref = prepare(tmp_path)
    write = module.write_json

    def broken(path, state, **kwargs):
        if state.current_state == S.WAITING_STORYBOARD_APPROVAL:
            if crash:
                raise KeyboardInterrupt("crash")
            raise OSError("publication failed")
        return write(path, state, **kwargs)

    monkeypatch.setattr(module, "write_json", broken)
    if crash:
        with pytest.raises(KeyboardInterrupt):
            workflow().run(project, store)
    else:
        assert workflow().run(project, store).state.current_state == S.FAILED
    orphan = store.load("storyboard", 1, StoryboardPackage)
    assert read(store).artifacts.storyboard is None and read(store).artifacts.approved_script == ref
    monkeypatch.setattr(module, "write_json", write)
    assert workflow().run(project, store).state.artifacts.storyboard.version == 2
    assert store.load("storyboard", 1, StoryboardPackage) == orphan


@pytest.mark.parametrize("failure", ["missing", "type", "project", "version", "state", "failed_state"])
def test_bad_authority_fails_before_provider(tmp_path, failure):
    project, store, _ = prepare(tmp_path)
    data = read(store).model_dump(mode="json")
    if failure == "missing":
        data["artifacts"]["approved_script"] = None  # candidate script remains present
    elif failure == "state":
        data["current_state"] = data["last_successful_state"] = "CREATED"
    elif failure == "failed_state":
        data.update(current_state="FAILED", failed_state="SCRIPT_GENERATING", latest_error="Failed")
    else:
        key, value = {"type": ("artifact_type", "story"), "project": ("project_id", "foreign"),
                      "version": ("version", 99)}[failure]
        data["artifacts"]["approved_script"][key] = value
    (store.project_dir / ".runtime/state.json").write_text(json.dumps(data))
    with pytest.raises((ValueError, OSError)):
        StoryboardWorkflow(provider_factory=lambda ctx: pytest.fail("No provider")).run(project, store)


def test_new_candidate_invalidates_only_downstream_and_requires_active_stage():
    def ref(kind, version=1):
        return ArtifactReference(project_id="project", artifact_type=kind, version=version)
    bindings = WorkflowArtifactBindings(**{key: ref(key.removeprefix("approved_"))
        for key in WorkflowArtifactBindings.model_fields if key != "assembly"})
    machine = ProjectStateMachine(RuntimeState(current_state=S.STORYBOARD_GENERATING,
        last_successful_state=S.SCRIPT_APPROVED, artifacts=bindings))
    changed = machine.complete_storyboard(ref("storyboard", 2), project_id="project").artifacts
    assert changed.storyboard.version == 2 and changed.approved_storyboard is None
    assert changed.media is None and changed.assembly is None
    for key in ("research", "verification", "approved_verification", "story", "approved_story", "script", "approved_script"):
        assert getattr(changed, key) == getattr(bindings, key)
    with pytest.raises(ValueError):
        ProjectStateMachine().complete_storyboard(ref("storyboard"), project_id="project")


def test_cli_delegates_without_client_and_accepts_config(tmp_path, monkeypatch):
    from history_studio.cli import main
    from history_studio.research.openai_provider import create_client
    import history_studio.research.openai_provider as provider_module
    project, store, _ = prepare(tmp_path)
    calls = []

    def forbidden(*args, **kwargs):
        pytest.fail("No paid API client")

    def run(self, supplied_project, supplied_store):
        calls.append(supplied_project)
        machine = ProjectStateMachine(read(supplied_store))
        machine.transition(S.STORYBOARD_GENERATING)
        machine.fail("storyboard_limit_reached")
        return StoryboardWorkflowOutcome(state=machine.state)

    monkeypatch.setattr(provider_module, "create_client", forbidden)
    monkeypatch.setattr(StoryboardWorkflow, "run", run)
    config = store.project_dir / "config.json"
    config.write_text("{}")
    assert main(["--projects-dir", str(tmp_path), "storyboard", "project", "--config", str(config)]) == 1
    assert calls == [project]
