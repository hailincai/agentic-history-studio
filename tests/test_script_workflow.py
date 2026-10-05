import json

import pytest

from history_studio.models import ArtifactReference, ProjectConfig, ScriptPackage
from history_studio.storage.artifact_store import write_json
from history_studio.workflow import RuntimeState, ProjectState as S, WorkflowArtifactBindings, ProjectStateMachine
from history_studio.workflow.script import ScriptWorkflow
from test_script_context import setup
from test_script_generation import FakeProvider, response, call
from test_script_submission import submission, section, segment


def prepare(tmp_path):
    store, ref, story = setup(tmp_path)
    newer = story.model_copy(deep=True)
    newer.plan.title = "Newer unapproved Story"
    store.save("story", newer)
    state = RuntimeState(current_state=S.STORY_APPROVED, last_successful_state=S.STORY_APPROVED,
        artifacts=WorkflowArtifactBindings(story=ref.model_copy(update={"version": 2}), approved_story=ref))
    project = ProjectConfig(project_id="project", topic="Test topic")
    write_json(store.project_dir / "project.json", project)
    write_json(store.project_dir / ".runtime/state.json", state)
    return project, store, ref


def read(store):
    return RuntimeState.model_validate_json((store.project_dir / ".runtime/state.json").read_text())


def terminal(fact_id="f1"):
    proposal = submission(sections=[section("s1", [segment(grounding=dict(
        story_beat_id="b1", research_fact_ids=[fact_id]))])])
    return response([call(proposal.model_dump_json())])


def test_success_exact_authority_validation_and_waiting_noop(tmp_path, monkeypatch):
    import history_studio.workflow.script as module
    project, store, ref = prepare(tmp_path)
    contexts, validations = [], []
    validator = module.validate_script_grounding

    def check(context, package):
        validations.append(package)
        return validator(context, package)

    def factory(ctx):
        assert read(store).current_state == S.SCRIPT_GENERATING
        assert ctx.title == "Approved title" and ctx.story_input_ref == ref
        contexts.append(ctx)
        return FakeProvider([terminal()])

    monkeypatch.setattr(module, "validate_script_grounding", check)
    monkeypatch.setattr(store, "load_latest", lambda *args: pytest.fail("No latest lookup"))
    load = store.load

    def bounded_load(kind, version, contract):
        assert kind in ("story", "script")
        return load(kind, version, contract)

    monkeypatch.setattr(store, "load", bounded_load)
    outcome = ScriptWorkflow(provider_factory=factory).run(project, store)
    assert len(validations) == 2
    assert outcome.state.current_state == S.WAITING_SCRIPT_APPROVAL
    bound = outcome.state.artifacts.script
    assert bound.version == 1 and bound.artifact_type == "script"
    assert store.load("script", bound.version, ScriptPackage).story_input_ref == ref
    assert outcome.state.artifacts.approved_story == ref
    assert outcome.state.artifacts.story.version == 2
    assert outcome.state.artifacts.approved_script is None
    assert read(store) == outcome.state
    assert ScriptWorkflow(provider_factory=factory).run(project, store).stage is None
    assert len(contexts) == 1


@pytest.mark.parametrize("failure", ["limit", "provider", "grounding", "persistence", "validation"])
def test_failures_preserve_input_and_recovery_regenerates(tmp_path, monkeypatch, failure):
    import history_studio.workflow.script as module
    from history_studio.script import ScriptGroundingValidationReport, ScriptGroundingValidationIssue
    project, store, ref = prepare(tmp_path)
    provider = FakeProvider([response()] if failure == "limit" else
        [RuntimeError("failure")] if failure == "provider" else
        [terminal("unknown")] if failure == "grounding" else [terminal()])
    save, validate = store.save, module.validate_script_grounding
    if failure == "persistence":
        monkeypatch.setattr(store, "save", lambda *args: (_ for _ in ()).throw(OSError("save failed")))
    if failure == "validation":
        monkeypatch.setattr(module, "validate_script_grounding", lambda *args: ScriptGroundingValidationReport(
            issues=[ScriptGroundingValidationIssue(code="UNKNOWN_FACT", message="Unknown fact", research_fact_id="bad")]))
    result = ScriptWorkflow(provider_factory=lambda ctx: provider).run(project, store, max_steps=1)
    assert result.state.current_state == S.FAILED and result.state.failed_state == S.SCRIPT_GENERATING
    assert result.state.artifacts.approved_story == ref and result.state.artifacts.script is None
    assert store.list_versions("script") == []
    if failure == "validation":
        assert result.validation_report.issues[0].code == "UNKNOWN_FACT"
    monkeypatch.setattr(store, "save", save)
    monkeypatch.setattr(module, "validate_script_grounding", validate)
    resumed = ScriptWorkflow(provider_factory=lambda ctx: FakeProvider([terminal()])).run(project, store)
    assert resumed.state.current_state == S.WAITING_SCRIPT_APPROVAL


@pytest.mark.parametrize("crash", [False, True])
def test_orphan_never_adopted_after_failed_publication(tmp_path, monkeypatch, crash):
    import history_studio.workflow.script as module
    project, store, ref = prepare(tmp_path)
    write = module.write_json

    def broken(path, state, **kwargs):
        if state.current_state == S.WAITING_SCRIPT_APPROVAL:
            if crash:
                raise KeyboardInterrupt("crash")
            raise OSError("publication failed")
        return write(path, state, **kwargs)

    monkeypatch.setattr(module, "write_json", broken)
    workflow = ScriptWorkflow(provider_factory=lambda ctx: FakeProvider([terminal()]))
    if crash:
        with pytest.raises(KeyboardInterrupt):
            workflow.run(project, store)
    else:
        assert workflow.run(project, store).state.current_state == S.FAILED
    assert read(store).artifacts.script is None and read(store).artifacts.approved_story == ref
    orphan = store.load("script", 1, ScriptPackage)
    monkeypatch.setattr(module, "write_json", write)
    recovered = workflow.run(project, store)
    assert recovered.state.artifacts.script.version == 2
    assert store.load("script", 1, ScriptPackage) == orphan


@pytest.mark.parametrize("failure", ["missing", "type", "project", "version", "state"])
def test_invalid_input_fails_before_generation(tmp_path, failure):
    project, store, _ = prepare(tmp_path)
    data = read(store).model_dump(mode="json")
    if failure == "missing":
        data["artifacts"]["approved_story"] = None
    elif failure == "state":
        data["current_state"] = data["last_successful_state"] = "CREATED"
    else:
        key, value = {"type": ("artifact_type", "research"), "project": ("project_id", "foreign"),
                      "version": ("version", 99)}[failure]
        data["artifacts"]["approved_story"][key] = value
    (store.project_dir / ".runtime/state.json").write_text(json.dumps(data))
    with pytest.raises((ValueError, OSError)):
        ScriptWorkflow(provider_factory=lambda ctx: pytest.fail("No provider")).run(project, store)


def test_new_output_clears_only_downstream_and_requires_active_stage():
    def ref(kind, version=1):
        return ArtifactReference(project_id="project", artifact_type=kind, version=version)
    bindings = WorkflowArtifactBindings(**{key: ref(key.removeprefix("approved_"))
        for key in WorkflowArtifactBindings.model_fields})
    machine = ProjectStateMachine(RuntimeState(current_state=S.SCRIPT_GENERATING,
        last_successful_state=S.STORY_APPROVED, artifacts=bindings))
    changed = machine.complete_script(ref("script", 2), project_id="project").artifacts
    for key in ("research", "verification", "approved_verification", "story", "approved_story"):
        assert getattr(changed, key) == getattr(bindings, key)
    assert all(getattr(changed, key) is None for key in ("approved_script", "storyboard", "approved_storyboard"))
    with pytest.raises(ValueError):
        ProjectStateMachine().complete_script(ref("script"), project_id="project")


def test_cli_delegates_workflow_and_lazy_client(tmp_path, monkeypatch):
    from history_studio.cli import main
    from history_studio.workflow.script import ScriptWorkflowOutcome
    project, store, _ = prepare(tmp_path)
    calls = []

    def run(self, supplied_project, supplied_store):
        calls.append(supplied_project)
        machine = ProjectStateMachine(read(supplied_store))
        machine.transition(S.SCRIPT_GENERATING)
        machine.fail("script_limit_reached")
        return ScriptWorkflowOutcome(state=machine.state)

    monkeypatch.setattr(ScriptWorkflow, "run", run)
    assert main(["--projects-dir", str(tmp_path), "script", "project"]) == 1
    assert calls == [project]
