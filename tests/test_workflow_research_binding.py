"""Exact completed research publication binding, with fake research dependencies only."""
import pytest
from pydantic import ValidationError

from history_studio.cli import read_project, show_status
from history_studio.models import ArtifactReference
from history_studio.models.research_package import ResearchPackage, ResearchRunStatus
from history_studio.research.agent import ResearchAgent
from history_studio.storage.artifact_store import write_json
from history_studio.workflow import ProjectState as S, ProjectStateMachine, RuntimeState, InvalidTransitionError
from test_research_agent import setup_run, FakeProvider, FakeTools, calls, settings, state


def reference(version=4, project_id="test", artifact_type="research"):
    return ArtifactReference(project_id=project_id, artifact_type=artifact_type, version=version)


def test_legacy_and_pre_research_states_remain_readable_but_missing_reference_fails_closed():
    assert RuntimeState().research_input_ref is None
    legacy = RuntimeState.model_validate({"current_state": "RESEARCH_COMPLETE", "last_successful_state": "RESEARCH_COMPLETE"})
    assert legacy.research_input_ref is None
    with pytest.raises(ValueError, match="binding is missing"):
        legacy.require_research_input_ref("test")


def test_binding_survives_later_transitions_failure_reload_and_recovery():
    machine = ProjectStateMachine()
    machine.transition(S.RESEARCHING)
    machine.complete_research(reference(), project_id="test")
    assert machine.state.current_state == S.RESEARCH_COMPLETE
    assert RuntimeState.model_validate_json(machine.state.model_dump_json()) == machine.state
    machine.transition(S.FACT_CHECKING)
    machine.fail("provider_request_failed")
    failed = machine.state
    assert failed.failed_state == S.FACT_CHECKING and failed.last_successful_state == S.RESEARCH_COMPLETE
    assert failed.latest_error == "provider_request_failed" and failed.research_input_ref == reference()
    restored = ProjectStateMachine(RuntimeState.model_validate_json(failed.model_dump_json()))
    restored.recover()
    assert restored.state.current_state == S.FACT_CHECKING and restored.state.research_input_ref == reference()
    assert restored.state.latest_error is None and restored.state.failed_state is None
    restored.transition(S.WAITING_FACT_APPROVAL)
    assert restored.state.research_input_ref == reference()


@pytest.mark.parametrize("ref", [reference(project_id="foreign"), reference(artifact_type="story")])
def test_completion_rejects_foreign_or_wrong_type_reference_without_changing_state(ref):
    machine = ProjectStateMachine()
    machine.transition(S.RESEARCHING)
    before = machine.state.model_dump_json()
    with pytest.raises(ValueError):
        machine.complete_research(ref, project_id="test")
    assert machine.state.model_dump_json() == before


def test_wrong_type_state_and_foreign_project_reader_rejected(tmp_path):
    with pytest.raises(ValidationError, match="research artifact"):
        RuntimeState(research_input_ref=reference(artifact_type="story"))
    project, store = setup_run(tmp_path)
    foreign = RuntimeState(research_input_ref=reference(project_id="foreign"))
    write_json(store.project_dir / ".runtime/state.json", foreign, replace=True)
    with pytest.raises(ValueError, match="different project"):
        read_project(store.project_dir)
    provider = FakeProvider([])
    with pytest.raises(ValueError, match="different project"):
        ResearchAgent(provider, FakeTools(), settings()).run(project, store)
    assert provider.calls == 0


@pytest.mark.parametrize("state_value", [S.CREATED, S.RESEARCH_COMPLETE, S.FACT_CHECKING, S.WAITING_FACT_APPROVAL])
def test_binding_cannot_be_replaced_outside_research_completion(state_value):
    successful = state_value if state_value != S.FACT_CHECKING else S.RESEARCH_COMPLETE
    machine = ProjectStateMachine(RuntimeState(current_state=state_value, last_successful_state=successful,
                                               research_input_ref=reference()))
    before = machine.state
    with pytest.raises(InvalidTransitionError):
        machine.complete_research(reference(5), project_id="test")
    assert machine.state == before


def test_replacement_is_explicit_at_active_research_completion_only():
    # No transition from downstream to research is introduced. This models an explicitly
    # active research stage; a future legitimate rerun policy must still supply publication identity.
    machine = ProjectStateMachine(RuntimeState(current_state=S.RESEARCHING,
        last_successful_state=S.CREATED, research_input_ref=reference()))
    machine.complete_research(reference(5), project_id="test")
    assert machine.state.research_input_ref == reference(5)


def test_completion_binds_returned_v4_even_when_v5_is_published_before_state_write(tmp_path, monkeypatch):
    project, store = setup_run(tmp_path)
    save = store.save
    published = []
    def interleaved(artifact_type, package):
        version = save(artifact_type, package)
        if artifact_type == "research" and version == 1:
            save("research", package)
            save("research", package)
        if artifact_type == "research" and package.progress.status == ResearchRunStatus.COMPLETE:
            published.append(version)
            save("research", package)
        return version
    monkeypatch.setattr(store, "save", interleaved)
    provider = FakeProvider(calls())
    result = ResearchAgent(provider, FakeTools(), settings()).run(project, store)
    assert result.progress.status == ResearchRunStatus.COMPLETE
    assert published == [4] and store.list_versions("research") == [1, 2, 3, 4, 5]
    assert state(store).research_input_ref == reference(4)
    assert state(store).current_state == S.RESEARCH_COMPLETE
    assert store.load("research", 4, ResearchPackage) == result
    assert state(store).require_research_input_ref(project.project_id) == reference(4)


def test_bound_completed_reentry_ignores_newer_artifacts_and_does_not_call_provider(tmp_path, monkeypatch, capsys):
    project, store = setup_run(tmp_path)
    result = ResearchAgent(FakeProvider(calls()), FakeTools(), settings()).run(project, store)
    bound = state(store).research_input_ref
    newer = result.model_copy(deep=True)
    newer.progress.status = ResearchRunStatus.FAILED
    store.save("research", newer)
    store.save("notes", RuntimeState())
    def no_latest(*args):
        raise AssertionError("Bound research must never use latest discovery")
    monkeypatch.setattr(ResearchAgent, "_load_package", no_latest)
    provider = FakeProvider([])
    loaded = ResearchAgent(provider, FakeTools(), settings()).run(project, store)
    assert loaded == result and provider.calls == 0
    assert state(store).research_input_ref == bound
    config, restored = read_project(store.project_dir)
    show_status(store.project_dir, config, restored)
    assert f"test/research:v{bound.version}" in capsys.readouterr().out


def test_failed_artifact_publication_never_creates_binding(tmp_path, monkeypatch):
    project, store = setup_run(tmp_path)
    save = store.save
    def fail_completion(artifact_type, package):
        if artifact_type == "research" and package.progress.status == ResearchRunStatus.COMPLETE:
            raise OSError("Publication refused")
        return save(artifact_type, package)
    monkeypatch.setattr(store, "save", fail_completion)
    result = ResearchAgent(FakeProvider(calls()), FakeTools(), settings()).run(project, store)
    assert result.progress.status == ResearchRunStatus.FAILED
    assert state(store).current_state == S.FAILED and state(store).failed_state == S.RESEARCHING
    assert state(store).research_input_ref is None
    assert all(store.load("research", v, ResearchPackage).progress.status != ResearchRunStatus.COMPLETE
               for v in store.list_versions("research"))


@pytest.mark.parametrize("workflow_state", [S.RESEARCHING, S.RESEARCH_COMPLETE])
def test_unbound_completed_checkpoint_cannot_be_guessed_into_legacy_state(tmp_path, workflow_state):
    project, store = setup_run(tmp_path)
    result = ResearchAgent(FakeProvider(calls()), FakeTools(), settings()).run(project, store)
    write_json(store.project_dir / ".runtime/state.json", RuntimeState(current_state=workflow_state,
        last_successful_state=S.CREATED if workflow_state == S.RESEARCHING else S.RESEARCH_COMPLETE), replace=True)
    provider = FakeProvider([])
    with pytest.raises(ValueError, match="binding is missing"):
        ResearchAgent(provider, FakeTools(), settings()).run(project, store)
    assert provider.calls == 0 and state(store).research_input_ref is None
    assert result.progress.status == ResearchRunStatus.COMPLETE


def test_model_cannot_supply_snapshot_version(tmp_path):
    from test_research_agent import update
    from history_studio.research.actions import ResearchSelectionUpdate
    data = update().arguments | {"research_input_ref": reference(99).model_dump()}
    with pytest.raises(ValidationError):
        ResearchSelectionUpdate.model_validate(data)


def test_downstream_research_rerun_is_still_forbidden(tmp_path):
    project, store = setup_run(tmp_path)
    ResearchAgent(FakeProvider(calls()), FakeTools(), settings()).run(project, store)
    machine = ProjectStateMachine(state(store))
    machine.transition(S.FACT_CHECKING)
    write_json(store.project_dir / ".runtime/state.json", machine.state, replace=True)
    before = state(store)
    with pytest.raises(ValueError, match="current workflow stage"):
        ResearchAgent(FakeProvider([]), FakeTools(), settings()).run(project, store)
    assert state(store) == before
