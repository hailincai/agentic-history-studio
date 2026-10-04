import pytest
from pydantic import ValidationError

from history_studio.models import ArtifactReference
from history_studio.workflow import WorkflowArtifactBindings as Bindings, RuntimeState, ProjectStateMachine, ProjectState as S


def ref(kind="research", version=4, project="test"):
    return ArtifactReference(project_id=project, artifact_type=kind, version=version)


def test_empty_and_reference_only_serialization():
    assert all(value is None for value in Bindings().model_dump().values())
    bindings = Bindings(research=ref(), verification=ref("verification", 7),
                        approved_verification=ref("verification", 7))
    assert Bindings.model_validate_json(bindings.model_dump_json()) == bindings
    assert set(bindings.model_dump()["research"]) == {"project_id", "artifact_type", "version"}
    with pytest.raises(ValidationError):
        Bindings(whatever=ref())
    with pytest.raises(ValidationError):
        bindings.verification = ref("verification", 8)


@pytest.mark.parametrize("field", list(Bindings.model_fields))
def test_each_semantic_field_enforces_artifact_type(field):
    kind = field.removeprefix("approved_")
    assert getattr(Bindings(**{field: ref(kind)}), field) == ref(kind)
    with pytest.raises(ValidationError):
        Bindings(**{field: ref("wrong")})


def test_project_consistency_and_boundary_validation():
    with pytest.raises(ValidationError, match="different projects"):
        Bindings(research=ref(), verification=ref("verification", project="foreign"))
    state = RuntimeState(artifacts=Bindings(verification=ref("verification", project="foreign")))
    with pytest.raises(ValueError, match="different project"):
        state.artifacts.validate_project("test")
    machine = ProjectStateMachine(RuntimeState(current_state=S.FACT_CHECKING,
        last_successful_state=S.RESEARCH_COMPLETE, artifacts=Bindings(research=ref())))
    before = machine.state
    with pytest.raises(ValueError):
        machine.complete_verification(ref("verification", project="foreign"), project_id="test")
    assert machine.state == before


def test_legacy_structural_migration_is_read_only_and_never_guesses():
    state = RuntimeState.model_validate({"research_input_ref": ref().model_dump()})
    assert state.artifacts.research == ref() and state.artifacts.verification is None
    assert "research_input_ref" not in state.model_dump()
    assert RuntimeState.model_validate_json(state.model_dump_json()) == state
    assert RuntimeState.model_validate({}).artifacts == Bindings()
    assert RuntimeState.model_validate({"research_input_ref": None}).artifacts == Bindings()
    assert RuntimeState(research_input_ref=ref(), artifacts=Bindings(research=ref())).artifacts.research == ref()
    with pytest.raises(ValidationError, match="Conflicting"):
        RuntimeState(research_input_ref=ref(), artifacts=Bindings(research=ref(version=5)))


def test_all_bindings_survive_transition_failure_recovery():
    bindings = Bindings(research=ref(), verification=ref("verification", 7),
                        approved_verification=ref("verification", 7))
    machine = ProjectStateMachine(RuntimeState(current_state=S.FACTS_APPROVED,
        last_successful_state=S.FACTS_APPROVED, artifacts=bindings))
    machine.transition(S.STORY_GENERATING)
    machine.fail("interrupted")
    restored = ProjectStateMachine(RuntimeState.model_validate_json(machine.state.model_dump_json()))
    restored.recover()
    assert restored.state.artifacts == bindings
    restored.transition(S.WAITING_STORY_APPROVAL)
    assert restored.state.artifacts == bindings


def test_explicit_replacement_clears_dependent_lineage_without_new_rerun_transition():
    old = Bindings(research=ref(), verification=ref("verification", 7),
        approved_verification=ref("verification", 7), story=ref("story", 2))
    assert old.with_research(ref()) == old
    assert old.with_research(ref(version=5)) == Bindings(research=ref(version=5))
    assert old.with_verification(ref("verification", 8)) == Bindings(research=ref(), verification=ref("verification", 8))
    machine = ProjectStateMachine(RuntimeState(current_state=S.RESEARCHING, artifacts=old))
    machine.complete_research(ref(version=5), project_id="test")
    assert machine.state.artifacts == Bindings(research=ref(version=5))


def test_verification_gate_binds_returned_version_and_newer_artifact_does_not_replace_it(tmp_path):
    from test_fact_checking_workflow import prepare, read_state
    from test_fact_checking_runner import Dependencies
    from history_studio.workflow.fact_checking import FactCheckingWorkflow
    project, store, _ = prepare(tmp_path)
    original_save = store.save
    def interleaved(kind, package):
        version = original_save(kind, package)
        if kind == "verification" and package.is_complete:
            original_save(kind, package)
        return version
    store.save = interleaved
    outcome = FactCheckingWorkflow(Dependencies().runner()).run(project, store)
    assert outcome.stage.verification_version == 3
    assert store.list_versions("verification") == [1, 2, 3, 4]
    assert read_state(store).artifacts.verification == ref("verification", 3)
    assert read_state(store).current_state == S.WAITING_FACT_APPROVAL
    assert read_state(store).artifacts.approved_verification is None
    assert FactCheckingWorkflow(Dependencies().runner()).run(project, store).state == outcome.state
