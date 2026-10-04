"""Workflow integration using actual runner/checker and injected fake dependencies."""
import json

import pytest

from history_studio.cli import main
from history_studio.models import ProjectConfig, VerificationPackage, create_verification_package, add_verification_result
from history_studio.models.research_package import ResearchPackage
from history_studio.storage.artifact_store import ArtifactStore, write_json
from history_studio.verification import FactCheckingOutcome, FactCheckingStopReason
from history_studio.workflow import RuntimeState, ProjectState as S, ProjectStateMachine, InvalidTransitionError
from history_studio.workflow.fact_checking import FactCheckingWorkflow
from test_fact_checking_runner import Dependencies, IDS
from test_fact_checker_dispatch import native
from test_fact_checker_investigation import decision
from test_verification_context import research_ref
from test_verification_package import research_data, result_for
from test_verification_submission import payload, terminal


def prepare(tmp_path, current=S.RESEARCH_COMPLETE, *, empty=False, bound=True):
    research = research_data()
    from history_studio.models.research_package import ResearchPlan
    research.plan = ResearchPlan(gaps=[dict(gap_id="G1", question="What do these records establish?",
        completion_criteria=["Assess the dated records"], status="COVERED",
        fact_ids=IDS, coverage_assessment=dict(criteria=[dict(criterion="Assess the dated records", addressed=True)],
            supporting_fact_ids=IDS, rationale="The records were researched", unresolved_issues=[]))])
    research.progress.status = "COMPLETE"
    research = ResearchPackage.model_validate(research.model_dump())
    if empty:
        # Deliberately bypass validation only to test the workflow's defensive artifact load.
        research = research.model_copy(update={"facts": [], "plan": ResearchPlan()})
    project = ProjectConfig(project_id="test", topic=research.topic, research_scope=research.research_scope)
    store = ArtifactStore(tmp_path / "test")
    for _ in range(4):
        store.save("research", research)
    write_json(store.project_dir / "project.json", project)
    state = RuntimeState(current_state=current, last_successful_state=(current if current in
        (S.CREATED, S.RESEARCH_COMPLETE, S.WAITING_FACT_APPROVAL, S.FACTS_APPROVED) else S.RESEARCH_COMPLETE),
        research_input_ref=research_ref(4) if bound else None)
    write_json(store.project_dir / ".runtime/state.json", state)
    return project, store, research


def read_state(store):
    return RuntimeState.model_validate_json((store.project_dir / ".runtime/state.json").read_text(encoding="utf-8"))


def test_normal_completion_enters_stage_before_runner_and_advances_only_after_durable_completion(tmp_path):
    project, store, _ = prepare(tmp_path)
    deps = Dependencies()
    real = deps.runner()
    class ObservedRunner:
        def run(self, store, **kwargs):
            assert read_state(store).current_state == S.FACT_CHECKING
            assert kwargs["research_input_ref"] == research_ref(4)
            return real.run(store, **kwargs)
    outcome = FactCheckingWorkflow(ObservedRunner()).run(project, store)
    assert outcome.state.current_state == S.WAITING_FACT_APPROVAL
    assert outcome.state.last_successful_state == S.WAITING_FACT_APPROVAL
    assert outcome.state.research_input_ref == research_ref(4)
    assert outcome.stage.completed_count == 3 and outcome.stage.package.is_complete
    assert outcome.stage.package == store.load("verification", 3, VerificationPackage)
    assert read_state(store) == outcome.state
    assert [c.target_fact.fact_id for c in deps.contexts] == IDS
    assert set(outcome.stage.package.model_dump()) == {"schema_version", "research_input_ref", "research_facts", "results"}
    assert "completed_fact_ids" not in outcome.state.model_dump()
    assert "current_state" not in outcome.stage.package.model_dump()
    assert not store.list_versions("approvals") and not store.list_versions("story")


@pytest.mark.parametrize("current", [S.CREATED, S.RESEARCHING, S.FACTS_APPROVED, S.STORY_GENERATING])
def test_wrong_entry_state_rejected_before_provider_or_state_changes(tmp_path, current):
    project, store, _ = prepare(tmp_path, current)
    before = read_state(store)
    deps = Dependencies()
    with pytest.raises(InvalidTransitionError):
        FactCheckingWorkflow(deps.runner()).run(project, store)
    assert read_state(store) == before and deps.contexts == []


@pytest.mark.parametrize("reason", ["limit", "text", "none", "provider", "protocol", "finalize", "tool"])
def test_incomplete_or_failed_stage_uses_failed_interrupted_state_and_keeps_checkpoints(tmp_path, reason):
    project, store, _ = prepare(tmp_path)
    bad = payload("VERIFIED", source_id="SRC-A")
    replies = {
        "limit": [decision(native())], "text": [decision(text="ordinary text")], "none": [decision()],
        "provider": [RuntimeError("secret-raw-provider-message")],
        "protocol": [decision(native("submit_verification", "{bad"))],
        "finalize": [decision(terminal(bad))], "tool": [decision(native())],
    }
    deps = Dependencies({IDS[1]: replies[reason]})
    if reason == "tool":
        original = deps.tool
        def broken(context):
            tools = original(context)
            if context.target_fact.fact_id == IDS[1]:
                def fail(query):
                    raise OSError("secret-raw-tool-message")
                tools.search_web = fail
            return tools
        deps.tool = broken
    outcome = FactCheckingWorkflow(deps.runner()).run(project, store, max_steps=1)
    assert outcome.state.current_state == S.FAILED
    assert outcome.state.failed_state == S.FACT_CHECKING
    assert outcome.state.last_successful_state == S.RESEARCH_COMPLETE
    assert outcome.state.research_input_ref == research_ref(4)
    assert outcome.stage.completed_count == 1 and not outcome.stage.package.is_complete
    assert outcome.stage.package.completed_fact_ids == IDS[:1]
    assert outcome.stage.package.pending_fact_ids == IDS[1:]
    assert store.list_versions("verification") == [1]
    assert len(outcome.stage.package.results) == 1  # No synthetic result for the interrupted fact.
    assert "secret-raw" not in outcome.model_dump_json()
    before = (store.project_dir / "verification/verification_v1.json").read_bytes()
    fresh = Dependencies()
    resumed = FactCheckingWorkflow(fresh.runner()).run(project, store)
    assert resumed.state.current_state == S.WAITING_FACT_APPROVAL and resumed.stage.completed_count == 2
    assert [c.target_fact.fact_id for c in fresh.contexts] == IDS[1:]
    assert (store.project_dir / "verification/verification_v1.json").read_bytes() == before


def test_checkpoint_publication_failure_records_fact_checking_failure(tmp_path, monkeypatch):
    project, store, _ = prepare(tmp_path)
    save = store.save
    def fail(kind, package):
        if kind == "verification":
            raise OSError("checkpoint failure")
        return save(kind, package)
    monkeypatch.setattr(store, "save", fail)
    outcome = FactCheckingWorkflow(Dependencies().runner()).run(project, store)
    assert outcome.state.failed_state == S.FACT_CHECKING and outcome.state.current_state == S.FAILED
    assert outcome.stage.completed_count == 0 and outcome.stage.package.results == ()
    assert not store.list_versions("verification")


def test_crash_after_final_checkpoint_recovers_without_reverification(tmp_path, monkeypatch):
    import history_studio.workflow.fact_checking as module
    project, store, _ = prepare(tmp_path)
    write = module.write_json
    def crash(path, state, **kwargs):
        if state.current_state == S.WAITING_FACT_APPROVAL:
            raise KeyboardInterrupt("crash before final state publication")
        return write(path, state, **kwargs)
    monkeypatch.setattr(module, "write_json", crash)
    with pytest.raises(KeyboardInterrupt):
        FactCheckingWorkflow(Dependencies().runner()).run(project, store)
    assert read_state(store).current_state == S.FACT_CHECKING
    assert store.load_latest("verification", VerificationPackage).is_complete
    monkeypatch.setattr(module, "write_json", write)
    deps = Dependencies()
    recovered = FactCheckingWorkflow(deps.runner()).run(project, store)
    assert recovered.state.current_state == S.WAITING_FACT_APPROVAL
    assert recovered.stage.completed_count == 0 and deps.contexts == [] and deps.tools == []
    assert store.list_versions("verification") == [1, 2, 3]


def test_final_state_write_failure_preserves_interrupted_stage_and_recovers(tmp_path, monkeypatch):
    import history_studio.workflow.fact_checking as module
    project, store, _ = prepare(tmp_path)
    write = module.write_json
    def fail(path, state, **kwargs):
        if state.current_state == S.WAITING_FACT_APPROVAL:
            raise OSError("state publication failure")
        return write(path, state, **kwargs)
    monkeypatch.setattr(module, "write_json", fail)
    outcome = FactCheckingWorkflow(Dependencies().runner()).run(project, store)
    assert outcome.state.failed_state == S.FACT_CHECKING and outcome.state.current_state == S.FAILED
    monkeypatch.setattr(module, "write_json", write)
    deps = Dependencies()
    resumed = FactCheckingWorkflow(deps.runner()).run(project, store)
    assert resumed.state.current_state == S.WAITING_FACT_APPROVAL and deps.contexts == []


def test_failed_other_stage_cannot_resume_verification(tmp_path):
    project, store, _ = prepare(tmp_path)
    write_json(store.project_dir / ".runtime/state.json", RuntimeState(current_state=S.FAILED,
        failed_state=S.RESEARCHING, last_successful_state=S.CREATED, latest_error="interrupted",
        research_input_ref=research_ref(4)), replace=True)
    deps = Dependencies()
    with pytest.raises(InvalidTransitionError):
        FactCheckingWorkflow(deps.runner()).run(project, store)
    assert deps.contexts == [] and read_state(store).failed_state == S.RESEARCHING


def test_waiting_gate_is_idempotent(tmp_path):
    project, store, _ = prepare(tmp_path, S.WAITING_FACT_APPROVAL)
    path = store.project_dir / ".runtime/state.json"
    before = path.read_bytes()
    deps = Dependencies()
    outcome = FactCheckingWorkflow(deps.runner()).run(project, store)
    assert outcome.stage is None and outcome.state.current_state == S.WAITING_FACT_APPROVAL
    assert deps.contexts == [] and path.read_bytes() == before


@pytest.mark.parametrize("current", [S.RESEARCH_COMPLETE, S.FACT_CHECKING])
def test_legacy_missing_reference_fails_closed_without_latest_discovery(tmp_path, current, monkeypatch):
    project, store, _ = prepare(tmp_path, current, bound=False)
    def forbidden(*args):
        raise AssertionError("No research latest discovery")
    monkeypatch.setattr(store, "load_latest", forbidden)
    deps = Dependencies()
    with pytest.raises(ValueError, match="binding is missing"):
        FactCheckingWorkflow(deps.runner()).run(project, store)
    assert deps.contexts == [] and read_state(store).current_state == current


def test_newer_research_never_replaces_bound_input_on_resume(tmp_path, monkeypatch):
    project, store, research = prepare(tmp_path)
    deps = Dependencies({IDS[1]: [RuntimeError("stop")]})
    FactCheckingWorkflow(deps.runner()).run(project, store)
    newer = research.model_copy(deep=True)
    newer.facts[0].claim = "Newer research differs."
    assert store.save("research", newer) == 5
    def no_latest(*args):
        raise AssertionError("No research latest discovery")
    monkeypatch.setattr(store, "load_latest", no_latest)
    fresh = Dependencies()
    outcome = FactCheckingWorkflow(fresh.runner()).run(project, store)
    assert outcome.state.research_input_ref == research_ref(4)
    assert outcome.stage.package.research_input_ref == research_ref(4)
    assert [c.target_fact.fact_id for c in fresh.contexts] == IDS[1:]
    assert outcome.stage.package.research_facts[0].claim_snapshot == research.facts[0].claim


def test_complete_nondurable_nonempty_result_cannot_advance_workflow(tmp_path):
    project, store, research = prepare(tmp_path)
    package = create_verification_package(research, research_input_ref=research_ref(4))
    for index in range(3):
        package = add_verification_result(package, result_for(package, index))
    class UnsafeRunner:
        def run(self, *args, **kwargs):
            return FactCheckingOutcome(package=package, stop_reason="COMPLETE")
    outcome = FactCheckingWorkflow(UnsafeRunner()).run(project, store)
    assert outcome.state.current_state == S.FAILED and not store.list_versions("verification")


def test_empty_research_cannot_enter_verification_as_completed_research(tmp_path):
    project, store, _ = prepare(tmp_path, empty=True)
    deps = Dependencies()
    from pydantic import ValidationError
    with pytest.raises(ValidationError, match="Complete research requires"):
        FactCheckingWorkflow(deps.runner()).run(project, store)
    assert read_state(store).current_state == S.RESEARCH_COMPLETE and deps.contexts == []
    assert store.list_versions("verification") == []


def fake_cli_dependencies(monkeypatch, deps):
    from types import SimpleNamespace
    client_calls = []
    def client(config):
        client_calls.append(config)
        return SimpleNamespace(close=lambda: None)
    monkeypatch.setattr("history_studio.research.openai_provider.create_client", client)
    # Only configured transport constructors are replaced; workflow/runner remain real.
    from test_verification_evidence import TextTools
    monkeypatch.setattr("history_studio.research.openai_provider.OpenAIWebTools", lambda *args: TextTools())
    original = deps.provider
    # Capture actual request context in provider rather than manufacturing a target.
    class ContextProvider:
        def decide(self, request):
            data = json.loads(request.input)
            from history_studio.models import VerificationContext
            context = VerificationContext.model_validate(data.get("verification_context", data))
            return original(context).decide(request)
    monkeypatch.setattr("history_studio.openai_model.OpenAIModelProvider", lambda *args: ContextProvider())
    return client_calls


def test_cli_verify_uses_fakes_reports_completion_and_is_idempotent(tmp_path, monkeypatch, capsys):
    prepare(tmp_path)
    deps = Dependencies()
    clients = fake_cli_dependencies(monkeypatch, deps)
    assert main(["--projects-dir", str(tmp_path), "verify", "test"]) == 0
    assert "WAITING_FACT_APPROVAL" in capsys.readouterr().out
    assert len(clients) == 1
    assert main(["--projects-dir", str(tmp_path), "verify", "test"]) == 0
    assert "already awaits" in capsys.readouterr().out and len(clients) == 1


def test_cli_wrong_state_and_missing_binding_do_not_construct_client(tmp_path, monkeypatch, capsys):
    prepare(tmp_path, S.CREATED, bound=False)
    deps = Dependencies()
    clients = fake_cli_dependencies(monkeypatch, deps)
    assert main(["--projects-dir", str(tmp_path), "verify", "test"]) == 1
    assert "Error:" in capsys.readouterr().err and clients == []


def test_cli_resume_remains_read_only_and_points_to_verify(tmp_path, capsys):
    _, store, _ = prepare(tmp_path, S.FACT_CHECKING)
    before = (store.project_dir / ".runtime/state.json").read_bytes()
    assert main(["--projects-dir", str(tmp_path), "resume", "test"]) == 0
    assert "Use verify" in capsys.readouterr().out
    assert (store.project_dir / ".runtime/state.json").read_bytes() == before


def test_cli_reports_incomplete_failure_without_hiding_it(tmp_path, monkeypatch, capsys):
    _, store, _ = prepare(tmp_path)
    deps = Dependencies({IDS[0]: [decision(text="ordinary model prose")]})
    clients = fake_cli_dependencies(monkeypatch, deps)
    assert main(["--projects-dir", str(tmp_path), "verify", "test"]) == 1
    text = capsys.readouterr().out
    assert "fact_checking_model_text" in text and "FAILED" in text
    assert read_state(store).failed_state == S.FACT_CHECKING and len(clients) == 1
    assert store.list_versions("verification") == []


def test_cli_complete_checkpoint_recovery_needs_no_client(tmp_path, monkeypatch, capsys):
    _, store, _ = prepare(tmp_path, S.FACT_CHECKING)
    assert Dependencies().runner().run(store, research_input_ref=research_ref(4)).is_complete
    deps = Dependencies()
    clients = fake_cli_dependencies(monkeypatch, deps)
    assert main(["--projects-dir", str(tmp_path), "verify", "test"]) == 0
    assert "0 facts checkpointed" in capsys.readouterr().out
    assert clients == [] and read_state(store).current_state == S.WAITING_FACT_APPROVAL
