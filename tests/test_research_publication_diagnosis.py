"""P9-F2 diagnosis cases migrated into durable completion regression tests.

The historical COMPLETE -> FAILED cause remains unknown. Only synthetic local
faults are injected; every artifact belongs to a new tmp_path fixture.
"""
import errno
import hashlib
import json

import pytest

import history_studio.research.agent as module
from history_studio.models.research_package import ResearchPackage, ResearchRunStatus as R
from history_studio.research.agent import ResearchAgent
from history_studio.research.completion import CompletionIntent, CompletionPublicationError, digest
from history_studio.research.usage import UsageLedger
from history_studio.workflow import ProjectState as S, ProjectStateMachine
from history_studio.storage.artifact_store import write_json
from test_research_agent import setup_run, FakeProvider, FakeTools, calls, settings, state, ledger


def intent_path(store):
    return store.project_dir / ".runtime/research_completion.json"


def load_intent(store):
    return CompletionIntent.model_validate_json(intent_path(store).read_bytes())


def install_fault(patch, store, window, error):
    publish, persist = module.publish_intent, UsageLedger.persist
    finish, transition = ResearchAgent._finish_workflow, ProjectStateMachine.complete_research
    write, summary = module.write_json, ResearchAgent._summary
    def intent(path, value):
        if window == "before_intent":
            raise error
        publish(path, value)
        if window == "after_intent":
            raise error
    def progress(self, path):
        completing = intent_path(store).exists() and self.last_checkpoint_outcome is not None
        if window == "before_ledger" and completing:
            raise error
        persist(self, path)
        if window == "after_ledger" and completing:
            raise error
    def binding(self, machine, path, **kwargs):
        if window == "before_binding":
            raise error
        return finish(self, machine, path, **kwargs)
    def bound(self, reference, **kwargs):
        result = transition(self, reference, **kwargs)
        if window == "after_binding" and intent_path(store).exists():
            raise error
        return result
    def state_write(path, value, **kwargs):
        completing = path.name == "state.json" and value.current_state == S.RESEARCH_COMPLETE
        if window == "before_state" and completing:
            raise error
        write(path, value, **kwargs)
        if window == "after_state" and completing:
            raise error
    def report(self, package, usage):
        if window == "reporting":
            raise error
        return summary(self, package, usage)
    patch.setattr(module, "publish_intent", intent)
    patch.setattr(UsageLedger, "persist", progress)
    patch.setattr(ResearchAgent, "_finish_workflow", binding)
    patch.setattr(ProjectStateMachine, "complete_research", bound)
    patch.setattr(module, "write_json", state_write)
    patch.setattr(ResearchAgent, "_summary", report)


@pytest.mark.parametrize("window", ["before_intent", "after_intent", "before_ledger", "after_ledger",
                                   "before_binding", "after_binding", "before_state", "after_state", "reporting"])
@pytest.mark.parametrize("crash", [False, True])
def test_completion_publication_windows_and_repeated_restart(tmp_path, monkeypatch, window, crash):
    project, store = setup_run(tmp_path)
    provider = FakeProvider(calls())
    error = KeyboardInterrupt("synthetic crash") if crash else OSError(errno.EACCES, "private-payload-and-key")
    with monkeypatch.context() as patch:
        install_fault(patch, store, window, error)
        with pytest.raises(KeyboardInterrupt if crash else CompletionPublicationError) as caught:
            ResearchAgent(provider, FakeTools(), settings()).run(project, store)
        if not crash:
            assert caught.value.__cause__ is error
            assert "private-payload-and-key" not in str(caught.value)
    assert provider.calls == 3
    assert store.list_versions("research") == [1, 2]
    candidate = store.load("research", 2, ResearchPackage)
    assert candidate.progress.status == R.COMPLETE
    candidate_bytes = (store.project_dir / "research/research_v2.json").read_bytes()
    usage_before = ledger(store)
    for _ in range(2):
        empty, tools = FakeProvider([]), FakeTools()
        if window == "before_intent":
            with pytest.raises(ValueError, match="binding is missing"):
                ResearchAgent(empty, tools, settings()).run(project, store)
        else:
            monkeypatch.setattr(ResearchAgent, "_load_package", lambda *args: pytest.fail("No latest discovery"))
            assert ResearchAgent(empty, tools, settings()).run(project, store) == candidate
            assert state(store).current_state == S.RESEARCH_COMPLETE
            assert state(store).research_input_ref == load_intent(store).candidate_ref
        assert empty.calls == 0 and not tools.queries and not tools.reads
    assert store.list_versions("research") == [1, 2]
    assert (store.project_dir / "research/research_v2.json").read_bytes() == candidate_bytes
    accounting = {"progress_seen", "last_checkpoint_outcome", "consecutive_no_progress"}
    assert ledger(store).model_dump(exclude=accounting) == usage_before.model_dump(exclude=accounting)
    assert ledger(store).committed_budget_usd == 0.004
    for path in (store.project_dir / ".runtime/diagnostics").glob("*.json"):
        raw = path.read_text(encoding="utf-8")
        assert "private-payload-and-key" not in raw
        assert json.loads(raw)["stage"] == "research_completion"
        assert json.loads(raw)["errors"][0]["errno"] == errno.EACCES


def test_validated_completion_intent_exact_schema_and_free_reentry(tmp_path, monkeypatch):
    project, store = setup_run(tmp_path)
    result = ResearchAgent(FakeProvider(calls()), FakeTools(), settings()).run(project, store)
    intent = load_intent(store)
    intent.verify_commit()
    assert intent.run_id == result.progress.run_id == ledger(store).run_id
    assert intent.candidate_ref == state(store).research_input_ref
    assert intent.candidate_sha256 == hashlib.sha256((store.project_dir / "research/research_v2.json").read_bytes()).hexdigest()
    assert intent.sources_sha256 == digest([source.model_dump(mode="json") for source in result.sources])
    assert intent.state_before.current_state == S.RESEARCHING
    assert intent.state_after == state(store)
    before = {path: path.read_bytes() for path in (intent_path(store), store.project_dir / ".runtime/research_usage.json",
                                                  store.project_dir / ".runtime/state.json")}
    monkeypatch.setattr(ResearchAgent, "_load_package", lambda *args: pytest.fail("No latest discovery"))
    for _ in range(2):
        provider = FakeProvider([])
        assert ResearchAgent(provider, FakeTools(), settings()).run(project, store) == result
        assert provider.calls == 0
    assert all(path.read_bytes() == data for path, data in before.items())


@pytest.mark.parametrize("corruption", ["candidate", "missing", "source", "state", "ledger", "run", "intent",
                                      "intent_accounting", "missing_ledger"])
def test_conflicting_completion_evidence_fails_before_dispatch_or_repair(tmp_path, corruption):
    project, store = setup_run(tmp_path)
    ResearchAgent(FakeProvider(calls()), FakeTools(), settings()).run(project, store)
    candidate = store.project_dir / "research/research_v2.json"
    if corruption == "missing":
        candidate.unlink()
    elif corruption == "candidate":
        candidate.write_bytes(candidate.read_bytes() + b"\n")
    elif corruption == "source":
        package = json.loads(candidate.read_bytes())
        package["sources"][0]["title"] = "tampered source"
        candidate.write_text(json.dumps(package), encoding="utf-8")
    elif corruption == "state":
        snapshot = state(store).model_dump(mode="json")
        snapshot["artifacts"]["research"]["version"] = 99
        from history_studio.workflow import RuntimeState
        write_json(store.project_dir / ".runtime/state.json", RuntimeState.model_validate(snapshot), replace=True)
    elif corruption in ("ledger", "run"):
        usage = ledger(store)
        if corruption == "run":
            usage.run_id = "different-run"
        else:
            usage.committed_budget_usd += 1
        usage.persist(store.project_dir / ".runtime/research_usage.json")
    elif corruption == "missing_ledger":
        (store.project_dir / ".runtime/research_usage.json").unlink()
    elif corruption == "intent_accounting":
        value = json.loads(intent_path(store).read_bytes())
        value["ledger_before"]["committed_budget_usd"] += 1
        value["commit_sha256"] = digest({key: item for key, item in value.items() if key != "commit_sha256"})
        intent_path(store).write_text(json.dumps(value), encoding="utf-8")
    else:
        value = json.loads(intent_path(store).read_bytes())
        value["candidate_sha256"] = "different"
        intent_path(store).write_text(json.dumps(value), encoding="utf-8")
    before = {path: path.read_bytes() for path in store.project_dir.rglob("*.json")}
    for _ in range(2):
        provider = FakeProvider([])
        with pytest.raises(CompletionPublicationError, match="completion_authentication"):
            ResearchAgent(provider, FakeTools(), settings()).run(project, store)
        assert provider.calls == 0
    assert all(path.read_bytes() == data for path, data in before.items())
    assert store.list_versions("research") == ([1] if corruption == "missing" else [1, 2])


def test_v5_completion_never_publishes_contradictory_v6(tmp_path, monkeypatch):
    project, store = setup_run(tmp_path)
    original = store.save
    def reserve_versions(kind, package):
        if kind == "research" and package.progress.status == R.COMPLETE:
            while len(store.list_versions("research")) < 4:
                original(kind, package.model_copy(update={"progress": package.progress.model_copy(update={"status": R.RUNNING})}))
        return original(kind, package)
    monkeypatch.setattr(store, "save", reserve_versions)
    with monkeypatch.context() as patch:
        install_fault(patch, store, "before_state", OSError("synthetic publication fault"))
        with pytest.raises(CompletionPublicationError):
            ResearchAgent(FakeProvider(calls()), FakeTools(), settings()).run(project, store)
    assert store.list_versions("research") == [1, 2, 3, 4, 5]
    assert load_intent(store).candidate_ref.version == 5
    empty = FakeProvider([])
    assert ResearchAgent(empty, FakeTools(), settings()).run(project, store).progress.status == R.COMPLETE
    assert empty.calls == 0 and state(store).research_input_ref.version == 5
    assert store.list_versions("research") == [1, 2, 3, 4, 5]


def test_publication_exception_chain_and_diagnostic_write_failure(tmp_path, monkeypatch):
    project, store = setup_run(tmp_path)
    inner = PermissionError(errno.EACCES, "credential-and-path")
    inner.winerror = 32
    outer = RuntimeError("private-body")
    outer.__cause__ = inner
    with monkeypatch.context() as patch:
        install_fault(patch, store, "before_state", outer)
        with pytest.raises(CompletionPublicationError) as caught:
            ResearchAgent(FakeProvider(calls()), FakeTools(), settings()).run(project, store)
        assert caught.value.__cause__ is outer and outer.__cause__ is inner
    path, = (store.project_dir / ".runtime/diagnostics").glob("*.json")
    record = json.loads(path.read_bytes())
    assert record["operation"] == "completion_state"
    assert record["errors"][1] == dict(error_class="PermissionError", errno=errno.EACCES, winerror=32)
    assert "credential-and-path" not in path.read_text() and "private-body" not in path.read_text()
    save = module.ArtifactStore.save
    def unavailable(self, kind, model):
        if kind == "diagnostics":
            raise OSError("secondary diagnostic failure")
        return save(self, kind, model)
    with monkeypatch.context() as patch:
        patch.setattr(module.ArtifactStore, "save", unavailable)
        patch.setattr(ResearchAgent, "_summary", lambda *args: (_ for _ in ()).throw(outer))
        with pytest.raises(CompletionPublicationError) as repeated:
            ResearchAgent(FakeProvider([]), FakeTools(), settings()).run(project, store)
        assert repeated.value.__cause__ is outer
    assert state(store).current_state == S.RESEARCH_COMPLETE


def test_missing_intent_never_adopts_complete_hidden_by_newer_failure(tmp_path):
    project, store = setup_run(tmp_path)
    complete = ResearchAgent(FakeProvider(calls()), FakeTools(), settings()).run(project, store)
    intent_path(store).unlink()
    from history_studio.workflow import RuntimeState
    write_json(store.project_dir / ".runtime/state.json",
               RuntimeState(current_state=S.RESEARCHING), replace=True)
    newer = complete.model_copy(deep=True)
    newer.progress.status = R.FAILED
    store.save("research", newer)
    empty = FakeProvider([])
    with pytest.raises(ValueError, match="binding is missing"):
        ResearchAgent(empty, FakeTools(), settings()).run(project, store)
    assert empty.calls == 0 and state(store).research_input_ref is None


def test_completion_recovery_never_debits_or_releases_project_reservations(tmp_path, monkeypatch):
    from history_studio.budget import ProjectBudget
    project, store = setup_run(tmp_path)
    budget = ProjectBudget(store.project_dir, stage="research")
    budget.initialize_new()
    budget.reserve(request_id="uncertain-prior-request", operation="responses.create", model="fake",
                   request_sha256="fake-request-hash", maximum_usd="0.10", basis={"fixture": "uncertain"})
    original = budget.path.read_bytes()
    with monkeypatch.context() as patch:
        install_fault(patch, store, "before_state", OSError("synthetic state failure"))
        with pytest.raises(CompletionPublicationError):
            ResearchAgent(FakeProvider(calls()), FakeTools(), settings()).run(project, store)
    for _ in range(2):
        empty = FakeProvider([])
        assert ResearchAgent(empty, FakeTools(), settings()).run(project, store).progress.status == R.COMPLETE
        assert empty.calls == 0
        assert budget.path.read_bytes() == original
