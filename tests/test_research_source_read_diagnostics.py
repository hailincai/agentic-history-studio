"""Source-read exception metadata uses fake providers and contains no raw data."""
import errno

import pytest

from history_studio.research.agent import ResearchAgent
from history_studio.research.diagnostics import SourceReadDiagnostic, source_read_diagnostic
from history_studio.research.usage import UsageLedger
from history_studio.storage import ArtifactStore
from test_research_agent import setup_run, FakeProvider, FakeTools, calls, settings, state, ledger


SECRET = "sk-private-payload-user-data"


class FailingTools(FakeTools):
    def __init__(self, *, chained=False):
        super().__init__()
        self.chained = chained
        self.original = None

    def read_source(self, source, max_chars):
        super().read_source(source, max_chars)
        try:
            failure = OSError(errno.EACCES, SECRET, f"C:/private-user/{SECRET}/asset")
            failure.winerror = 32
            raise failure
        except OSError as inner:
            if self.chained:
                outer = RuntimeError(SECRET)
                self.original = outer
                raise outer from inner
            self.original = inner
            raise


def saved(store):
    return ArtifactStore(store.project_dir / ".runtime").load("diagnostics", 1, SourceReadDiagnostic)


@pytest.mark.parametrize("chained", [False, True])
def test_os_error_and_original_chain_preserve_failure_semantics(tmp_path, chained):
    project, store = setup_run(tmp_path)
    tools, events = FailingTools(chained=chained), []
    agent = ResearchAgent(FakeProvider(calls()), tools, settings(), events.append)
    package = agent.run(project, store)
    record = saved(store)
    assert package.progress.status == "FAILED"
    assert package.progress.stop_reason == "research_source_read_failed"
    assert state(store).failed_state == "RESEARCHING" and state(store).research_input_ref is None
    assert store.list_versions("research") == [1, 2]
    assert not (store.project_dir / ".runtime/research_completion.json").exists()
    assert ledger(store).source_reads == 1 and ledger(store).pending_request and ledger(store).unknown_usage
    assert agent.last_source_read_exception is tools.original
    assert record.operation == "source_read" and record.iteration == 1 and record.turn == 2
    assert record.pending_request
    if chained:
        assert agent.last_source_read_exception.__cause__.winerror == 32
        assert [item.error_class for item in record.errors] == ["RuntimeError", "PermissionError"]
    else:
        assert record.errors[0].error_class == "PermissionError"
    os_error = record.errors[-1]
    assert os_error.category == "os" and os_error.errno == errno.EACCES and os_error.winerror == 32
    assert any(frame.file == "research/agent.py" and frame.function == "_run"
               for frame in record.errors[0].traceback)
    assert all(frame.line > 0 for item in record.errors for frame in item.traceback)
    assert SECRET not in record.model_dump_json() + str(events)
    assert "private-user" not in record.model_dump_json() + str(events)
    assert not any(key in record.model_dump_json() for key in ('"message"', '"locals"', '"payload"'))


def test_ledger_failure_identifies_trusted_trace_locations_without_retry(tmp_path, monkeypatch):
    project, store = setup_run(tmp_path)
    persist = UsageLedger.persist
    failure = OSError(errno.EIO, SECRET)
    def fail_after_read(self, path):
        if self.source_reads == 1 and not self.pending_request:
            raise failure
        return persist(self, path)
    tools = FakeTools()
    # Fail just once; original failure handler must still publish its normal state.
    failed = False
    def once(self, path):
        nonlocal failed
        if not failed and self.source_reads == 1 and not self.pending_request:
            failed = True
            return fail_after_read(self, path)
        return persist(self, path)
    monkeypatch.setattr(UsageLedger, "persist", once)
    agent = ResearchAgent(FakeProvider(calls()), tools, settings())
    result = agent.run(project, store)
    assert result.progress.stop_reason == "research_source_read_failed"
    assert len(tools.reads) == 1
    assert agent.last_source_read_exception is failure
    record = saved(store)
    assert not record.pending_request
    assert any(frame.file == "research/usage.py" and frame.function == "record"
               for frame in record.errors[0].traceback)
    assert SECRET not in record.model_dump_json()


def test_user_paths_function_and_exception_names_are_redacted():
    namespace = {}
    code = compile(f"def {SECRET.replace('-', '_')}():\n    raise RuntimeError('payload')\n",
                   f"C:/users/private-person/{SECRET}.py", "exec")
    exec(code, namespace)
    try:
        namespace[SECRET.replace('-', '_')]()
    except RuntimeError as error:
        record = source_read_diagnostic(error, run_id="test", iteration=1, turn=1, pending_request=True)
    assert all(frame.file == "<external>" and frame.function == "<redacted>"
               for frame in record.errors[0].traceback)
    assert SECRET not in record.model_dump_json() and "private-person" not in record.model_dump_json()
    arbitrary = type(SECRET, (Exception,), {})()
    assert source_read_diagnostic(arbitrary, run_id="test", iteration=1, turn=1,
                                  pending_request=False).errors[0].error_class == "Exception"


def test_invalid_os_codes_and_cyclic_chains_are_bounded():
    error = OSError(SECRET)
    error.errno, error.winerror = True, SECRET
    error.__cause__ = error
    record = source_read_diagnostic(error, run_id="test", iteration=1, turn=1, pending_request=False)
    assert len(record.errors) == 1
    assert record.errors[0].errno is record.errors[0].winerror is None
    assert SECRET not in record.model_dump_json()


@pytest.mark.parametrize("unavailable", ["storage", "reporting"])
def test_diagnostic_failure_never_changes_original_failure_handling(tmp_path, monkeypatch, unavailable):
    project, store = setup_run(tmp_path)
    tools = FailingTools()
    save = ArtifactStore.save
    if unavailable == "storage":
        def cannot_save(self, kind, package):
            if kind == "diagnostics":
                raise OSError(SECRET)
            return save(self, kind, package)
        monkeypatch.setattr(ArtifactStore, "save", cannot_save)
    def emit(event):
        if unavailable == "reporting" and event.startswith("Research source-read"):
            raise RuntimeError(SECRET)
    agent = ResearchAgent(FakeProvider(calls()), tools, settings(), emit)
    result = agent.run(project, store)
    assert result.progress.stop_reason == "research_source_read_failed"
    assert agent.last_source_read_exception is tools.original
    assert state(store).failed_state == "RESEARCHING"
    assert store.list_versions("research") == [1, 2]
