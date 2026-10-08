"""Local fake-provider recovery, crash windows and independent process attempts."""
import json
import multiprocessing
import time
from pathlib import Path

import pytest

from history_studio.media.recovery import RecoveryJournal, AssetRecovery
from history_studio.models import ProjectConfig, StoryboardPackage, ScriptPackage
from history_studio.storage import ArtifactStore, MediaStore
from history_studio.storage.artifact_store import write_json
from history_studio.workflow import RuntimeState, ProjectState as S
from history_studio.workflow.media import MediaProviders, MediaWorkflow
from test_media_workflow import prepared, workflow, read, ref
from test_narration_media import FakeTTSProvider
from test_visual_media import FakeImageProvider, FakeVideoProvider


def journal(store):
    return RecoveryJournal.model_validate_json((store.project_dir / ".runtime/media_recovery.json").read_text(encoding="utf-8"))


def counts(providers):
    return len(providers.tts.texts), len(providers.image.prompts), len(providers.video.calls)


def test_multi_asset_completion_and_bound_restart_uses_no_providers(prepared):
    project, store, providers = prepared
    authority = read(store).artifacts
    outcome = workflow(providers).run(project, store)
    assert outcome.state.current_state == S.ASSEMBLING
    record = journal(store)
    assert record.scope.storyboard_ref == ref("storyboard")
    assert record.scope.script_ref == ref("script")
    assert len(record.entries) == 6
    assert {entry.state for entry in record.entries.values()} == {"COMPLETED"}
    assert len({entry.dispatch_id for entry in record.entries.values()}) == 6
    assert all(len(entry.request_sha256) == 64 for entry in record.entries.values())
    assert outcome.state.artifacts.approved_storyboard == authority.approved_storyboard
    before = counts(providers), record.model_dump_json()
    assert workflow(providers).run(project, store).package == outcome.package
    assert before == (counts(providers), journal(store).model_dump_json())


def test_failure_before_second_dispatch_reuses_first_and_only_generates_untouched(prepared, monkeypatch):
    project, store, providers = prepared
    perform = AssetRecovery.perform
    def interrupt(self, **kwargs):
        if kwargs["key"] == "narration:SEG-A":
            raise OSError("local orchestration interruption before intent")
        return perform(self, **kwargs)
    monkeypatch.setattr(AssetRecovery, "perform", interrupt)
    first = workflow(providers).run(project, store)
    assert first.state.current_state == S.FAILED and first.state.artifacts.media is None
    assert list(journal(store).entries) == ["narration:SEG-Z"]
    first_asset = journal(store).entries["narration:SEG-Z"].result
    assert providers.tts.texts == ["Exact first narration — 李白。"]
    monkeypatch.setattr(AssetRecovery, "perform", perform)
    resumed = workflow(providers).run(project, store)
    assert resumed.state.current_state == S.ASSEMBLING
    assert counts(providers) == (2, 2, 2)
    assert journal(store).entries["narration:SEG-Z"].result == first_asset
    assert [asset.segment_id for asset in resumed.package.narration_assets] == ["SEG-Z", "SEG-A"]


@pytest.mark.parametrize("failure", ["provider", "before_provider", "invalid_response", "publication", "completion_write"])
def test_uncertain_dispatch_never_reissues_on_restart(prepared, monkeypatch, failure):
    import history_studio.media.recovery as module
    project, store, providers = prepared
    binaries = MediaStore(store.project_dir / "media")
    if failure == "provider":
        providers.tts.fail_at = 1
    elif failure == "invalid_response":
        # Keep scope fixed across retry; invalid bytes are provider output, not a
        # changed configured synthetic payload.
        from history_studio.media.tts import TTSResult
        monkeypatch.setattr(providers.tts, "synthesize", lambda **kwargs: TTSResult(audio_bytes=b"bad wav", provider="fake", model="pcm"))
    elif failure == "publication":
        monkeypatch.setattr(binaries, "save_bytes", lambda **kwargs: (_ for _ in ()).throw(OSError("publication failed")))
    elif failure == "before_provider":
        def crash(**kwargs):
            raise KeyboardInterrupt("crash between intent and invocation")
        monkeypatch.setattr(providers.tts, "synthesize", crash)
    else:
        write = module.write_json
        def broken(path, record, **kwargs):
            if record.entries and any(entry.state == "COMPLETED" for entry in record.entries.values()):
                raise OSError("completion journal publication failed")
            return write(path, record, **kwargs)
        monkeypatch.setattr(module, "write_json", broken)
    if failure == "before_provider":
        with pytest.raises(KeyboardInterrupt):
            workflow(providers).run(project, store, media_store=binaries)
    else:
        assert workflow(providers).run(project, store, media_store=binaries).state.current_state == S.FAILED
    assert journal(store).entries["narration:SEG-Z"].state == "UNCERTAIN"
    before = counts(providers)
    providers.tts.fail_at = None
    monkeypatch.undo()
    retry = workflow(providers).run(project, store, media_store=binaries)
    assert retry.state.current_state == S.FAILED and retry.error_type == "UncertainMediaRequest"
    assert retry.state.artifacts.media is None and before == counts(providers)


@pytest.mark.parametrize("mode", ["missing", "tamper", "identity"])
def test_completed_asset_requires_exact_identity_hash_and_duration(prepared, monkeypatch, mode):
    project, store, providers = prepared
    perform = AssetRecovery.perform
    def interrupt(self, **kwargs):
        if kwargs["key"] == "narration:SEG-A":
            raise OSError("stop after first completion")
        return perform(self, **kwargs)
    monkeypatch.setattr(AssetRecovery, "perform", interrupt)
    workflow(providers).run(project, store)
    monkeypatch.setattr(AssetRecovery, "perform", perform)
    record = journal(store)
    entry = record.entries["narration:SEG-Z"]
    target = store.project_dir / "media" / entry.result["asset"]["relative_path"]
    if mode == "missing":
        target.unlink()
    elif mode == "tamper":
        target.write_bytes(b"changed")
    else:
        entry.result["segment_id"] = "SEG-A"
        write_json(store.project_dir / ".runtime/media_recovery.json", record, replace=True)
    before = counts(providers)
    retry = workflow(providers).run(project, store)
    assert retry.state.current_state == S.FAILED and retry.state.artifacts.media is None
    assert before == counts(providers)


@pytest.mark.parametrize("changed", ["storyboard", "script", "source_hash", "provider", "config"])
def test_recovery_scope_never_crosses_candidate_or_configuration(prepared, monkeypatch, changed):
    project, store, providers = prepared
    providers.tts.fail_at = 1
    workflow(providers).run(project, store)
    providers.tts.fail_at = None
    state = read(store).model_dump(mode="json")
    if changed in ("storyboard", "script"):
        board = store.load("storyboard", 1, StoryboardPackage).model_dump(mode="json")
        if changed == "script":
            script = store.load("script", 1, ScriptPackage)
            script_version = store.save("script", script)
            board["script_input_ref"]["version"] = script_version
            state["artifacts"]["approved_script"]["version"] = script_version
        version = store.save("storyboard", StoryboardPackage.model_validate(board))
        for key in ("storyboard", "approved_storyboard"):
            state["artifacts"][key]["version"] = version
        write_json(store.project_dir / ".runtime/state.json", RuntimeState.model_validate(state), replace=True)
    elif changed == "source_hash":
        path = store.project_dir / "script/script_v1.json"
        path.write_text(path.read_text(encoding="utf-8") + "\n", encoding="utf-8")
    elif changed == "provider":
        providers = MediaProviders(providers.tts, providers.image, providers.video, recovery_identity={"voice_revision": "different"})
    else:
        (store.project_dir / ".runtime/media_config.json").write_text('{"revision":2}', encoding="utf-8")
    before = counts(providers)
    retry = workflow(providers).run(project, store)
    assert retry.state.current_state == S.FAILED and retry.state.artifacts.media is None
    assert before == counts(providers)


def test_intent_write_failure_after_atomic_publication_still_blocks_dispatch(prepared, monkeypatch):
    import history_studio.media.recovery as module
    project, store, providers = prepared
    write = module.write_json
    def published_then_error(path, record, **kwargs):
        write(path, record, **kwargs)
        if record.entries:
            raise OSError("ambiguous intent write acknowledgement")
    monkeypatch.setattr(module, "write_json", published_then_error)
    workflow(providers).run(project, store)
    assert counts(providers) == (0, 0, 0)
    monkeypatch.setattr(module, "write_json", write)
    assert workflow(providers).run(project, store).error_type == "UncertainMediaRequest"
    assert counts(providers) == (0, 0, 0)


def test_explicit_no_charge_reconciliation_is_audited_without_budget_refund(prepared):
    from datetime import datetime, timezone
    from history_studio.budget import ProjectBudget, BudgetReconciliation
    project, store, providers = prepared
    budget = ProjectBudget(store.project_dir, stage="media")
    budget.initialize(BudgetReconciliation(project_id=project.project_id, historical_committed_usd=0,
        evidence="synthetic project has no paid history", decided_by="Operator", decided_at=datetime.now(timezone.utc)))
    budget.reserve(request_id="budget-request", operation="tts", model="synthetic", request_sha256="synthetic",
                   maximum_usd="0.5", basis={"test": True})
    budget_before = budget.path.read_bytes()
    providers.tts.fail_at = 1
    workflow(providers).run(project, store)
    record = journal(store)
    entry = record.entries["narration:SEG-Z"]
    recovery = AssetRecovery(store.project_dir, record.scope)
    with pytest.raises(ValueError, match="exact uncertain dispatch"):
        recovery.confirm_no_charge(key=entry.asset_key, dispatch_id="wrong", evidence="proof", decided_by="Operator")
    with pytest.raises(ValueError):
        recovery.confirm_no_charge(key=entry.asset_key, dispatch_id=entry.dispatch_id, evidence="", decided_by="Operator")
    recovery.confirm_no_charge(key=entry.asset_key, dispatch_id=entry.dispatch_id,
        evidence="Operator stopped all writers; provider record proves no charge and no result", decided_by="Operator")
    assert journal(store).entries == {}
    assert journal(store).no_charge_reconciliations[0].dispatch_id == entry.dispatch_id
    assert budget.path.read_bytes() == budget_before and budget.snapshot()["outstanding_usd"] == 0.5
    providers.tts.fail_at = None
    assert workflow(providers).run(project, store).state.current_state == S.ASSEMBLING
    assert journal(store).entries[entry.asset_key].dispatch_id != entry.dispatch_id
    assert len(journal(store).no_charge_reconciliations) == 1


@pytest.mark.parametrize("key,action", [("image:SHOT-0", "missing"), ("video:SHOT-1", "tamper"),
                                      ("image:intermediate:SHOT-2", "tamper")])
def test_completed_visual_and_intermediate_integrity_are_required_before_reuse(prepared, monkeypatch, key, action):
    project, store, providers = prepared
    save = store.save
    monkeypatch.setattr(store, "save", lambda *args: (_ for _ in ()).throw(OSError("manifest publication failed")))
    workflow(providers).run(project, store)
    monkeypatch.setattr(store, "save", save)
    entry = journal(store).entries[key]
    target = store.project_dir / "media" / entry.result["asset"]["relative_path"]
    if action == "missing":
        target.unlink()
    else:
        target.write_bytes(b"tampered")
    before = counts(providers)
    retry = workflow(providers).run(project, store)
    assert retry.state.current_state == S.FAILED and retry.state.artifacts.media is None
    assert counts(providers) == before


def test_dispatch_intent_not_published_never_calls_provider_and_can_retry(prepared, monkeypatch):
    import history_studio.media.recovery as module
    project, store, providers = prepared
    write = module.write_json
    def denied(path, record, **kwargs):
        if record.entries:
            raise OSError("intent publication never happened")
        return write(path, record, **kwargs)
    monkeypatch.setattr(module, "write_json", denied)
    workflow(providers).run(project, store)
    assert counts(providers) == (0, 0, 0) and journal(store).entries == {}
    monkeypatch.setattr(module, "write_json", write)
    assert workflow(providers).run(project, store).state.current_state == S.ASSEMBLING


def test_missing_journal_cannot_adopt_or_regenerate_orphan_bytes(prepared, monkeypatch):
    project, store, providers = prepared
    save = store.save
    monkeypatch.setattr(store, "save", lambda *args: (_ for _ in ()).throw(OSError("manifest publication failed")))
    workflow(providers).run(project, store)
    monkeypatch.setattr(store, "save", save)
    (store.project_dir / ".runtime/media_recovery.json").unlink()
    before = counts(providers)
    retry = workflow(providers).run(project, store)
    assert retry.state.current_state == S.FAILED and retry.state.artifacts.media is None
    assert counts(providers) == before


def test_no_charge_cannot_adopt_or_ignore_published_uncertain_bytes(prepared, monkeypatch):
    import history_studio.media.recovery as module
    project, store, providers = prepared
    write = module.write_json
    def broken(path, record, **kwargs):
        if record.entries and any(entry.state == "COMPLETED" for entry in record.entries.values()):
            raise OSError("lost completion write")
        return write(path, record, **kwargs)
    monkeypatch.setattr(module, "write_json", broken)
    workflow(providers).run(project, store)
    monkeypatch.setattr(module, "write_json", write)
    record = journal(store)
    entry = record.entries["narration:SEG-Z"]
    with pytest.raises(ValueError, match="Published bytes"):
        AssetRecovery(store.project_dir, record.scope).confirm_no_charge(key=entry.asset_key,
            dispatch_id=entry.dispatch_id, evidence="unsupported claim", decided_by="Operator")
    assert journal(store).entries[entry.asset_key].state == "UNCERTAIN"


class CountingTTS(FakeTTSProvider):
    def __init__(self, path):
        super().__init__()
        self.path = path

    def synthesize(self, *, text):
        with self.path.open("a", encoding="utf-8") as stream:
            stream.write(json.dumps(text) + "\n")
        time.sleep(0.2)
        return super().synthesize(text=text)


def recovery_worker(root, queue):
    try:
        store = ArtifactStore(Path(root))
        project = ProjectConfig.model_validate_json((store.project_dir / "project.json").read_text(encoding="utf-8"))
        providers = MediaProviders(CountingTTS(store.project_dir / "dispatches.log"), FakeImageProvider([]), FakeVideoProvider([]))
        outcome = workflow(providers).run(project, store)
        queue.put(outcome.state.current_state.value)
    except Exception as exc:
        queue.put(type(exc).__name__)


def test_independent_processes_cannot_duplicate_provider_dispatches(prepared):
    project, store, _ = prepared
    ctx = multiprocessing.get_context("spawn")
    queue = ctx.Queue()
    workers = [ctx.Process(target=recovery_worker, args=(str(store.project_dir), queue)) for _ in range(2)]
    try:
        for worker in workers:
            worker.start()
        results = [queue.get(timeout=40) for _ in workers]
        for worker in workers:
            worker.join(timeout=10)
            assert worker.exitcode == 0
        assert "ASSEMBLING" in results
        assert len((store.project_dir / "dispatches.log").read_text().splitlines()) == 2
        assert read(store).current_state == S.ASSEMBLING and read(store).artifacts.media is not None
        assert {entry.state for entry in journal(store).entries.values()} == {"COMPLETED"}
    finally:
        for worker in workers:
            if worker.is_alive():
                worker.terminate()
                worker.join(timeout=5)
        queue.close()
