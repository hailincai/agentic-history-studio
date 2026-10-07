import json

import pytest

from history_studio.media import MediaIntegrityIssue, MediaIntegrityReport
from history_studio.models import ArtifactReference, MediaPackage, ProjectConfig, ScriptPackage, StoryboardPackage
from history_studio.storage import ArtifactStore, MediaStore
from history_studio.storage.artifact_store import write_json
from history_studio.workflow import ProjectState as S, RuntimeState, WorkflowArtifactBindings, ProjectStateMachine
from history_studio.workflow.media import MediaProviders, MediaWorkflow
from test_narration_media import FakeTTSProvider, script_package, storyboard_package
from test_visual_media import FakeImageProvider, FakeVideoProvider


def ref(kind, version=1):
    return ArtifactReference(project_id="project", artifact_type=kind, version=version)


def read(store):
    return RuntimeState.model_validate_json((store.project_dir / ".runtime/state.json").read_text(encoding="utf-8"))


@pytest.fixture
def prepared(tmp_path):
    project = ProjectConfig(project_id="project", topic="Test")
    store = ArtifactStore(tmp_path / "project")
    store.save("script", script_package())
    newer = script_package().model_dump(mode="json")
    newer["sections"][0]["segments"][0]["narration"] = "Forbidden newer narration"
    store.save("script", ScriptPackage.model_validate(newer))
    board = storyboard_package().model_dump(mode="json")
    for shot, method in zip(board["sections"][0]["shots"], ["STATIC_IMAGE", "TEXT_TO_VIDEO", "IMAGE_TO_VIDEO"]):
        shot["generation_method"] = method
    store.save("storyboard", StoryboardPackage.model_validate(board))
    store.save("storyboard", storyboard_package(script_version=2))
    state = RuntimeState(current_state=S.STORYBOARD_APPROVED, last_successful_state=S.STORYBOARD_APPROVED,
        artifacts=WorkflowArtifactBindings(research=ref("research"), verification=ref("verification", 42),
            approved_verification=ref("verification", 42), story=ref("story", 3), approved_story=ref("story", 3),
            script=ref("script", 2), approved_script=ref("script"), storyboard=ref("storyboard", 2),
            approved_storyboard=ref("storyboard")))
    write_json(store.project_dir / "project.json", project)
    write_json(store.project_dir / ".runtime/state.json", state)
    events = []
    providers = MediaProviders(FakeTTSProvider(), FakeImageProvider(events), FakeVideoProvider(events))
    return project, store, providers


def workflow(providers):
    return MediaWorkflow(provider_factory=lambda storyboard: providers)


def test_success_exact_authority_publication_order_bound_rerun_and_no_gate(prepared, monkeypatch):
    import history_studio.workflow.media as module
    project, store, providers = prepared
    upstream = read(store).artifacts
    events, factories = [], []
    validate, save, load, write = module.validate_media_package, store.save, store.load, module.write_json
    def check(**kwargs):
        assert kwargs["storyboard_input_ref"] == ref("storyboard")
        assert kwargs["script_input_ref"] == ref("script")
        events.append("validate")
        return validate(**kwargs)
    def publish(kind, package):
        assert kind == "media" and events[-1] == "validate"
        assert read(store).current_state == S.GENERATING_MEDIA and read(store).artifacts.media is None
        events.append("persist")
        return save(kind, package)
    def reload(kind, version, model):
        assert kind in ("storyboard", "script", "media") and version == 1
        if kind == "media":
            events.append("reload")
        return load(kind, version, model)
    def snapshot(path, state, **kwargs):
        if state.current_state == S.ASSEMBLING:
            assert events == ["validate", "persist", "reload", "validate"]
            assert state.artifacts.media == ref("media")
            events.append("bind")
        return write(path, state, **kwargs)
    def factory(board):
        assert read(store).current_state == S.GENERATING_MEDIA
        factories.append(board)
        board.title = "Factory mutation must not change authority"
        return providers
    monkeypatch.setattr(module, "validate_media_package", check)
    monkeypatch.setattr(module, "write_json", snapshot)
    monkeypatch.setattr(store, "save", publish)
    monkeypatch.setattr(store, "load", reload)
    monkeypatch.setattr(store, "load_latest", lambda *args: pytest.fail("No latest discovery"))
    runner = MediaWorkflow(provider_factory=factory)
    outcome = runner.run(project, store)
    assert outcome.state.current_state == S.ASSEMBLING
    assert read(store) == outcome.state
    assert outcome.package.title == "Documentary"
    assert [asset.segment_id for asset in outcome.package.narration_assets] == ["SEG-Z", "SEG-A"]
    assert providers.tts.texts == ["Exact first narration — 李白。", "Second\nline of narration."]
    assert [asset.shot_id for asset in outcome.package.visual_assets] == ["SHOT-0", "SHOT-1", "SHOT-2"]
    assert len(providers.image.prompts) == 2 and len(providers.video.calls) == 2
    for field in WorkflowArtifactBindings.model_fields:
        if field != "media":
            assert getattr(outcome.state.artifacts, field) == getattr(upstream, field)
    assert not (store.project_dir / "approvals").exists()
    state_bytes = (store.project_dir / ".runtime/state.json").read_bytes()
    events.clear()
    again = runner.run(project, store)
    assert again.package == outcome.package and len(factories) == 1
    assert events == ["reload", "validate"]
    assert (store.project_dir / ".runtime/state.json").read_bytes() == state_bytes


@pytest.mark.parametrize("failure", ["tts", "visual", "validation", "binary_save", "manifest_save", "reload", "bind", "duration"])
def test_failure_preserves_approval_no_binding_and_retry_never_adopts_orphans(prepared, monkeypatch, failure):
    import history_studio.workflow.media as module
    project, store, providers = prepared
    upstream = read(store).artifacts
    save, load, validate, write, narration = store.save, store.load, module.validate_media_package, module.write_json, module.generate_narration_assets
    binaries = MediaStore(store.project_dir / "media")
    binary_save = binaries.save_bytes
    if failure == "tts":
        providers.tts.fail_at = 1
    elif failure == "visual":
        providers.image.fail_at = 1
    elif failure == "validation":
        monkeypatch.setattr(module, "validate_media_package", lambda **kwargs: MediaIntegrityReport(issues=[
            MediaIntegrityIssue(code="NARRATION_MISSING", message="Missing narration")]))
    elif failure == "binary_save":
        monkeypatch.setattr(binaries, "save_bytes", lambda **kwargs: (_ for _ in ()).throw(OSError("storage failed")))
    elif failure == "manifest_save":
        monkeypatch.setattr(store, "save", lambda *args: (_ for _ in ()).throw(OSError("manifest failed")))
    elif failure == "reload":
        def broken(kind, version, model):
            if kind == "media":
                raise OSError("reload failed")
            return load(kind, version, model)
        monkeypatch.setattr(store, "load", broken)
    elif failure == "bind":
        def broken_write(path, state, **kwargs):
            if state.current_state == S.ASSEMBLING:
                raise OSError("state publication failed")
            return write(path, state, **kwargs)
        monkeypatch.setattr(module, "write_json", broken_write)
    else:
        def wrong_duration(**kwargs):
            values = narration(**kwargs)
            return tuple(asset.model_copy(update={"duration_seconds": 999}) for asset in values)
        monkeypatch.setattr(module, "generate_narration_assets", wrong_duration)
    outcome = workflow(providers).run(project, store, media_store=binaries)
    assert outcome.state.current_state == S.FAILED and outcome.state.failed_state == S.GENERATING_MEDIA
    assert outcome.state.artifacts.media is None and read(store) == outcome.state
    assert outcome.state.artifacts == upstream
    expected_orphan = failure in ("reload", "bind")
    assert store.list_versions("media") == ([1] if expected_orphan else [])
    providers.tts.fail_at = providers.image.fail_at = None
    monkeypatch.setattr(binaries, "save_bytes", binary_save)
    monkeypatch.setattr(store, "save", save)
    monkeypatch.setattr(store, "load", load)
    monkeypatch.setattr(module, "validate_media_package", validate)
    monkeypatch.setattr(module, "write_json", write)
    monkeypatch.setattr(module, "generate_narration_assets", narration)
    monkeypatch.setattr(store, "load_latest", lambda *args: pytest.fail("No latest lookup"))
    resumed = workflow(providers).run(project, store, media_store=binaries)
    assert resumed.state.current_state == S.ASSEMBLING
    assert resumed.state.artifacts.media.version == (2 if expected_orphan else 1)
    assert resumed.state.artifacts.approved_storyboard == ref("storyboard")


def test_crash_after_manifest_publication_is_orphan_not_adopted(prepared, monkeypatch):
    import history_studio.workflow.media as module
    project, store, providers = prepared
    write = module.write_json
    def crash(path, state, **kwargs):
        if state.current_state == S.ASSEMBLING:
            raise KeyboardInterrupt("hard crash")
        return write(path, state, **kwargs)
    monkeypatch.setattr(module, "write_json", crash)
    with pytest.raises(KeyboardInterrupt):
        workflow(providers).run(project, store)
    assert read(store).current_state == S.GENERATING_MEDIA and read(store).artifacts.media is None
    orphan = store.load("media", 1, MediaPackage)
    monkeypatch.setattr(module, "write_json", write)
    assert workflow(providers).run(project, store).state.artifacts.media == ref("media", 2)
    assert store.load("media", 1, MediaPackage) == orphan


@pytest.mark.parametrize("when", ["before_save", "after_save", "bound_rerun"])
def test_final_hash_tampering_never_becomes_new_authority(prepared, monkeypatch, when):
    import history_studio.workflow.media as module
    project, store, providers = prepared
    def tamper(package):
        reference = package.visual_assets[0].asset
        (store.project_dir / "media" / reference.relative_path).write_bytes(b"tampered")
    if when == "before_save":
        visual = module.generate_visual_assets
        def generate(**kwargs):
            values = visual(**kwargs)
            path = store.project_dir / "media" / values[0].asset.relative_path
            path.write_bytes(b"tampered")
            return values
        monkeypatch.setattr(module, "generate_visual_assets", generate)
    elif when == "after_save":
        save = store.save
        def persist(kind, package):
            version = save(kind, package)
            tamper(package)
            return version
        monkeypatch.setattr(store, "save", persist)
    else:
        first = workflow(providers).run(project, store)
        tamper(first.package)
        with pytest.raises(ValueError, match="SHA-256"):
            MediaWorkflow(provider_factory=lambda board: pytest.fail("No regeneration")).run(project, store)
        return
    result = workflow(providers).run(project, store)
    assert result.state.current_state == S.FAILED and result.state.artifacts.media is None
    assert store.list_versions("media") == ([1] if when == "after_save" else [])


def test_bound_rerun_ignores_newer_manifest_and_recovers_failed_assembly(prepared):
    project, store, providers = prepared
    first = workflow(providers).run(project, store)
    newer = first.package.model_copy(update={"title": "Unbound newer manifest"})
    store.save("media", newer)
    machine = ProjectStateMachine(first.state)
    machine.fail("assembly interrupted")
    write_json(store.project_dir / ".runtime/state.json", machine.state, replace=True)
    runner = MediaWorkflow(provider_factory=lambda board: pytest.fail("No regeneration"))
    recovered = runner.run(project, store)
    assert recovered.state.current_state == S.ASSEMBLING
    assert recovered.state.artifacts.media == ref("media") and recovered.package == first.package


@pytest.mark.parametrize("failure", ["missing_approval", "wrong_state", "missing_media", "wrong_script"])
def test_invalid_authority_never_calls_factory(prepared, failure):
    project, store, _ = prepared
    data = read(store).model_dump(mode="json")
    if failure == "missing_approval":
        data["artifacts"]["approved_storyboard"] = None
    elif failure == "wrong_state":
        data.update(current_state="WAITING_STORYBOARD_APPROVAL", last_successful_state="WAITING_STORYBOARD_APPROVAL")
    elif failure == "missing_media":
        data["current_state"] = "ASSEMBLING"
    else:
        data["artifacts"]["approved_script"]["version"] = 2
    (store.project_dir / ".runtime/state.json").write_text(json.dumps(data), encoding="utf-8")
    runner = MediaWorkflow(provider_factory=lambda board: pytest.fail("No factory"))
    if failure == "wrong_script":
        assert runner.run(project, store).state.current_state == S.FAILED
    else:
        with pytest.raises(ValueError):
            runner.run(project, store)


def test_new_generation_clears_stale_media_and_typed_binding_checks(prepared):
    project, store, _ = prepared
    state = read(store)
    stale = state.artifacts.with_media(ref("media", 8))
    machine = ProjectStateMachine(RuntimeState(current_state=S.STORYBOARD_APPROVED,
        last_successful_state=S.STORYBOARD_APPROVED, artifacts=stale))
    assert machine.begin_media(project_id="project").artifacts == state.artifacts
    assert machine.complete_media(ref("media", 9), project_id="project").current_state == S.ASSEMBLING
    assert machine.state.artifacts.approved_storyboard == ref("storyboard")
    with pytest.raises(ValueError):
        ProjectStateMachine(state).complete_media(ref("media"), project_id="project")
    with pytest.raises(ValueError):
        stale.with_media(ref("script"))
    assert not hasattr(stale, "approved_media")


def test_cli_routes_to_workflow_with_config_and_lazy_client(prepared, monkeypatch):
    from history_studio.cli import main
    from history_studio.workflow.media import MediaWorkflowOutcome
    import history_studio.research.openai_provider as provider_module
    project, store, _ = prepared
    config_path = store.project_dir / "media-settings.json"
    config_path.write_text(json.dumps(dict(tts_model="tts-1", tts_voice="alloy", image_model="gpt-image-1", video_model="sora-2")))
    calls = []
    def run(self, supplied_project, supplied_store):
        calls.append(supplied_project)
        machine = ProjectStateMachine(read(supplied_store))
        machine.begin_media(project_id="project")
        machine.complete_media(ref("media"), project_id="project")
        return MediaWorkflowOutcome(state=machine.state)
    monkeypatch.setattr(MediaWorkflow, "run", run)
    monkeypatch.setattr(provider_module, "create_client", lambda *args: pytest.fail("No client creation without factory"))
    assert main(["--projects-dir", str(store.project_dir.parent), "media", "project", "--config", str(config_path)]) == 0
    assert calls == [project]


def test_cli_missing_media_configuration_fails_without_fake_or_network(prepared, monkeypatch, capsys):
    from history_studio.cli import main
    import history_studio.research.openai_provider as provider_module
    _, store, _ = prepared
    monkeypatch.setattr(provider_module, "create_client", lambda *args: pytest.fail("No client on missing config"))
    assert main(["--projects-dir", str(store.project_dir.parent), "media", "project"]) == 1
    assert "Media requires --config" in capsys.readouterr().err
    assert read(store).artifacts.media is None


@pytest.mark.parametrize("missing", ["image", "video", "tts"])
def test_factory_missing_provider_fails_before_any_generation(prepared, missing):
    project, store, providers = prepared
    values = dict(tts=providers.tts, image=providers.image, video=providers.video)
    values[missing] = None
    result = workflow(MediaProviders(**values)).run(project, store)
    assert result.state.current_state == S.FAILED
    assert providers.tts.texts == [] and providers.image.events == []
    assert result.state.artifacts.media is None


@pytest.mark.parametrize("failure", ["missing_binary", "manifest_provenance", "duration"])
def test_bound_rerun_fails_closed_on_corruption_without_regeneration(prepared, failure):
    project, store, providers = prepared
    first = workflow(providers).run(project, store)
    if failure == "missing_binary":
        (store.project_dir / "media" / first.package.narration_assets[0].asset.relative_path).unlink()
    else:
        data = first.package.model_dump(mode="json")
        if failure == "manifest_provenance":
            data["storyboard_input_ref"]["version"] = 2
        else:
            data["narration_assets"][0]["duration_seconds"] = 1
        (store.project_dir / "media/media_v1.json").write_text(json.dumps(data), encoding="utf-8")
    with pytest.raises((ValueError, FileNotFoundError)):
        MediaWorkflow(provider_factory=lambda board: pytest.fail("No regeneration")).run(project, store)


def test_exact_reloaded_manifest_must_equal_accepted_output(prepared, monkeypatch):
    project, store, providers = prepared
    load = store.load
    def different(kind, version, model):
        value = load(kind, version, model)
        return value.model_copy(update={"title": "Changed persisted identity"}) if kind == "media" else value
    monkeypatch.setattr(store, "load", different)
    result = workflow(providers).run(project, store)
    assert result.state.current_state == S.FAILED and result.state.artifacts.media is None
    assert store.list_versions("media") == [1]


def test_cli_incomplete_configuration_fails_before_client(prepared, monkeypatch, capsys):
    from history_studio.cli import main
    import history_studio.research.openai_provider as provider_module
    _, store, _ = prepared
    config = store.project_dir / "media-settings.json"
    config.write_text(json.dumps(dict(tts_model="tts-1", tts_voice="alloy")))
    monkeypatch.setattr(provider_module, "create_client", lambda *args: pytest.fail("No client before model preflight"))
    assert main(["--projects-dir", str(store.project_dir.parent), "media", "project", "--config", str(config)]) == 1
    assert "requires image_model" in capsys.readouterr().err


def test_cli_concrete_wiring_runs_only_mocked_sdk_and_closes_client(prepared, monkeypatch):
    import base64
    import httpx
    from history_studio.cli import main
    from test_media_openai_provider import sdk
    from test_visual_media import png_bytes
    from test_narration_media import wav_bytes
    import history_studio.research.openai_provider as provider_module
    project, store, _ = prepared
    version = store.save("storyboard", storyboard_package())  # Approved all-static plan.
    state = read(store)
    data = state.model_dump(mode="json")
    data["artifacts"]["approved_storyboard"] = ref("storyboard", version).model_dump(mode="json")
    write_json(store.project_dir / ".runtime/state.json", RuntimeState.model_validate(data), replace=True)
    config = store.project_dir / "media-settings.json"
    config.write_text(json.dumps(dict(tts_model="tts-1", tts_voice="alloy", image_model="gpt-image-1")))
    def respond(request):
        if request.url.path == "/v1/audio/speech":
            return httpx.Response(200, content=wav_bytes(), headers={"Content-Type": "audio/wav"})
        assert request.url.path == "/v1/images/generations"
        return httpx.Response(200, json=dict(created=1, output_format="png", data=[
            dict(b64_json=base64.b64encode(png_bytes()).decode())]))
    with sdk(respond) as (client, requests):
        monkeypatch.setattr(provider_module, "create_client", lambda configuration: client)
        assert main(["--projects-dir", str(store.project_dir.parent), "media", "project", "--config", str(config)]) == 0
        assert len(requests) == 5  # Two represented segments, three final static visuals.
        assert client.is_closed()
    assert read(store).current_state == S.ASSEMBLING and read(store).artifacts.media == ref("media")


def test_approved_storyboard_replacement_invalidates_media(prepared):
    _, store, _ = prepared
    bindings = read(store).artifacts.with_media(ref("media"))
    changed = bindings.with_storyboard(ref("storyboard", 3))
    assert changed.media is None and changed.approved_storyboard is None
    assert changed.approved_script == bindings.approved_script
