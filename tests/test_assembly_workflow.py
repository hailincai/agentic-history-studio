"""Focused P8-C workflow checks: fake rendering, exact refs, real local stores."""
import hashlib
import inspect
import json
from pathlib import Path

import pytest
from pydantic import ValidationError

from history_studio.assembly import RenderError, RenderResult
from history_studio.assembly.rendering import frame_boundary
from history_studio.models import (
    ArtifactReference, AssemblyPackage, MediaAssetReference, MediaPackage, MediaType, NarrationAsset,
    ProjectConfig, ScriptPackage, StoryboardPackage, VisualAsset,
)
from history_studio.storage import ArtifactStore, MediaStore
from history_studio.storage.artifact_store import write_json
from history_studio.workflow import InvalidTransitionError, ProjectState as S, ProjectStateMachine, RuntimeState, WorkflowArtifactBindings
from history_studio.workflow.assembly import AssemblyConfiguration, AssemblyWorkflow
from test_narration_media import script_package, storyboard_package, wav_bytes
from test_visual_media import mp4_bytes, png_bytes


def ref(kind, version=1, project="project"):
    return ArtifactReference(project_id=project, artifact_type=kind, version=version)


def read(store):
    return RuntimeState.model_validate_json((store.project_dir / ".runtime/state.json").read_text(encoding="utf-8"))


def snapshot(store, state):
    write_json(store.project_dir / ".runtime/state.json", state, replace=True)


class FakeRenderer:
    """Implements the real render API and immutable publication, without FFmpeg."""
    def __init__(self, root):
        self.root = root.resolve()
        self.calls = []
        self.video = mp4_bytes()
        self.failure = None
        self.duration_delta = 0
        self.bad_hash = False

    def render(self, timeline, media_package, media_store, output_path, *, media_input_ref, srt_content):
        self.calls.append(dict(timeline=timeline, media_package=media_package, media_store=media_store,
                               output_path=output_path, media_input_ref=media_input_ref, srt_content=srt_content))
        if self.failure == "before":
            raise RenderError("synthetic rendering failed before publication")
        outputs = MediaStore(self.root)
        video = outputs.save_bytes(asset_id="fake-video", relative_path=output_path.as_posix(), media_type=MediaType.VIDEO, data=self.video)
        if self.failure == "after_video":
            raise RenderError("synthetic failure after MP4 publication")
        srt_path = output_path.with_suffix(".srt")
        outputs.save_bytes(asset_id="fake-srt", relative_path=srt_path.as_posix(), media_type=MediaType.SUBTITLE, data=srt_content)
        if self.failure == "after_srt":
            raise RenderError("synthetic failure after SRT publication")
        return RenderResult(output_path=self.root / output_path, sha256="b" * 64 if self.bad_hash else video.sha256,
            duration_seconds=timeline.total_duration_ms / 1000 + self.duration_delta,
            frame_count=frame_boundary(timeline.total_duration_ms), ffmpeg_version="fake-local-v1", subtitle_path=self.root / srt_path)


@pytest.fixture
def prepared(tmp_path):
    project = ProjectConfig(project_id="project", topic="Synthetic")
    store = ArtifactStore(tmp_path / "project")
    script = script_package()
    store.save("script", script)
    board = storyboard_package().model_dump(mode="json")
    for shot, method in zip(board["sections"][0]["shots"], ("STATIC_IMAGE", "TEXT_TO_VIDEO", "IMAGE_TO_VIDEO")):
        shot["generation_method"] = method
    board = StoryboardPackage.model_validate(board)
    store.save("storyboard", board)
    media_store = MediaStore(store.project_dir / "media")
    narration = [NarrationAsset(segment_id=identifier, duration_seconds=0.25,
        asset=media_store.save_bytes(asset_id=f"n-{identifier}", relative_path=f"audio/{identifier}.wav", media_type=MediaType.AUDIO, data=wav_bytes()))
        for identifier in ("SEG-Z", "SEG-A")]
    visuals = [VisualAsset(shot_id=shot.shot_id, source_segment_id=shot.source_segment_id, generation_method=shot.generation_method,
        asset=media_store.save_bytes(asset_id=f"v-{shot.shot_id}", relative_path=f"visual/{shot.shot_id}.{'png' if shot.generation_method == 'STATIC_IMAGE' else 'mp4'}",
            media_type="IMAGE" if shot.generation_method == "STATIC_IMAGE" else "VIDEO",
            data=png_bytes() if shot.generation_method == "STATIC_IMAGE" else mp4_bytes())) for shot in board.sections[0].shots]
    store.save("media", MediaPackage(storyboard_input_ref=ref("storyboard"), title=board.title,
                                    narration_assets=narration, visual_assets=visuals))
    state = RuntimeState(current_state=S.ASSEMBLING, last_successful_state=S.STORYBOARD_APPROVED,
        artifacts=WorkflowArtifactBindings(approved_script=ref("script"), approved_storyboard=ref("storyboard"), media=ref("media")))
    write_json(store.project_dir / "project.json", project)
    write_json(store.project_dir / ".runtime/state.json", state)
    renderer = FakeRenderer(tmp_path / "render")
    return project, store, renderer


def runner(renderer):
    return AssemblyWorkflow(render_root=renderer.root, renderer_factory=lambda: renderer)


def test_exact_inputs_timeline_srt_persist_reload_then_atomic_complete(prepared, monkeypatch):
    import history_studio.workflow.assembly as module
    project, store, renderer = prepared
    before = read(store)
    events = []
    original_save, original_load, original_write = store.save, store.load, module.write_json
    def save(kind, package):
        assert kind == "assembly" and read(store) == before
        assert renderer.calls and renderer.calls[0]["media_input_ref"] == ref("media")
        outputs = MediaStore(renderer.root)
        outputs.verify(package.final_video)
        outputs.verify(package.subtitles)
        events.append("persist")
        return original_save(kind, package)
    def load(kind, version, model):
        assert version == 1
        if kind == "assembly":
            events.append("reload")
        return original_load(kind, version, model)
    def publish(path, state, **kwargs):
        if state.current_state == S.COMPLETE:
            assert events == ["persist", "reload"]
            assert state.artifacts.assembly == ref("assembly")
            assert state.artifacts.media == ref("media")
            events.append("bind+complete")
        return original_write(path, state, **kwargs)
    monkeypatch.setattr(store, "save", save)
    monkeypatch.setattr(store, "load", load)
    monkeypatch.setattr(module, "write_json", publish)
    monkeypatch.setattr(store, "load_latest", lambda *args: pytest.fail("No latest discovery"))
    outcome = runner(renderer).run(project, store)
    assert events == ["persist", "reload", "bind+complete"]
    assert outcome.state == read(store) and outcome.state.current_state == S.COMPLETE
    assert outcome.package == original_load("assembly", 1, AssemblyPackage)
    call = renderer.calls[0]
    plan = call["timeline"]
    assert (plan.media_input_ref, plan.storyboard_input_ref, plan.script_input_ref) == (ref("media"), ref("storyboard"), ref("script"))
    assert [(segment.segment_id, segment.start_ms, segment.end_ms) for segment in plan.segments] == [
        ("SEG-Z", 0, 250), ("SEG-A", 250, 500)]
    assert [shot.shot_id for shot in plan.shots] == ["SHOT-1", "SHOT-2", "SHOT-0"]
    assert call["srt_content"] == "1\n00:00:00,000 --> 00:00:00,250\nExact first narration — 李白。\n\n2\n00:00:00,250 --> 00:00:00,500\nSecond\nline of narration.\n\n".encode("utf-8")
    assert call["output_path"] == Path("final/project/media-v1/documentary.mp4")
    assert outcome.package.timeline_duration_ms == 500 and outcome.package.measured_duration_seconds == 0.5
    assert outcome.package.subtitles.media_type == MediaType.SUBTITLE
    assert outcome.state.last_successful_state == S.COMPLETE
    assert store.list_versions("approvals") == []
    for field in WorkflowArtifactBindings.model_fields:
        if field != "assembly":
            assert getattr(outcome.state.artifacts, field) == getattr(before.artifacts, field)


def test_complete_restart_authenticates_only_exact_bound_output_without_factory_or_publication(prepared, monkeypatch):
    import history_studio.workflow.assembly as module
    project, store, renderer = prepared
    first = runner(renderer).run(project, store)
    # A valid-looking newer manifest is an orphan and cannot change authority.
    store.save("assembly", first.package)
    state_bytes = (store.project_dir / ".runtime/state.json").read_bytes()
    monkeypatch.setattr(store, "save", lambda *args: pytest.fail("No republishing COMPLETE"))
    monkeypatch.setattr(store, "list_versions", lambda *args: pytest.fail("No version discovery COMPLETE"))
    monkeypatch.setattr(store, "load_latest", lambda *args: pytest.fail("No latest"))
    monkeypatch.setattr(module, "write_json", lambda *args, **kwargs: pytest.fail("COMPLETE rerun is read-only"))
    workflow = AssemblyWorkflow(render_root=renderer.root, renderer_factory=lambda: pytest.fail("No render construction"))
    second = workflow.run(project, store)
    assert second == first and second.state.artifacts.assembly == ref("assembly", 1)
    assert len(renderer.calls) == 1 and (store.project_dir / ".runtime/state.json").read_bytes() == state_bytes


def test_newer_upstream_versions_ignored_and_not_scanned(prepared, monkeypatch):
    project, store, renderer = prepared
    script = script_package().model_dump(mode="json")
    script["sections"][0]["segments"][0]["narration"] = "Forbidden newer narration"
    store.save("script", ScriptPackage.model_validate(script))
    store.save("storyboard", storyboard_package(script_version=2))
    newer = store.load("media", 1, MediaPackage).model_copy(update={"storyboard_input_ref": ref("storyboard", 2)})
    store.save("media", newer)
    versions = store.list_versions
    monkeypatch.setattr(store, "list_versions", lambda kind: versions(kind) if kind == "assembly" else pytest.fail("No upstream discovery"))
    monkeypatch.setattr(store, "load_latest", lambda *args: pytest.fail("No latest"))
    result = runner(renderer).run(project, store)
    assert result.state.current_state == S.COMPLETE and result.package.media_input_ref == ref("media", 1)
    assert b"Forbidden" not in renderer.calls[0]["srt_content"]


def test_orphan_manifest_is_not_adopted_active_retry_uses_fixed_output_path(prepared):
    project, store, renderer = prepared
    before = read(store)
    first = runner(renderer).run(project, store)
    snapshot(store, before)  # Simulate crash before the final state snapshot.
    second = runner(renderer).run(project, store)
    assert first.package == second.package
    assert len(renderer.calls) == 2
    assert store.list_versions("assembly") == [1, 2]
    assert second.state.artifacts.assembly == ref("assembly", 2)
    assert renderer.calls[0]["output_path"] == renderer.calls[1]["output_path"]


@pytest.mark.parametrize("field,missing", [("final_video", False), ("subtitles", False), ("final_video", True), ("subtitles", True)])
def test_corrupted_or_missing_bound_outputs_fail_closed_without_repair(prepared, field, missing):
    project, store, renderer = prepared
    result = runner(renderer).run(project, store)
    artifact = getattr(result.package, field)
    path = renderer.root / artifact.relative_path
    if missing:
        path.unlink()
    else:
        path.write_bytes(b"tampered published bytes")
    state_bytes = (store.project_dir / ".runtime/state.json").read_bytes()
    with pytest.raises((ValueError, FileNotFoundError)):
        AssemblyWorkflow(render_root=renderer.root, renderer_factory=lambda: pytest.fail("No repair")).run(project, store)
    assert len(renderer.calls) == 1 and (store.project_dir / ".runtime/state.json").read_bytes() == state_bytes
    assert store.list_versions("assembly") == [1]


@pytest.mark.parametrize("change", ["video_digest", "media_ref", "duration", "subtitle_wording"])
def test_bound_manifest_tampering_rejected_even_when_newer_valid_orphan_exists(prepared, change):
    project, store, renderer = prepared
    result = runner(renderer).run(project, store)
    store.save("assembly", result.package)
    data = result.package.model_dump(mode="json")
    if change == "video_digest":
        data["final_video"]["sha256"] = "b" * 64
    elif change == "media_ref":
        data["media_input_ref"]["version"] = 2
    elif change == "duration":
        data["timeline_duration_ms"] = 501
    else:
        path = renderer.root / result.package.subtitles.relative_path
        text = path.read_bytes().replace(b"Second", b"Changed")
        path.write_bytes(text)
        data["subtitles"]["sha256"] = hashlib.sha256(text).hexdigest()
    (store.project_dir / "assembly/assembly_v1.json").write_text(json.dumps(data), encoding="utf-8")
    with pytest.raises(ValueError):
        runner(renderer).run(project, store)
    assert len(renderer.calls) == 1 and read(store).artifacts.assembly == ref("assembly")


@pytest.mark.parametrize("change", ["missing_binary", "tampered_binary", "missing_visual", "tampered_visual", "wav_duration", "provenance", "visual_missing", "wrong_owner", "missing_media"])
def test_input_integrity_fails_before_render_and_preserves_authority(prepared, change):
    project, store, renderer = prepared
    before = read(store)
    media = store.load("media", 1, MediaPackage)
    path = store.project_dir / "media" / media.narration_assets[0].asset.relative_path
    if change == "missing_binary":
        path.unlink()
    elif change == "tampered_binary":
        path.write_bytes(b"tampered WAV")
    elif change in ("missing_visual", "tampered_visual"):
        visual_path = store.project_dir / "media" / media.visual_assets[-1].asset.relative_path
        if change == "missing_visual":
            visual_path.unlink()
        else:
            visual_path.write_bytes(b"tampered video")
    elif change == "missing_media":
        (store.project_dir / "media/media_v1.json").unlink()
    else:
        data = media.model_dump(mode="json")
        if change == "wav_duration":
            data["narration_assets"][0]["duration_seconds"] = 1
        elif change == "provenance":
            data["storyboard_input_ref"]["version"] = 2
        elif change == "visual_missing":
            data["visual_assets"].pop()
        else:
            data["visual_assets"][0]["source_segment_id"] = "SEG-Z"
        (store.project_dir / "media/media_v1.json").write_text(json.dumps(data), encoding="utf-8")
    result = runner(renderer).run(project, store)
    assert result.state.current_state == S.FAILED and result.error_type
    assert result.state.failed_state == S.ASSEMBLING and result.state.last_successful_state == S.STORYBOARD_APPROVED
    assert result.state.artifacts == before.artifacts and result.state.artifacts.assembly is None
    assert not renderer.calls and store.list_versions("assembly") == []


@pytest.mark.parametrize("phase", ["before", "after_video", "after_srt"])
def test_partial_renderer_failure_never_binds_incomplete_package(prepared, phase):
    project, store, renderer = prepared
    renderer.failure = phase
    result = runner(renderer).run(project, store)
    assert result.state.current_state == S.FAILED and result.state.failed_state == S.ASSEMBLING
    assert result.state.artifacts.media == ref("media") and result.state.artifacts.assembly is None
    assert store.list_versions("assembly") == [] and result.package is None


@pytest.mark.parametrize("phase", ["persist", "reload", "changed_reload", "post_reload_hash"])
def test_persistence_reload_and_reverification_failure_cannot_complete(prepared, monkeypatch, phase):
    project, store, renderer = prepared
    load = store.load
    if phase == "persist":
        monkeypatch.setattr(store, "save", lambda *args: (_ for _ in ()).throw(OSError("disk full")))
    else:
        def failed_reload(kind, version, model):
            value = load(kind, version, model)
            if kind == "assembly":
                if phase == "reload":
                    raise OSError("reload failed")
                if phase == "changed_reload":
                    return value.model_copy(update={"media_input_ref": ref("media", 2)})
                (renderer.root / value.final_video.relative_path).write_bytes(b"corrupt after reload")
            return value
        monkeypatch.setattr(store, "load", failed_reload)
    result = runner(renderer).run(project, store)
    assert result.state.current_state == S.FAILED and result.state.artifacts.assembly is None
    assert result.state.artifacts.media == ref("media") and result.package is None
    assert read(store) == result.state


def test_retry_requires_authorization_conflicts_never_version_hop_or_overwrite(prepared):
    project, store, renderer = prepared
    renderer.failure = "after_srt"
    first = runner(renderer).run(project, store)
    path = renderer.root / renderer.calls[0]["output_path"]
    original = path.read_bytes()
    with pytest.raises(InvalidTransitionError, match="explicit recovery"):
        runner(renderer).run(project, store)
    assert read(store) == first.state and len(renderer.calls) == 1
    renderer.failure = None
    renderer.video = mp4_bytes(b"different deterministic retry bytes")
    result = runner(renderer).run(project, store, recover=True)
    assert result.error_type == "MediaAssetConflictError" and result.state.current_state == S.FAILED
    assert path.read_bytes() == original and result.state.artifacts.assembly is None
    assert store.list_versions("assembly") == []
    renderer.video = original
    completed = runner(renderer).run(project, store, recover=True)
    assert completed.state.current_state == S.COMPLETE and completed.state.artifacts.media == ref("media")


def test_failed_preflight_does_not_recover_without_authentic_inputs(prepared):
    project, store, renderer = prepared
    machine = ProjectStateMachine(read(store))
    machine.fail("assembly interrupted")
    snapshot(store, machine.state)
    before = (store.project_dir / ".runtime/state.json").read_bytes()
    (store.project_dir / "media/media_v1.json").unlink()
    result = runner(renderer).run(project, store, recover=True)
    assert result.state == machine.state and result.error_type == "ArtifactNotFoundError"
    assert not renderer.calls and (store.project_dir / ".runtime/state.json").read_bytes() == before


@pytest.mark.parametrize("bad", ["hash", "duration", "destination", "missing_srt"])
def test_invalid_renderer_result_cannot_publish_manifest(prepared, monkeypatch, bad):
    from dataclasses import replace
    project, store, renderer = prepared
    render = renderer.render
    def wrong(*args, **kwargs):
        result = render(*args, **kwargs)
        updates = {"hash": dict(sha256="b" * 64), "duration": dict(duration_seconds=9),
                   "destination": dict(output_path=renderer.root / "unrelated.mp4"), "missing_srt": dict(subtitle_path=None)}
        return replace(result, **updates[bad])
    monkeypatch.setattr(renderer, "render", wrong)
    result = runner(renderer).run(project, store)
    assert result.state.current_state == S.FAILED and result.state.artifacts.assembly is None
    assert store.list_versions("assembly") == []


def test_atomic_complete_state_failure_leaves_recoverable_orphan(prepared, monkeypatch):
    import history_studio.workflow.assembly as module
    project, store, renderer = prepared
    original = module.write_json
    def fail_complete(path, state, **kwargs):
        if state.current_state == S.COMPLETE:
            raise OSError("complete snapshot failed before replace")
        return original(path, state, **kwargs)
    monkeypatch.setattr(module, "write_json", fail_complete)
    result = runner(renderer).run(project, store)
    assert result.state.current_state == S.FAILED and result.state.artifacts.assembly is None
    assert store.list_versions("assembly") == [1]
    monkeypatch.setattr(module, "write_json", original)
    retry = runner(renderer).run(project, store, recover=True)
    assert retry.state.current_state == S.COMPLETE and retry.state.artifacts.assembly == ref("assembly", 2)


def test_error_after_atomic_commit_does_not_downgrade_complete(prepared, monkeypatch):
    import history_studio.workflow.assembly as module
    project, store, renderer = prepared
    original = module.write_json
    def post_commit(path, state, **kwargs):
        original(path, state, **kwargs)
        if state.current_state == S.COMPLETE:
            raise OSError("error after atomic replace")
    monkeypatch.setattr(module, "write_json", post_commit)
    with pytest.raises(OSError, match="after atomic replace"):
        runner(renderer).run(project, store)
    assert read(store).current_state == S.COMPLETE and read(store).artifacts.assembly == ref("assembly")
    monkeypatch.setattr(module, "write_json", lambda *args, **kwargs: pytest.fail("No rewrite"))
    assert runner(renderer).run(project, store).state.current_state == S.COMPLETE
    assert len(renderer.calls) == 1


def test_contract_roundtrip_immutable_minimal_and_subtitle_type_preserves_p7_semantics(prepared):
    project, store, renderer = prepared
    value = runner(renderer).run(project, store).package
    assert AssemblyPackage.model_validate_json(value.model_dump_json()) == value
    assert set(AssemblyPackage.model_fields) == {"schema_version", "media_input_ref", "final_video", "subtitles",
                                               "timeline_duration_ms", "measured_duration_seconds", "ffmpeg_version"}
    assert not Path(value.final_video.relative_path).is_absolute() and not Path(value.subtitles.relative_path).is_absolute()
    for contract in (value, value.final_video, value.subtitles, value.media_input_ref):
        field = next(iter(type(contract).model_fields))
        with pytest.raises(ValidationError):
            setattr(contract, field, getattr(contract, field))
    with pytest.raises(ValidationError):
        NarrationAsset(segment_id="s", asset=value.subtitles, duration_seconds=1)
    with pytest.raises(ValidationError):
        VisualAsset(shot_id="s", source_segment_id="s", asset=value.subtitles, generation_method="STATIC_IMAGE")


@pytest.mark.parametrize("update", [dict(timeline_duration_ms=0), dict(timeline_duration_ms=0.5),
                                  dict(measured_duration_seconds=float("nan")), dict(measured_duration_seconds=0),
                                  dict(media_input_ref=ref("script"))])
def test_invalid_assembly_contract(prepared, update):
    project, store, renderer = prepared
    data = runner(renderer).run(project, store).package.model_dump(mode="python") | update
    with pytest.raises(ValidationError):
        AssemblyPackage.model_validate(data)


def test_state_machine_completion_requires_media_and_explicit_assembly_binding():
    active = RuntimeState(current_state=S.ASSEMBLING, last_successful_state=S.STORYBOARD_APPROVED)
    machine = ProjectStateMachine(active)
    with pytest.raises(InvalidTransitionError):
        machine.transition(S.COMPLETE)
    with pytest.raises(ValueError, match="media binding"):
        machine.complete_assembly(ref("assembly"), project_id="project")
    with pytest.raises(ValidationError, match="exact media and assembly"):
        RuntimeState(current_state=S.COMPLETE, last_successful_state=S.COMPLETE)
    bound = active.model_copy(update={"artifacts": WorkflowArtifactBindings(media=ref("media"))})
    machine = ProjectStateMachine(bound)
    with pytest.raises(ValueError):
        machine.complete_assembly(ref("assembly", project="foreign"), project_id="project")
    assert machine.state == bound
    assert machine.complete_assembly(ref("assembly"), project_id="project").current_state == S.COMPLETE
    assert machine.state.require_assembly_ref("project") == ref("assembly")


def test_upstream_replacement_invalidates_assembly_lineage():
    bindings = WorkflowArtifactBindings(approved_storyboard=ref("storyboard"), media=ref("media"), assembly=ref("assembly"))
    assert bindings.with_media(ref("media")) == bindings
    assert bindings.with_media(ref("media", 2)).assembly is None
    assert bindings.without_media().assembly is None
    assert bindings.with_storyboard(ref("storyboard", 2)).assembly is None


def test_all_input_hashes_verified_before_planning_or_render_construction(prepared, monkeypatch):
    import history_studio.workflow.assembly as module
    project, store, renderer = prepared
    verified = set()
    read_bytes, build = MediaStore.read_bytes, module.build_timeline_plan
    media = store.load("media", 1, MediaPackage)
    expected = {item.asset.asset_id for item in (*media.narration_assets, *media.visual_assets)}
    def authenticated_read(self, asset):
        data = read_bytes(self, asset)
        if self.root == store.project_dir / "media":
            verified.add(asset.asset_id)
        return data
    def planning(**kwargs):
        assert verified == expected
        return build(**kwargs)
    monkeypatch.setattr(MediaStore, "read_bytes", authenticated_read)
    monkeypatch.setattr(module, "build_timeline_plan", planning)
    assert runner(renderer).run(project, store).state.current_state == S.COMPLETE


@pytest.mark.parametrize("change", ["foreign_script_project", "wrong_approved_script", "unknown_segment"])
def test_exact_script_chain_and_coverage_rejected(prepared, change):
    project, store, renderer = prepared
    if change == "foreign_script_project":
        data = script_package().model_dump(mode="json")
        data["story_input_ref"]["project_id"] = "foreign"
        (store.project_dir / "script/script_v1.json").write_text(json.dumps(data), encoding="utf-8")
    elif change == "wrong_approved_script":
        data = read(store).model_dump(mode="json")
        data["artifacts"]["approved_script"]["version"] = 2
        snapshot(store, RuntimeState.model_validate(data))
    else:
        data = store.load("storyboard", 1, StoryboardPackage).model_dump(mode="json")
        data["sections"][0]["shots"][0]["source_segment_id"] = "unknown"
        (store.project_dir / "storyboard/storyboard_v1.json").write_text(json.dumps(data), encoding="utf-8")
    assert runner(renderer).run(project, store).state.current_state == S.FAILED
    assert not renderer.calls and store.list_versions("assembly") == []


def test_chain_script_ref_remains_authority_when_optional_legacy_approval_binding_absent(prepared):
    project, store, renderer = prepared
    data = read(store).model_dump(mode="json")
    data["artifacts"]["approved_script"] = None
    snapshot(store, RuntimeState.model_validate(data))
    assert runner(renderer).run(project, store).state.current_state == S.COMPLETE
    assert renderer.calls[0]["timeline"].script_input_ref == ref("script")


@pytest.mark.parametrize("state", [S.CREATED, S.STORYBOARD_APPROVED, S.GENERATING_MEDIA])
def test_non_assembly_stage_never_triggers_render_or_paid_generation(prepared, state):
    project, store, renderer = prepared
    checkpoint = S.STORYBOARD_APPROVED if state == S.GENERATING_MEDIA else state
    snapshot(store, RuntimeState(current_state=state, last_successful_state=checkpoint, artifacts=read(store).artifacts))
    before = (store.project_dir / ".runtime/state.json").read_bytes()
    with pytest.raises(InvalidTransitionError):
        runner(renderer).run(project, store)
    assert not renderer.calls and (store.project_dir / ".runtime/state.json").read_bytes() == before


def test_cli_explicit_config_lazy_render_no_provider_calls_and_complete_rerun(prepared, monkeypatch, capsys):
    import history_studio.assembly as module
    from history_studio.cli import main
    project, store, renderer = prepared
    constructed = []
    def factory(root, **kwargs):
        constructed.append((root, kwargs))
        return renderer
    monkeypatch.setattr(module, "FFmpegRenderer", factory)
    configuration = store.project_dir / "assembly.json"
    configuration.write_text(json.dumps(dict(render_root=str(renderer.root), ffmpeg="explicit-ffmpeg", ffprobe="explicit-ffprobe")))
    arguments = ["--projects-dir", str(store.project_dir.parent), "assemble", "project", "--config", str(configuration)]
    assert main(arguments) == 0
    assert "assembly:v1; COMPLETE" in capsys.readouterr().out
    assert constructed[0][1] == dict(ffmpeg="explicit-ffmpeg", ffprobe="explicit-ffprobe", timeout_seconds=300)
    assert main(arguments) == 0 and len(constructed) == 1
    assert main(["--projects-dir", str(store.project_dir.parent), "status", "project"]) == 0
    assert "Bound assembly snapshot: project/assembly:v1" in capsys.readouterr().out
    with pytest.raises(SystemExit):
        main(["--projects-dir", str(store.project_dir.parent), "assemble", "project"])


def test_cli_failed_recovery_flag_is_explicit_and_resume_is_read_only(prepared, monkeypatch, capsys):
    import history_studio.assembly as module
    from history_studio.cli import main
    project, store, renderer = prepared
    renderer.failure = "before"
    monkeypatch.setattr(module, "FFmpegRenderer", lambda *args, **kwargs: renderer)
    configuration = store.project_dir / "assembly.json"
    configuration.write_text(json.dumps(dict(render_root=str(renderer.root), ffmpeg="ffmpeg", ffprobe="ffprobe")))
    arguments = ["--projects-dir", str(store.project_dir.parent), "assemble", "project", "--config", str(configuration)]
    assert main(arguments) == 1 and read(store).current_state == S.FAILED
    assert main(arguments) == 1 and len(renderer.calls) == 1
    before = (store.project_dir / ".runtime/state.json").read_bytes()
    assert main(["--projects-dir", str(store.project_dir.parent), "resume", "project"]) == 0
    assert "Use assemble" in capsys.readouterr().out
    assert (store.project_dir / ".runtime/state.json").read_bytes() == before
    renderer.failure = None
    assert main([*arguments, "--recover"]) == 0 and read(store).current_state == S.COMPLETE


@pytest.mark.parametrize("configuration", [{}, dict(render_root="", ffmpeg="f", ffprobe="p"), dict(render_root="r", ffmpeg="f")])
def test_invalid_explicit_config_rejected(configuration):
    with pytest.raises(ValidationError):
        AssemblyConfiguration.model_validate(configuration)


def test_cli_relative_config_root_and_invalid_config_are_explicit_and_lazy(prepared, monkeypatch):
    import history_studio.assembly as module
    from history_studio.cli import main
    _, store, renderer = prepared
    configuration = store.project_dir / "assembly.json"
    configuration.write_text("{}")
    arguments = ["--projects-dir", str(store.project_dir.parent), "assemble", "project", "--config", str(configuration)]
    before = (store.project_dir / ".runtime/state.json").read_bytes()
    monkeypatch.setattr(module, "FFmpegRenderer", lambda *args, **kwargs: pytest.fail("No rendering for invalid config"))
    assert main(arguments) == 1
    assert (store.project_dir / ".runtime/state.json").read_bytes() == before
    configuration.write_text(json.dumps(dict(render_root="../render", ffmpeg="local-f", ffprobe="local-p")))
    def factory(root, **kwargs):
        assert root.resolve() == renderer.root
        return renderer
    monkeypatch.setattr(module, "FFmpegRenderer", factory)
    assert main(arguments) == 0


def test_no_latest_agent_provider_human_gate_or_ffmpeg_redesign():
    import ast
    import history_studio.workflow.assembly as module
    source = inspect.getsource(module)
    imports = [node.module for node in ast.walk(ast.parse(source)) if isinstance(node, ast.ImportFrom)]
    assert not any(name and ("provider" in name or ".agent" in name or "approvals" in name) for name in imports)
    assert "load_latest" not in source
    assert "subprocess" not in source
