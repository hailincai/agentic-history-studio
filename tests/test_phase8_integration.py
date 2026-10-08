"""P8-Z integration closure, added for manual execution only.

All assembly services, contracts, stores, state transitions, FFmpeg and ffprobe
are real. Spies delegate to production rendering; mocks only forbid unwanted
execution or inject publication failures. No historical/demo artifacts are used.
Real-media cases skip only when either local executable is unavailable.
"""
from dataclasses import dataclass
from fractions import Fraction
import hashlib
import io
import json
import math
from pathlib import Path
import shutil
import struct
import subprocess
import wave
import zlib

import pytest
from pydantic import ValidationError

from history_studio.assembly import FFmpegRenderer, build_subtitle_cues, build_timeline_plan, serialize_srt
from history_studio.assembly.rendering import DURATION_TOLERANCE_SECONDS, FPS
from history_studio.cli import main, read_project
from history_studio.media import measure_wav_duration, validate_media_package
from history_studio.models import (
    ArtifactReference, AssemblyPackage, MediaPackage, MediaType, NarrationAsset,
    ProjectConfig, ScriptPackage, StoryboardPackage, VisualAsset,
)
from history_studio.storage import ArtifactNotFoundError, ArtifactStore, MediaIntegrityError, MediaStore
from history_studio.storage.artifact_store import write_json
from history_studio.workflow import (
    InvalidTransitionError, ProjectState as S, ProjectStateMachine, RuntimeState, WorkflowArtifactBindings,
)
from history_studio.workflow.assembly import AssemblyWorkflow


PROJECT_ID = "phase8"
FIRST_NARRATION = "Exact first narration — 李白。"
SECOND_NARRATION = "Second café line.\nExact ending."
EXPECTED_SRT = (
    f"1\n00:00:00,000 --> 00:00:00,601\n{FIRST_NARRATION}\n\n"
    f"2\n00:00:00,601 --> 00:00:01,001\n{SECOND_NARRATION}\n\n"
).encode("utf-8")


def ref(kind, version=1):
    return ArtifactReference(project_id=PROJECT_ID, artifact_type=kind, version=version)


def forbidden(*args, **kwargs):
    pytest.fail("P8-Z must not invoke providers, paid stages, artifact discovery or forbidden rerendering")


@pytest.fixture(autouse=True)
def no_paid_stages_or_sdk_clients(monkeypatch):
    """Retain conftest's socket guard and fail before accidental client creation."""
    from openai import AsyncOpenAI, OpenAI
    import history_studio.cli as cli
    import history_studio.research.openai_provider as transport

    monkeypatch.setattr(OpenAI, "__init__", forbidden)
    monkeypatch.setattr(AsyncOpenAI, "__init__", forbidden)
    monkeypatch.setattr(transport, "create_client", forbidden)
    for name in ("run_research", "run_verification", "run_story", "run_script", "run_storyboard", "run_media"):
        monkeypatch.setattr(cli, name, forbidden)


@dataclass(frozen=True)
class LocalTools:
    ffmpeg: str
    ffprobe: str


@pytest.fixture(scope="module")
def local_tools():
    located = {name: shutil.which(name) for name in ("ffmpeg", "ffprobe")}
    missing = [name for name, path in located.items() if path is None]
    if missing:
        pytest.skip("P8-Z real integration requires local ffmpeg and ffprobe on PATH; missing: " + ", ".join(missing))
    return LocalTools(**{name: str(Path(path).resolve()) for name, path in located.items()})


def local_run(args):
    """Array-only local subprocess with bounded runtime and useful diagnostics."""
    result = subprocess.run(args, shell=False, capture_output=True, timeout=60, check=False)
    assert result.returncode == 0, result.stderr.decode("utf-8", errors="replace")
    return result.stdout


def probe(tools, path):
    return json.loads(local_run([tools.ffprobe, "-v", "error", "-protocol_whitelist", "file,pipe",
                                 "-show_streams", "-show_format", "-of", "json", str(path)]))


def wav_payload(milliseconds, frequency):
    """Exact millisecond boundaries at 48 kHz; distinct narration tones."""
    rate = 48000
    frames = milliseconds * 48
    samples = b"".join(struct.pack("<h", round(12000 * math.sin(2 * math.pi * frequency * index / rate)))
                       for index in range(frames))
    buffer = io.BytesIO()
    with wave.open(buffer, "wb") as audio:
        audio.setnchannels(1)
        audio.setsampwidth(2)
        audio.setframerate(rate)
        audio.writeframes(samples)
    return buffer.getvalue()


def png_payload():
    """Valid 16x8 red PNG, whose aspect ratio exercises padding."""
    def chunk(kind, data):
        return struct.pack(">I", len(data)) + kind + data + struct.pack(">I", zlib.crc32(kind + data))
    return (b"\x89PNG\r\n\x1a\n" + chunk(b"IHDR", struct.pack(">IIBBBBB", 16, 8, 8, 2, 0, 0, 0))
            + chunk(b"IDAT", zlib.compress((b"\0" + b"\xff\0\0" * 16) * 8)) + chunk(b"IEND", b""))


@pytest.fixture(scope="module")
def synthetic_videos(local_tools, tmp_path_factory):
    """Generate each tiny source once; function fixtures store independent copies."""
    root = tmp_path_factory.mktemp("phase8-synthetic-videos")
    sources = {}
    for name, color, duration in (("long", "green", "0.9"), ("short", "blue", "0.1")):
        path = root / f"{name}.mp4"
        local_run([local_tools.ffmpeg, "-hide_banner", "-loglevel", "error", "-nostdin", "-n",
                   "-filter_threads", "1", "-filter_complex_threads", "1",
                   "-f", "lavfi", "-i", f"color=c={color}:s=160x90:r=30:d={duration}",
                   "-f", "lavfi", "-i", f"sine=frequency=3000:sample_rate=48000:duration={duration}",
                   "-map", "0:v:0", "-map", "1:a:0", "-c:v", "libx264", "-preset", "ultrafast",
                   "-threads", "1", "-pix_fmt", "yuv420p", "-c:a", "aac", str(path)])
        assert {stream["codec_type"] for stream in probe(local_tools, path)["streams"]} == {"video", "audio"}
        sources[name] = path.read_bytes()
    return sources


@pytest.fixture
def render_trace(monkeypatch):
    """Observe the real render API without substituting commands or result bytes."""
    calls = []
    original = FFmpegRenderer.render
    def recorded(self, timeline, media_package, media_store, output_path, *, media_input_ref,
                 srt_content=None, srt_path=None):
        call = dict(timeline=timeline, media_package=media_package, media_store=media_store,
                    output_path=output_path, media_input_ref=media_input_ref, srt_content=srt_content)
        calls.append(call)
        result = original(self, timeline, media_package, media_store, output_path,
                          media_input_ref=media_input_ref, srt_content=srt_content, srt_path=srt_path)
        call["result"] = result
        return result
    monkeypatch.setattr(FFmpegRenderer, "render", recorded)
    return calls


@dataclass
class AssemblyCase:
    project: ProjectConfig
    store: ArtifactStore
    media_store: MediaStore
    render_root: Path
    workflow: AssemblyWorkflow
    script: ScriptPackage
    storyboard: StoryboardPackage
    media: MediaPackage
    initial_state: RuntimeState
    configuration: Path
    calls: list

    @property
    def state_path(self):
        return self.store.project_dir / ".runtime/state.json"

    def read_state(self):
        return RuntimeState.model_validate_json(self.state_path.read_text(encoding="utf-8"))


@pytest.fixture
def assembly_case(tmp_path, local_tools, synthetic_videos, render_trace):
    """Enter ASSEMBLING using real Phase 7 completion primitives and exact refs."""
    project = ProjectConfig(project_id=PROJECT_ID, topic="Synthetic assembly integration")
    store = ArtifactStore(tmp_path / PROJECT_ID)
    script = ScriptPackage.model_validate(dict(story_input_ref=ref("story").model_dump(), title="Synthetic documentary",
        sections=[dict(section_id="opening", title="Opening", segments=[
            dict(segment_id="SEG-Z", kind="STRUCTURAL", narration=FIRST_NARRATION)]),
                  dict(section_id="ending", title="Ending", segments=[
            dict(segment_id="SEG-A", kind="STRUCTURAL", narration=SECOND_NARRATION)])]))
    assert store.save("script", script) == 1
    shot_specs = [("shot-short", "SEG-A", "IMAGE_TO_VIDEO", 1),
                  ("shot-static", "SEG-Z", "STATIC_IMAGE", 2),
                  ("shot-long", "SEG-Z", "TEXT_TO_VIDEO", 1),
                  ("shot-ending", "SEG-A", "STATIC_IMAGE", 1)]
    storyboard = StoryboardPackage.model_validate(dict(script_input_ref=ref("script").model_dump(), title=script.title,
        sections=[dict(section_id="visuals", title="Visuals", shots=[dict(
            shot_id=identifier, kind="STRUCTURAL", source_segment_id=segment, generation_method=method,
            visual_description="Synthetic color", generation_prompt="Synthetic local test input",
            framing="WIDE", camera_motion="NONE", estimated_duration_seconds=weight)
            for identifier, segment, method, weight in shot_specs])]))
    assert store.save("storyboard", storyboard) == 1
    media_store = MediaStore(store.project_dir / "media")
    narration = []
    for identifier, milliseconds, frequency in (("SEG-Z", 601, 440), ("SEG-A", 400, 880)):
        data = wav_payload(milliseconds, frequency)
        asset = media_store.save_bytes(asset_id=f"n-{identifier}", relative_path=f"audio/{identifier}.wav",
                                       media_type=MediaType.AUDIO, data=data)
        narration.append(NarrationAsset(segment_id=identifier, asset=asset, duration_seconds=measure_wav_duration(data)))
    visuals = []
    for shot in storyboard.sections[0].shots:
        image = shot.generation_method == "STATIC_IMAGE"
        data = png_payload() if image else synthetic_videos["short" if shot.shot_id == "shot-short" else "long"]
        asset = media_store.save_bytes(asset_id=f"v-{shot.shot_id}",
            relative_path=f"visuals/{shot.shot_id}.{'png' if image else 'mp4'}",
            media_type=MediaType.IMAGE if image else MediaType.VIDEO, data=data)
        visuals.append(VisualAsset(shot_id=shot.shot_id, source_segment_id=shot.source_segment_id,
                                   asset=asset, generation_method=shot.generation_method))
    media = MediaPackage(storyboard_input_ref=ref("storyboard"), title=storyboard.title,
                         narration_assets=narration, visual_assets=visuals)
    assert validate_media_package(package=media, storyboard_input_ref=ref("storyboard"), storyboard=storyboard,
                                  script_input_ref=ref("script"), script=script).is_valid
    assert store.save("media", media) == 1
    assert store.load("media", 1, MediaPackage) == media
    for item in (*media.narration_assets, *media.visual_assets):
        assert media_store.verify(item.asset)
    machine = ProjectStateMachine(RuntimeState(current_state=S.STORYBOARD_APPROVED, last_successful_state=S.STORYBOARD_APPROVED,
        artifacts=WorkflowArtifactBindings(script=ref("script"), approved_script=ref("script"),
            storyboard=ref("storyboard"), approved_storyboard=ref("storyboard"))))
    machine.begin_media(project_id=PROJECT_ID)
    initial = machine.complete_media(ref("media"), project_id=PROJECT_ID)
    write_json(store.project_dir / "project.json", project)
    write_json(store.project_dir / ".runtime/state.json", initial)
    render_root = tmp_path / "render"
    configuration = store.project_dir / "assembly.json"
    configuration.write_text(json.dumps(dict(render_root=str(render_root), ffmpeg=local_tools.ffmpeg,
        ffprobe=local_tools.ffprobe, timeout_seconds=60)), encoding="utf-8")
    workflow = AssemblyWorkflow(render_root=render_root, renderer_factory=lambda: FFmpegRenderer(render_root,
        ffmpeg=local_tools.ffmpeg, ffprobe=local_tools.ffprobe, timeout_seconds=60))
    return AssemblyCase(project, store, media_store, render_root, workflow, script, storyboard, media,
                        initial, configuration, render_trace)


def assert_upstream_unchanged(case, state):
    for name in WorkflowArtifactBindings.model_fields:
        if name != "assembly":
            assert getattr(state.artifacts, name) == getattr(case.initial_state.artifacts, name)
    assert case.store.load("media", 1, MediaPackage) == case.media
    assert case.store.load("storyboard", 1, StoryboardPackage) == case.storyboard
    assert case.store.load("script", 1, ScriptPackage) == case.script
    assert case.store.list_versions("approvals") == []


def assert_complete(case, outcome):
    assert outcome.error_type is None, outcome.error_message
    assert outcome.state.current_state == S.COMPLETE
    assert outcome.state.last_successful_state == S.COMPLETE
    assert case.read_state() == outcome.state
    binding = outcome.state.require_assembly_ref(PROJECT_ID)
    assert binding.project_id == PROJECT_ID and binding.artifact_type == "assembly"
    durable = case.store.load("assembly", binding.version, AssemblyPackage)
    assert durable == outcome.package and durable.media_input_ref == ref("media")
    assert durable.timeline_duration_ms == 1001
    assert durable.final_video.relative_path == "final/phase8/media-v1/documentary.mp4"
    assert durable.subtitles.relative_path == "final/phase8/media-v1/documentary.srt"
    outputs = MediaStore(case.render_root)
    for asset in (durable.final_video, durable.subtitles):
        data = outputs.read_bytes(asset)
        assert data and hashlib.sha256(data).hexdigest() == asset.sha256
    assert outputs.read_bytes(durable.subtitles) == EXPECTED_SRT
    assert_upstream_unchanged(case, outcome.state)
    return durable


def assert_decoded_output(case, package, tools):
    """Independent probing and full-stream decoding, not renderer metadata alone."""
    path = case.render_root / package.final_video.relative_path
    metadata = probe(tools, path)
    assert len(metadata["streams"]) == 2
    video = next(stream for stream in metadata["streams"] if stream["codec_type"] == "video")
    audio = next(stream for stream in metadata["streams"] if stream["codec_type"] == "audio")
    assert (video["width"], video["height"], video["codec_name"], video["pix_fmt"]) == (1280, 720, "h264", "yuv420p")
    assert Fraction(video["avg_frame_rate"]) == 30 and video["sample_aspect_ratio"] == "1:1"
    assert int(video["nb_frames"]) == 30
    assert audio["codec_name"] == "aac" and int(audio["sample_rate"]) == 48000
    measured = float(metadata["format"]["duration"])
    assert math.isfinite(measured)
    assert abs(measured - 1.001) <= DURATION_TOLERANCE_SECONDS
    assert measured == pytest.approx(package.measured_duration_seconds, abs=1e-6)
    assert abs(float(video["duration"]) - 1.0) <= 0.001
    assert float(audio["duration"]) >= 1.000  # No material truncation of the 1.001 s narration.
    for stream in (video, audio):
        assert abs(float(stream["start_time"])) <= 0.001
    local_run([tools.ffmpeg, "-v", "error", "-nostdin", "-xerror", "-protocol_whitelist", "file,pipe",
               "-i", str(path), "-map", "0:v:0", "-map", "0:a:0", "-f", "null", "-"])
    # Decode every frame's center pixels to prove ordering, trimming and final-frame extension.
    rgb = local_run([tools.ffmpeg, "-v", "error", "-nostdin", "-i", str(path), "-map", "0:v:0", "-an",
                     "-vf", "crop=2:2:640:360,format=rgb24", "-f", "rawvideo", "pipe:1"])
    assert len(rgb) == 30 * 2 * 2 * 3
    for frame, dominant in ((0, 0), (11, 0), (12, 1), (17, 1), (18, 2), (23, 2), (24, 0), (29, 0)):
        color = tuple(rgb[frame * 12:frame * 12 + 3])
        assert all(color[dominant] > color[other] + 50 for other in range(3) if other != dominant)
    # Narration tone order and absence of the source videos' 3 kHz audio.
    raw = local_run([tools.ffmpeg, "-v", "error", "-nostdin", "-i", str(path), "-map", "0:a:0",
                     "-ac", "1", "-ar", "48000", "-f", "f32le", "pipe:1"])
    assert len(raw) % 4 == 0
    samples = struct.unpack(f"<{len(raw) // 4}f", raw)
    assert len(samples) >= 48000
    for start, expected in ((0.1, 440), (0.75, 880)):
        window = samples[round(start * 48000):round((start + 0.1) * 48000)]
        def power(frequency):
            real = sum(value * math.cos(2 * math.pi * frequency * index / 48000) for index, value in enumerate(window))
            imaginary = sum(value * math.sin(2 * math.pi * frequency * index / 48000) for index, value in enumerate(window))
            return real * real + imaginary * imaginary
        assert power(expected) > 1
        assert power(expected) > 100 * power(3000)
        assert power(expected) > 100 * power(880 if expected == 440 else 440)


def test_real_assembly_complete_exact_timeline_srt_hashes_and_decoding(assembly_case, local_tools):
    case = assembly_case
    outcome = case.workflow.run(case.project, case.store, media_store=case.media_store)
    package = assert_complete(case, outcome)
    assert outcome.state.artifacts.assembly == ref("assembly", 1)
    assert case.store.list_versions("assembly") == [1]
    assert len(case.calls) == 1
    call = case.calls[0]
    plan = call["timeline"]
    assert (plan.media_input_ref, plan.storyboard_input_ref, plan.script_input_ref) == (ref("media"), ref("storyboard"), ref("script"))
    assert call["media_input_ref"] == ref("media") and call["media_package"] == case.media
    assert call["srt_content"] == EXPECTED_SRT
    assert [(segment.segment_id, segment.start_ms, segment.end_ms) for segment in plan.segments] == [
        ("SEG-Z", 0, 601), ("SEG-A", 601, 1001)]
    assert [(shot.shot_id, shot.start_ms, shot.end_ms) for shot in plan.shots] == [
        ("shot-static", 0, 401), ("shot-long", 401, 601),
        ("shot-short", 601, 801), ("shot-ending", 801, 1001)]
    assert [shot.end_ms - shot.start_ms for shot in plan.shots] == [401, 200, 200, 200]
    assert plan.total_duration_ms == sum(segment.end_ms - segment.start_ms for segment in plan.segments) == 1001
    for segment, narration in zip(plan.segments, case.media.narration_assets):
        assert narration.asset == segment.narration_asset
        with wave.open(io.BytesIO(case.media_store.read_bytes(narration.asset)), "rb") as audio:
            assert audio.getnframes() * 1000 / audio.getframerate() == segment.end_ms - segment.start_ms
        selected = [shot for shot in plan.shots if shot.source_segment_id == segment.segment_id]
        assert selected[0].start_ms == segment.start_ms and selected[-1].end_ms == segment.end_ms
        assert sum(shot.end_ms - shot.start_ms for shot in selected) == segment.end_ms - segment.start_ms
    assert case.calls[0]["result"].frame_count == FPS
    assert package.ffmpeg_version == case.calls[0]["result"].ffmpeg_version
    assert_decoded_output(case, package, local_tools)


def test_complete_restart_reloads_exact_binding_without_render_or_new_version(assembly_case, monkeypatch):
    case = assembly_case
    first = case.workflow.run(case.project, case.store)
    package = assert_complete(case, first)
    before = case.state_path.read_bytes()
    versions = case.store.list_versions("assembly")
    project, persisted = read_project(case.store.project_dir)
    assert persisted.artifacts.assembly == ref("assembly", 1)
    reloaded_store = ArtifactStore(case.store.project_dir)
    assert reloaded_store.load("assembly", persisted.artifacts.assembly.version, AssemblyPackage) == package
    monkeypatch.setattr(subprocess, "run", forbidden)
    monkeypatch.setattr(reloaded_store, "save", forbidden)
    monkeypatch.setattr(reloaded_store, "load_latest", forbidden)
    monkeypatch.setattr(reloaded_store, "list_versions", forbidden)
    restarted = AssemblyWorkflow(render_root=case.render_root, renderer_factory=forbidden).run(project, reloaded_store)
    assert restarted == first
    assert case.state_path.read_bytes() == before and len(case.calls) == 1
    assert case.store.list_versions("assembly") == versions
    assert_complete(case, restarted)


def test_valid_unbound_render_and_manifest_are_not_adopted(assembly_case, local_tools, monkeypatch):
    case = assembly_case
    plan = build_timeline_plan(media_input_ref=ref("media"), package=case.media,
        storyboard_input_ref=ref("storyboard"), storyboard=case.storyboard,
        script_input_ref=ref("script"), script=case.script)
    srt = serialize_srt(build_subtitle_cues(plan=plan, script_input_ref=ref("script"), script=case.script))
    orphan_result = FFmpegRenderer(case.render_root, ffmpeg=local_tools.ffmpeg, ffprobe=local_tools.ffprobe,
                                   timeout_seconds=60).render(plan, case.media, case.media_store, Path("orphans/documentary.mp4"),
                                                               media_input_ref=ref("media"), srt_content=srt)
    outputs = MediaStore(case.render_root)
    orphan_video = outputs.save_bytes(asset_id="orphan-video", relative_path="orphans/documentary.mp4",
                                      media_type=MediaType.VIDEO, data=orphan_result.output_path.read_bytes())
    orphan_srt = outputs.save_bytes(asset_id="orphan-srt", relative_path="orphans/documentary.srt",
                                    media_type=MediaType.SUBTITLE, data=orphan_result.subtitle_path.read_bytes())
    orphan = AssemblyPackage(media_input_ref=ref("media"), final_video=orphan_video, subtitles=orphan_srt,
        timeline_duration_ms=plan.total_duration_ms, measured_duration_seconds=orphan_result.duration_seconds,
        ffmpeg_version=orphan_result.ffmpeg_version)
    assert case.store.save("assembly", orphan) == 1
    assert case.read_state().artifacts.assembly is None
    original_load = case.store.load
    loaded_assemblies = []
    def exact_load(kind, version, model):
        if kind == "assembly":
            loaded_assemblies.append(version)
            assert version != 1, "An unbound orphan manifest must not be read as authority"
        return original_load(kind, version, model)
    monkeypatch.setattr(case.store, "load", exact_load)
    monkeypatch.setattr(case.store, "load_latest", forbidden)
    outcome = case.workflow.run(case.project, case.store)
    package = assert_complete(case, outcome)
    assert outcome.state.artifacts.assembly == ref("assembly", 2)
    assert loaded_assemblies and set(loaded_assemblies) == {2}
    assert len(case.calls) == 2 and case.calls[-1]["output_path"] == Path("final/phase8/media-v1/documentary.mp4")
    assert package.final_video != orphan.final_video and package.subtitles != orphan.subtitles
    assert outputs.verify(orphan_video) and outputs.verify(orphan_srt)
    assert case.store.list_versions("assembly") == [1, 2]


def test_newer_unapproved_upstream_versions_cannot_change_real_assembly(assembly_case, monkeypatch):
    case = assembly_case
    new_script = case.script.model_dump(mode="json")
    new_script["sections"][0]["segments"][0]["narration"] = "UNRELATED newer Script narration"
    assert case.store.save("script", ScriptPackage.model_validate(new_script)) == 2
    new_board = case.storyboard.model_dump(mode="json")
    new_board["script_input_ref"]["version"] = 2
    new_board["sections"][0]["shots"].reverse()
    assert case.store.save("storyboard", StoryboardPackage.model_validate(new_board)) == 2
    assert case.store.save("media", case.media.model_copy(update={"storyboard_input_ref": ref("storyboard", 2)})) == 2
    original_load, original_versions = case.store.load, case.store.list_versions
    def exact_load(kind, version, model):
        if kind in ("media", "storyboard", "script"):
            assert version == 1
        return original_load(kind, version, model)
    def no_upstream_scan(kind):
        assert kind not in ("media", "storyboard", "script"), "Upstream versions must not be discovered"
        return original_versions(kind)
    monkeypatch.setattr(case.store, "load", exact_load)
    monkeypatch.setattr(case.store, "list_versions", no_upstream_scan)
    monkeypatch.setattr(case.store, "load_latest", forbidden)
    outcome = case.workflow.run(case.project, case.store)
    package = assert_complete(case, outcome)
    assert case.calls[0]["timeline"].script_input_ref == ref("script", 1)
    assert case.calls[0]["timeline"].storyboard_input_ref == ref("storyboard", 1)
    assert MediaStore(case.render_root).read_bytes(package.subtitles) == EXPECTED_SRT
    assert b"UNRELATED" not in case.calls[0]["srt_content"]


@pytest.mark.parametrize("field", ["final_video", "subtitles"])
def test_bound_output_tampering_fails_without_repair_or_alternate_binding(assembly_case, monkeypatch, field):
    case = assembly_case
    first = case.workflow.run(case.project, case.store)
    package = assert_complete(case, first)
    # A valid alternate orphan cannot rescue corruption of the bound output.
    outputs = MediaStore(case.render_root)
    orphan_video = outputs.save_bytes(asset_id="alternate-video", relative_path="orphans/alternate.mp4",
        media_type=MediaType.VIDEO, data=outputs.read_bytes(package.final_video))
    orphan_srt = outputs.save_bytes(asset_id="alternate-srt", relative_path="orphans/alternate.srt",
        media_type=MediaType.SUBTITLE, data=outputs.read_bytes(package.subtitles))
    assert case.store.save("assembly", package.model_copy(update={"final_video": orphan_video, "subtitles": orphan_srt})) == 2
    path = case.render_root / getattr(package, field).relative_path
    path.write_bytes(b"tampered exact bound output")
    before = case.state_path.read_bytes()
    monkeypatch.setattr(subprocess, "run", forbidden)
    monkeypatch.setattr(case.store, "load_latest", forbidden)
    monkeypatch.setattr(case.store, "save", forbidden)
    with pytest.raises(MediaIntegrityError, match="SHA-256"):
        AssemblyWorkflow(render_root=case.render_root, renderer_factory=forbidden).run(case.project, case.store)
    assert case.state_path.read_bytes() == before and case.read_state().artifacts.assembly == ref("assembly", 1)
    assert len(case.calls) == 1 and case.store.list_versions("assembly") == [1, 2]
    assert path.read_bytes() == b"tampered exact bound output"
    assert outputs.verify(orphan_video) and outputs.verify(orphan_srt)


@pytest.mark.parametrize("kind", [MediaType.AUDIO, MediaType.IMAGE, MediaType.VIDEO])
def test_bound_p7_binary_tampering_rejected_before_real_renderer(assembly_case, monkeypatch, kind):
    case = assembly_case
    assets = [item.asset for item in (*case.media.narration_assets, *case.media.visual_assets)]
    asset = next(asset for asset in assets if asset.media_type == kind)
    path = case.media_store.root / asset.relative_path
    original = path.read_bytes()
    path.write_bytes(original + b"tamper")
    monkeypatch.setattr(subprocess, "run", forbidden)
    monkeypatch.setattr(case.store, "load_latest", forbidden)
    outcome = AssemblyWorkflow(render_root=case.render_root, renderer_factory=forbidden).run(case.project, case.store)
    assert outcome.state.current_state == S.FAILED and outcome.state.failed_state == S.ASSEMBLING
    assert outcome.error_type == "MediaIntegrityError"
    assert outcome.state.artifacts.assembly is None and outcome.state.artifacts.media == ref("media")
    assert case.read_state() == outcome.state and not case.calls
    assert case.store.list_versions("assembly") == []
    assert not list(case.render_root.rglob("*.mp4"))
    assert path.read_bytes() == original + b"tamper"
    assert_upstream_unchanged(case, outcome.state)


def fail_after_real_render_before_manifest(case, monkeypatch):
    """Fault injection changes only the assembly publication boundary."""
    original_save = case.store.save
    candidates = []
    def interrupted(kind, package):
        if kind == "assembly":
            assert len(case.calls) == 1 and "result" in case.calls[0]
            outputs = MediaStore(case.render_root)
            assert outputs.verify(package.final_video) and outputs.verify(package.subtitles)
            assert case.read_state().current_state == S.ASSEMBLING
            assert case.read_state().artifacts.assembly is None
            candidates.append(package)
            raise OSError("injected failure after real rendering, before durable assembly publication")
        return original_save(kind, package)
    with monkeypatch.context() as injection:
        injection.setattr(case.store, "save", interrupted)
        outcome = case.workflow.run(case.project, case.store)
    assert outcome.state.current_state == S.FAILED and outcome.state.failed_state == S.ASSEMBLING
    assert outcome.error_type == "OSError" and outcome.package is None
    assert outcome.state.artifacts.assembly is None and outcome.state.artifacts.media == ref("media")
    assert case.read_state() == outcome.state and case.store.list_versions("assembly") == []
    assert_upstream_unchanged(case, outcome.state)
    return candidates[0]


def test_real_publication_failure_cli_recovery_renders_again_without_adopting_orphans(assembly_case, monkeypatch):
    case = assembly_case
    candidate = fail_after_real_render_before_manifest(case, monkeypatch)
    before = case.state_path.read_bytes()
    paths = [case.render_root / asset.relative_path for asset in (candidate.final_video, candidate.subtitles)]
    identities = [(path.read_bytes(), path.stat().st_mtime_ns) for path in paths]
    # Crash residue can include a complete-looking but unbound manifest.
    assert case.store.save("assembly", candidate) == 1
    original_load = ArtifactStore.load
    loaded_assemblies = []
    def exact_load(self, kind, version, model):
        if self.project_dir == case.store.project_dir and kind == "assembly":
            loaded_assemblies.append(version)
            assert version != 1, "Recovery must not adopt an orphan manifest"
        return original_load(self, kind, version, model)
    monkeypatch.setattr(ArtifactStore, "load", exact_load)
    monkeypatch.setattr(ArtifactStore, "load_latest", forbidden)
    arguments = ["--projects-dir", str(case.store.project_dir.parent), "assemble", PROJECT_ID,
                 "--config", str(case.configuration)]
    assert main(arguments) == 1  # Failed-stage retry is not implicitly authorized.
    assert case.state_path.read_bytes() == before and len(case.calls) == 1
    assert main([*arguments, "--recover"]) == 0
    state = case.read_state()
    assert state.current_state == S.COMPLETE and state.artifacts.assembly == ref("assembly", 2)
    durable = case.store.load("assembly", 2, AssemblyPackage)
    assert durable == candidate and durable.media_input_ref == ref("media")
    assert len(case.calls) == 2
    assert case.calls[0]["timeline"] == case.calls[1]["timeline"]
    assert case.calls[0]["output_path"] == case.calls[1]["output_path"]
    assert loaded_assemblies and set(loaded_assemblies) == {2}
    assert case.store.list_versions("assembly") == [1, 2]
    for path, identity in zip(paths, identities):
        assert (path.read_bytes(), path.stat().st_mtime_ns) == identity
    assert_upstream_unchanged(case, state)


def test_real_recovery_conflict_preserves_published_bytes_and_media_authority(assembly_case, synthetic_videos, monkeypatch):
    case = assembly_case
    candidate = fail_after_real_render_before_manifest(case, monkeypatch)
    path = case.render_root / candidate.final_video.relative_path
    competing = synthetic_videos["long"]  # Valid MP4, different immutable publication bytes.
    path.write_bytes(competing)
    srt_path = case.render_root / candidate.subtitles.relative_path
    subtitle_bytes = srt_path.read_bytes()
    with pytest.raises(InvalidTransitionError, match="explicit recovery"):
        case.workflow.run(case.project, case.store)
    assert len(case.calls) == 1
    outcome = case.workflow.run(case.project, case.store, recover=True)
    assert outcome.state.current_state == S.FAILED and outcome.state.failed_state == S.ASSEMBLING
    assert outcome.error_type == "RenderConflictError" and outcome.error_message
    assert outcome.state.artifacts.media == ref("media") and outcome.state.artifacts.assembly is None
    assert case.read_state() == outcome.state and case.store.list_versions("assembly") == []
    assert path.read_bytes() == competing and srt_path.read_bytes() == subtitle_bytes
    assert len(case.calls) == 2 and "result" not in case.calls[-1]
    assert not list(case.render_root.glob(".render-*"))
    assert_upstream_unchanged(case, outcome.state)


def test_complete_with_missing_exact_manifest_cannot_fall_back_to_valid_orphan(assembly_case, monkeypatch):
    case = assembly_case
    outcome = case.workflow.run(case.project, case.store)
    package = assert_complete(case, outcome)
    assert case.store.save("assembly", package) == 2
    (case.store.project_dir / "assembly/assembly_v1.json").unlink()
    before = case.state_path.read_bytes()
    monkeypatch.setattr(subprocess, "run", forbidden)
    monkeypatch.setattr(case.store, "load_latest", forbidden)
    with pytest.raises(ArtifactNotFoundError, match="assembly version 1"):
        AssemblyWorkflow(render_root=case.render_root, renderer_factory=forbidden).run(case.project, case.store)
    assert case.state_path.read_bytes() == before and case.read_state().artifacts.assembly == ref("assembly", 1)
    assert case.store.list_versions("assembly") == [2] and len(case.calls) == 1
    outputs = MediaStore(case.render_root)
    assert outputs.verify(package.final_video) and outputs.verify(package.subtitles)


def test_complete_requires_exact_assembly_binding_and_explicit_completion_primitive():
    """This state-authority check also runs when local FFmpeg is unavailable."""
    active = RuntimeState(current_state=S.ASSEMBLING, last_successful_state=S.STORYBOARD_APPROVED,
                          artifacts=WorkflowArtifactBindings(media=ref("media")))
    machine = ProjectStateMachine(active)
    with pytest.raises(InvalidTransitionError):
        machine.transition(S.COMPLETE)
    assert machine.state == active
    for bindings in (WorkflowArtifactBindings(), WorkflowArtifactBindings(media=ref("media"))):
        with pytest.raises(ValidationError, match="exact media and assembly bindings"):
            RuntimeState(current_state=S.COMPLETE, last_successful_state=S.COMPLETE, artifacts=bindings)
    with pytest.raises(ValidationError, match="assembly artifact"):
        WorkflowArtifactBindings(media=ref("media"), assembly=ref("media"))
    assert "WAITING_MEDIA_APPROVAL" not in S.__members__
