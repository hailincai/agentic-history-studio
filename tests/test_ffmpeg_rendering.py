"""P8-B command tests and optional, local-only real FFmpeg integration."""
import hashlib
import io
import json
import math
import os
from pathlib import Path
import shutil
import struct
import subprocess
from types import SimpleNamespace
import wave
import zlib

import pytest

from history_studio.assembly import FFmpegRenderer, RenderConflictError, RenderError, serialize_srt
from history_studio.assembly.rendering import DURATION_TOLERANCE_SECONDS, FPS, frame_boundary
from history_studio.models import (
    ArtifactReference, MediaPackage, NarrationAsset, SegmentTiming, ShotTiming, SubtitleCue, TimelinePlan, VisualAsset,
)
from history_studio.storage import MediaIntegrityError, MediaStore


def reference(kind, version=1):
    return ArtifactReference(project_id="project", artifact_type=kind, version=version)


def wav(milliseconds, frequency=440, *, rate=48000, channels=1):
    frames = round(milliseconds * rate / 1000)
    samples = b"".join(struct.pack("<h", round(12000 * math.sin(2 * math.pi * frequency * index / rate))) * channels
                       for index in range(frames))
    output = io.BytesIO()
    with wave.open(output, "wb") as audio:
        audio.setnchannels(channels)
        audio.setsampwidth(2)
        audio.setframerate(rate)
        audio.writeframes(samples)
    return output.getvalue()


def png(color=(255, 0, 0)):
    def chunk(kind, data):
        return struct.pack(">I", len(data)) + kind + data + struct.pack(">I", zlib.crc32(kind + data))
    return (b"\x89PNG\r\n\x1a\n" + chunk(b"IHDR", struct.pack(">IIBBBBB", 16, 8, 8, 2, 0, 0, 0))
            + chunk(b"IDAT", zlib.compress((b"\0" + bytes(color) * 16) * 8)) + chunk(b"IEND", b""))


def inputs(tmp_path, *, durations=(401, 433, 367), methods=("STATIC_IMAGE", "TEXT_TO_VIDEO", "IMAGE_TO_VIDEO"), video_bytes=None):
    store = MediaStore(tmp_path / "media")
    visuals, timings = [], []
    offset = 0
    for index, (duration, method) in enumerate(zip(durations, methods)):
        segment_id = "first" if index == 0 else "second"
        image = method == "STATIC_IMAGE"
        data = png() if image else (video_bytes[index] if video_bytes else b"synthetic-mp4-test-double")
        asset = store.save_bytes(asset_id=f"visual-{index}", relative_path=f"visual/{index}.{'png' if image else 'mp4'}",
                                 media_type="IMAGE" if image else "VIDEO", data=data)
        visuals.append(VisualAsset(shot_id=f"shot-{index}", source_segment_id=segment_id, asset=asset, generation_method=method))
        timings.append(ShotTiming(shot_id=f"shot-{index}", source_segment_id=segment_id, visual_asset=asset,
                                  start_ms=offset, end_ms=offset + duration))
        offset += duration
    narration, segments = [], []
    boundaries = [("first", 0, durations[0])]
    if len(durations) > 1:
        boundaries.append(("second", durations[0], offset))
    for index, (identifier, start, end) in enumerate(boundaries):
        asset = store.save_bytes(asset_id=f"narration-{index}", relative_path=f"audio/{index}.wav", media_type="AUDIO",
                                 data=wav(end - start, frequency=440 if index == 0 else 880))
        narration.append(NarrationAsset(segment_id=identifier, asset=asset, duration_seconds=(end - start) / 1000))
        segments.append(SegmentTiming(segment_id=identifier, start_ms=start, end_ms=end, narration_asset=asset))
    package = MediaPackage(storyboard_input_ref=reference("storyboard"), title="Synthetic documentary",
                           narration_assets=narration, visual_assets=visuals)
    timeline = TimelinePlan(media_input_ref=reference("media"), storyboard_input_ref=reference("storyboard"),
                            script_input_ref=reference("script"), segments=segments, shots=timings, total_duration_ms=offset)
    return dict(timeline=timeline, media_package=package, media_store=store, output_path=Path("final/documentary.mp4"),
                media_input_ref=reference("media"))


class FakeFFmpeg:
    """Mock subprocess boundary; real PCM and publication filesystem behavior."""

    def __init__(self):
        self.calls = []
        self.output_duration_change = 0
        self.audio_duration_change = 0
        self.bad_codec = False
        self.failure = None
        self.frame_counts = []
        self.manifests = []

    def __call__(self, args, **kwargs):
        self.calls.append((args, kwargs))
        assert isinstance(args, list) and kwargs["shell"] is False and kwargs["check"] is True
        assert kwargs["timeout"] > 0
        if self.failure and self.failure in args:
            raise subprocess.CalledProcessError(1, args, stderr="synthetic decoder failure")
        if "-version" in args:
            return SimpleNamespace(stdout="ffmpeg version local-test-build\n")
        if "-show_streams" in args:
            path = Path(args[-1])
            if path.name == "completed.mp4":
                audio_path = path.parent / "narration.wav"
                with wave.open(str(audio_path), "rb") as audio:
                    audio_duration = audio.getnframes() / audio.getframerate()
                frames = sum(self.frame_counts[-len(list(path.parent.glob('shot-*.mp4'))):])
                streams = [dict(codec_type="video", codec_name="wrong" if self.bad_codec else "h264", width=1280, height=720,
                                pix_fmt="yuv420p", sample_aspect_ratio="1:1", avg_frame_rate="30/1", nb_frames=str(frames),
                                start_time="0.000000", duration=str(frames / FPS)),
                           dict(codec_type="audio", codec_name="aac", sample_rate="48000", channels=2,
                                start_time="0.000000", duration=str(audio_duration + self.audio_duration_change))]
                data = dict(streams=streams, format=dict(duration=str(max(frames / FPS, audio_duration) + self.output_duration_change)))
            else:
                data = dict(streams=[dict(codec_type="video", codec_name="png" if path.suffix == ".png" else "h264")],
                            format=dict(duration="0.1" if "000004" in path.name else "2.0"))
            return SimpleNamespace(stdout=json.dumps(data))
        work = Path(kwargs["cwd"])
        output = work / args[-1]
        if "-frames:v" in args:
            self.frame_counts.append(int(args[args.index("-frames:v") + 1]))
            output.write_bytes(b"normalized-shot")
        elif output.name.startswith("narration-"):
            source = args[args.index("-i") + 1]
            with wave.open(source, "rb") as audio:
                duration_ms = audio.getnframes() / audio.getframerate() * 1000
            output.write_bytes(wav(duration_ms, channels=2))
        elif output.name == "narration.wav":
            names = (work / "narration.txt").read_text().splitlines()
            self.manifests.append(names)
            duration = 0
            for name in names:
                with wave.open(str(work / name.split("'")[1]), "rb") as audio:
                    duration += audio.getnframes() / audio.getframerate() * 1000
            output.write_bytes(wav(duration, channels=2))
        elif output.name == "completed.mp4":
            self.manifests.append((work / "visuals.txt").read_text().splitlines())
            output.write_bytes(b"deterministic-rendered-MP4")
        else:
            raise AssertionError(f"Unexpected command: {args}")
        return SimpleNamespace(stdout="")


@pytest.fixture
def fake(monkeypatch):
    runner = FakeFFmpeg()
    monkeypatch.setattr(shutil, "which", lambda name: str(Path(name).absolute()))
    monkeypatch.setattr(subprocess, "run", runner)
    return runner


def test_mixed_commands_absolute_frame_grid_audio_order_and_publication(tmp_path, fake):
    values = inputs(tmp_path)
    result = FFmpegRenderer(tmp_path / "render").render(**values)
    assert fake.frame_counts == [12, 13, 11]
    assert result.frame_count == 36 == frame_boundary(1201)
    assert result.sha256 == hashlib.sha256(result.output_path.read_bytes()).hexdigest()
    assert result.duration_seconds == 1.201
    assert not list((tmp_path / "render").glob(".render-*"))
    clips = [args for args, _ in fake.calls if "-frames:v" in args]
    assert "-loop" in clips[0] and "-framerate" in clips[0]
    for args in clips:
        assert args[args.index("-map") + 1] == "0:v:0"
        assert "-an" in args
        filters = args[args.index("-vf") + 1]
        assert "scale=1280:720" in filters and "pad=1280:720" in filters
        assert "fps=30" in filters and "format=yuv420p" in filters and "setsar=1" in filters
        assert "tpad=stop_mode=clone" in filters
        assert args[args.index("-c:v") + 1] == "libx264"
    final = [args for args, _ in fake.calls if args[-1] == "completed.mp4"][0]
    assert [final[i + 1] for i, arg in enumerate(final) if arg == "-map"] == ["0:v:0", "1:a:0"]
    assert final[final.index("-c:a") + 1] == "aac"
    assert final[final.index("-bsf:v") + 1] == "setts=pts=N*1000:dts=N*1000:duration=1000:time_base=1/30000"
    assert "-shortest" not in final and "-t" not in final
    assert fake.manifests[0] == ["file 'narration-000000.wav'", "file 'narration-000001.wav'"]
    assert fake.manifests[1] == [f"file 'shot-{index:06d}.mp4'" for index in range(3)]


def test_absolute_boundary_quantization_does_not_accumulate():
    ends = list(range(17, 17001, 17))
    counts = [frame_boundary(end) - frame_boundary(start) for start, end in zip([0, *ends[:-1]], ends)]
    assert sum(counts) == frame_boundary(ends[-1])
    assert sum(counts) != sum(frame_boundary(17) for _ in ends)
    assert max(abs(frame_boundary(end) / FPS - end / 1000) for end in ends) <= 1 / (2 * FPS) + 1e-12


@pytest.mark.parametrize("milliseconds", [1, 16])
def test_unrepresentable_shot_fails_before_process(tmp_path, fake, milliseconds):
    values = inputs(tmp_path, durations=(milliseconds,), methods=("STATIC_IMAGE",))
    with pytest.raises(RenderError, match="no frame"):
        FFmpegRenderer(tmp_path / "render").render(**values)
    assert not fake.calls


def test_verified_snapshots_before_any_encoding(tmp_path, fake, monkeypatch):
    values = inputs(tmp_path)
    reads = []
    original = values["media_store"].read_bytes
    def read(reference):
        reads.append(reference.asset_id)
        return original(reference)
    monkeypatch.setattr(values["media_store"], "read_bytes", read)
    original_run = fake.__call__
    def run(args, **kwargs):
        if "-show_streams" in args or "-i" in args:
            assert len(reads) == 5
        return original_run(args, **kwargs)
    monkeypatch.setattr(subprocess, "run", run)
    FFmpegRenderer(tmp_path / "render").render(**values)
    assert reads == ["narration-0", "narration-1", "visual-0", "visual-1", "visual-2"]


@pytest.mark.parametrize("missing", [False, True])
def test_corrupt_or_missing_binary_rejected_without_encoding(tmp_path, fake, missing):
    values = inputs(tmp_path)
    path = values["media_store"].root / values["timeline"].shots[-1].visual_asset.relative_path
    if missing:
        path.unlink()
    else:
        path.write_bytes(b"corrupt")
    with pytest.raises((MediaIntegrityError, FileNotFoundError)):
        FFmpegRenderer(tmp_path / "render").render(**values)
    assert not any("-i" in args for args, _ in fake.calls)
    assert not list((tmp_path / "render").glob(".render-*"))


@pytest.mark.parametrize("name", ["ffmpeg", "ffprobe"])
def test_missing_tools_actionable(tmp_path, fake, monkeypatch, name):
    monkeypatch.setattr(shutil, "which", lambda candidate: None if candidate == name else candidate)
    with pytest.raises(RenderError, match=f"Missing executable {name}"):
        FFmpegRenderer(tmp_path / "render").render(**inputs(tmp_path))
    assert not any("-i" in args for args, _ in fake.calls)


@pytest.mark.parametrize("path", ["../outside.mp4", "bad.txt", "bad:stream.mp4", "CON.mp4", "./../out.mp4"])
def test_unsafe_output_paths(tmp_path, fake, path):
    values = inputs(tmp_path)
    values["output_path"] = Path(path)
    with pytest.raises(RenderError):
        FFmpegRenderer(tmp_path / "render").render(**values)
    assert not fake.calls


def test_external_absolute_output_and_srt_paths_rejected(tmp_path, fake):
    values = inputs(tmp_path)
    values["output_path"] = tmp_path / "outside.mp4"
    with pytest.raises(RenderError, match="inside render root"):
        FFmpegRenderer(tmp_path / "render").render(**values)
    values["output_path"] = Path("inside.mp4")
    with pytest.raises(RenderError, match="inside render root"):
        FFmpegRenderer(tmp_path / "render").render(**values, srt_path=tmp_path / "outside.srt")


def test_identical_output_idempotent_conflicting_output_preserved(tmp_path, fake):
    values = inputs(tmp_path)
    renderer = FFmpegRenderer(tmp_path / "render")
    first = renderer.render(**values)
    assert renderer.render(**values) == first
    first.output_path.write_bytes(b"already-published-different-output")
    with pytest.raises(RenderConflictError):
        renderer.render(**values)
    assert first.output_path.read_bytes() == b"already-published-different-output"
    assert not list(renderer.render_root.glob(".render-*"))


def subtitles(timeline):
    return serialize_srt(tuple(SubtitleCue(index=index, start_ms=segment.start_ms, end_ms=segment.end_ms,
                                          text=f"Exact narration {index} — 李白。")
                               for index, segment in enumerate(timeline.segments, 1)))


def test_precomputed_srt_bytes_and_path_remain_independent(tmp_path, fake):
    values = inputs(tmp_path)
    renderer = FFmpegRenderer(tmp_path / "render")
    content = subtitles(values["timeline"])
    result = renderer.render(**values, srt_content=content)
    assert result.subtitle_path.read_bytes() == content
    assert renderer.render(**values, srt_path=result.subtitle_path) == result
    assert not any("subtitles=" in str(args) or "mov_text" in args for args, _ in fake.calls)
    result.subtitle_path.write_bytes(b"different published SRT")
    with pytest.raises(RenderConflictError):
        renderer.render(**values, srt_content=content)
    assert result.subtitle_path.read_bytes() == b"different published SRT"


@pytest.mark.parametrize("content", [b"", b"bad SRT", b"\xff", b"1\n00:00:00,000 --> 00:00:00,999\nWrong timing\n\n"])
def test_inconsistent_srt_rejected(tmp_path, fake, content):
    with pytest.raises(RenderError, match="precomputed P8-A"):
        FFmpegRenderer(tmp_path / "render").render(**inputs(tmp_path), srt_content=content)
    assert not fake.calls


@pytest.mark.parametrize("change", ["ref", "board", "narration_order", "visual_missing", "owner", "identity", "method"])
def test_exact_manifest_and_timeline_match_required(tmp_path, fake, change):
    values = inputs(tmp_path)
    data = values["media_package"].model_dump(mode="json")
    if change == "ref":
        values["media_input_ref"] = reference("media", 2)
    elif change == "board":
        data["storyboard_input_ref"]["version"] = 2
    elif change == "narration_order":
        data["narration_assets"].reverse()
    elif change == "visual_missing":
        data["visual_assets"].pop()
    elif change == "owner":
        data["visual_assets"][0]["source_segment_id"] = "second"
    elif change == "identity":
        data["visual_assets"][0]["asset"]["sha256"] = "b" * 64
    elif change == "method":
        data["visual_assets"][0]["generation_method"] = "TEXT_TO_VIDEO"
    values["media_package"] = MediaPackage.model_validate(data)
    with pytest.raises(RenderError):
        FFmpegRenderer(tmp_path / "render").render(**values)
    assert not fake.calls


@pytest.mark.parametrize("failure", ["completed.mp4", "libx264", "narration.wav"])
def test_failed_process_no_success_or_published_output(tmp_path, fake, failure):
    fake.failure = failure
    with pytest.raises(RenderError, match="synthetic decoder failure"):
        FFmpegRenderer(tmp_path / "render").render(**inputs(tmp_path))
    assert not list((tmp_path / "render").rglob("*.mp4"))
    assert not list((tmp_path / "render").glob(".render-*"))


@pytest.mark.parametrize("change", ["duration", "codec", "truncation"])
def test_completed_probe_must_validate_before_publication(tmp_path, fake, change):
    if change == "duration":
        fake.output_duration_change = DURATION_TOLERANCE_SECONDS + 0.01
    elif change == "codec":
        fake.bad_codec = True
    else:
        fake.audio_duration_change = -0.1
    with pytest.raises(RenderError):
        FFmpegRenderer(tmp_path / "render").render(**inputs(tmp_path))
    assert not list((tmp_path / "render").rglob("*.mp4"))


def test_manifest_duration_mismatch_and_unsupported_pcm_rejected(tmp_path, fake):
    values = inputs(tmp_path)
    narration = values["media_package"].narration_assets
    values["media_package"] = values["media_package"].model_copy(update={"narration_assets": (
        narration[0].model_copy(update={"duration_seconds": 0.5}), narration[1])})
    with pytest.raises(RenderError, match="measured WAV"):
        FFmpegRenderer(tmp_path / "render").render(**values)
    assert not any("-i" in args for args, _ in fake.calls)


def replace_narration(values, index, data, measured_duration):
    original = values["timeline"].segments[index].narration_asset
    replacement = values["media_store"].save_bytes(asset_id=original.asset_id, relative_path=f"audio/replacement-{index}.wav",
                                                    media_type="AUDIO", data=data)
    manifest = values["media_package"].model_dump(mode="json")
    manifest["narration_assets"][index].update(asset=replacement.model_dump(mode="json"), duration_seconds=measured_duration)
    plan = values["timeline"].model_dump(mode="json")
    plan["segments"][index]["narration_asset"] = replacement.model_dump(mode="json")
    values["media_package"] = MediaPackage.model_validate(manifest)
    values["timeline"] = TimelinePlan.model_validate(plan)


def test_unsupported_narration_is_not_replaced_by_silence(tmp_path, fake):
    values = inputs(tmp_path)
    replace_narration(values, 0, b"not a WAV", 0.401)
    with pytest.raises(ValueError, match="PCM WAV"):
        FFmpegRenderer(tmp_path / "render").render(**values)
    assert not any("-i" in args for args, _ in fake.calls)


def test_cumulative_audio_drift_fails_without_trimming_or_replanning(tmp_path, fake):
    values = inputs(tmp_path)
    replace_narration(values, 0, wav(401 + 40 / 48), (401 + 40 / 48) / 1000)
    replace_narration(values, 1, wav(800 + 40 / 48), (800 + 40 / 48) / 1000)
    with pytest.raises(RenderError, match="Cumulative narration timing"):
        FFmpegRenderer(tmp_path / "render").render(**values)
    assert not any("-i" in args for args, _ in fake.calls)


def test_decoders_use_verified_snapshots_if_original_changes(tmp_path, fake, monkeypatch):
    values = inputs(tmp_path)
    original = values["media_store"].read_bytes
    def read_then_change(reference):
        data = original(reference)
        (values["media_store"].root / reference.relative_path).write_bytes(b"later unrelated replacement")
        return data
    monkeypatch.setattr(values["media_store"], "read_bytes", read_then_change)
    # Source mutations cannot replace the verified PCM snapshot consumed later.
    assert FFmpegRenderer(tmp_path / "render").render(**values).output_path.exists()


@pytest.mark.parametrize("probe", ["malformed", "unsupported", "failed"])
def test_visual_probe_failure_prevents_encoding(tmp_path, fake, monkeypatch, probe):
    original_run = fake.__call__
    def run(args, **kwargs):
        if "-show_streams" in args:
            if probe == "malformed":
                return SimpleNamespace(stdout="not JSON")
            if probe == "unsupported":
                return SimpleNamespace(stdout=json.dumps(dict(streams=[dict(codec_type="audio", codec_name="aac")])))
            raise subprocess.CalledProcessError(1, args, stderr="cannot decode input")
        return original_run(args, **kwargs)
    monkeypatch.setattr(subprocess, "run", run)
    with pytest.raises(RenderError):
        FFmpegRenderer(tmp_path / "render").render(**inputs(tmp_path))
    assert not any("-i" in args for args, _ in fake.calls)


def test_publication_failure_never_returns_success(tmp_path, fake, monkeypatch):
    values = inputs(tmp_path)
    def unsupported(*args, **kwargs):
        raise OSError("hard links unavailable")
    monkeypatch.setattr(os, "link", unsupported)
    with pytest.raises(RenderError, match="filesystem supporting hard links"):
        FFmpegRenderer(tmp_path / "render").render(**values)
    assert not list((tmp_path / "render").rglob("*.mp4"))
    assert not list((tmp_path / "render").glob(".render-*"))


def test_concurrent_publication_conflict_never_overwrites(tmp_path, fake, monkeypatch):
    values = inputs(tmp_path)
    def competing_publish(source, destination):
        destination.write_bytes(b"concurrently published different MP4")
        raise FileExistsError("destination won by another publisher")
    monkeypatch.setattr(os, "link", competing_publish)
    with pytest.raises(RenderConflictError):
        FFmpegRenderer(tmp_path / "render").render(**values)
    assert (tmp_path / "render" / values["output_path"]).read_bytes() == b"concurrently published different MP4"


def test_symlink_output_directory_rejected(tmp_path, fake):
    root, outside = tmp_path / "render", tmp_path / "outside"
    root.mkdir()
    outside.mkdir()
    link = root / "linked"
    try:
        link.symlink_to(outside, target_is_directory=True)
    except OSError:
        pytest.skip("OS does not allow creating test symbolic links")
    values = inputs(tmp_path)
    values["output_path"] = Path("linked/documentary.mp4")
    with pytest.raises(RenderError, match="symbolic links or junctions"):
        FFmpegRenderer(root).render(**values)
    assert not fake.calls


def test_no_provider_agent_workflow_or_timing_reallocation_imports():
    import ast
    import inspect
    import history_studio.assembly.rendering as module
    source = inspect.getsource(module)
    imports = [node.module for node in ast.walk(ast.parse(source)) if isinstance(node, ast.ImportFrom)]
    assert not any(name and ("workflow" in name or "provider" in name or "agent" in name or "planning" in name) for name in imports)


REAL_TOOLS = bool(shutil.which("ffmpeg") and shutil.which("ffprobe"))
requires_ffmpeg = pytest.mark.skipif(not REAL_TOOLS, reason="Real FFmpeg integration requires local ffmpeg and ffprobe on PATH")


def real_run(args, *, cwd=None):
    return subprocess.run(args, shell=False, check=True, capture_output=True, cwd=cwd)


def video(tmp_path, name, duration, color):
    path = tmp_path / name
    real_run([shutil.which("ffmpeg"), "-nostdin", "-n", "-loglevel", "error",
              "-f", "lavfi", "-i", f"color=c={color}:s=160x90:r=30:d={duration}",
              "-f", "lavfi", "-i", f"sine=frequency=3000:sample_rate=48000:duration={duration}",
              "-map", "0:v:0", "-map", "1:a:0", "-c:v", "libx264", "-threads", "1", "-pix_fmt", "yuv420p",
              "-c:a", "aac", str(path)])
    return path.read_bytes()


@requires_ffmpeg
@pytest.mark.parametrize("method,duration", [("STATIC_IMAGE", None), ("TEXT_TO_VIDEO", 1.4), ("IMAGE_TO_VIDEO", 0.1)])
def test_real_static_long_video_and_short_video(tmp_path, method, duration):
    data = None if duration is None else [video(tmp_path, "source.mp4", duration, "green")]
    values = inputs(tmp_path, durations=(401,), methods=(method,), video_bytes=data)
    result = FFmpegRenderer(tmp_path / "render").render(**values)
    assert result.frame_count == 12
    assert abs(result.duration_seconds - 0.401) <= DURATION_TOLERANCE_SECONDS
    assert result.output_path.is_file()


@requires_ffmpeg
def test_real_mixed_codec_frames_source_audio_exclusion_and_narration_order(tmp_path):
    data = [None, video(tmp_path, "long.mp4", 1.4, "green"), video(tmp_path, "short.mp4", 0.1, "blue")]
    values = inputs(tmp_path, video_bytes=data)
    renderer = FFmpegRenderer(tmp_path / "render")
    result = renderer.render(**values, srt_content=subtitles(values["timeline"]))
    # A real repeated render checks byte stability for this installed local build.
    assert renderer.render(**values, srt_content=subtitles(values["timeline"])) == result
    probe = json.loads(real_run([shutil.which("ffprobe"), "-v", "error", "-show_streams", "-of", "json", str(result.output_path)]).stdout)
    visual, audio = probe["streams"]
    assert (visual["width"], visual["height"], visual["codec_name"], visual["pix_fmt"], visual["avg_frame_rate"]) == (1280, 720, "h264", "yuv420p", "30/1")
    assert int(visual["nb_frames"]) == 36 and audio["codec_name"] == "aac"
    rgb = real_run([shutil.which("ffmpeg"), "-v", "error", "-i", str(result.output_path), "-an",
                    "-vf", "crop=2:2:640:360,format=rgb24", "-f", "rawvideo", "pipe:1"]).stdout
    colors = [tuple(rgb[index * 12:index * 12 + 3]) for index in (0, 11, 12, 24, 25, 35)]
    assert all(red > green + 80 and red > blue + 80 for red, green, blue in colors[:2])
    assert all(green > red + 50 and green > blue + 50 for red, green, blue in colors[2:4])
    assert all(blue > red + 80 and blue > green + 80 for red, green, blue in colors[4:])
    raw = real_run([shutil.which("ffmpeg"), "-v", "error", "-i", str(result.output_path), "-map", "0:a:0",
                    "-ac", "1", "-ar", "48000", "-f", "f32le", "pipe:1"]).stdout
    samples = struct.unpack(f"<{len(raw) // 4}f", raw)
    for start, expected in ((0.1, 440), (0.6, 880)):
        window = samples[round(start * 48000):round((start + 0.1) * 48000)]
        def power(frequency):
            real = sum(value * math.cos(2 * math.pi * frequency * index / 48000) for index, value in enumerate(window))
            imaginary = sum(value * math.sin(2 * math.pi * frequency * index / 48000) for index, value in enumerate(window))
            return real * real + imaginary * imaginary
        assert power(expected) > 100 * power(3000)
        assert power(expected) > 100 * power(880 if expected == 440 else 440)
    assert result.subtitle_path.read_bytes() == subtitles(values["timeline"])
