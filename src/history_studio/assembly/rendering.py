"""Local P8-B renderer; callers authenticate the supplied exact plan/manifest.

Absolute millisecond boundaries round to the nearest 30 fps frame (ties up).
The maximum boundary error is half a frame; no per-shot errors accumulate.
Shots that occupy no frame fail rather than disappearing. PCM narration is
never trimmed, stretched or replaced with silence. Cumulative PCM boundaries
must agree with the plan within 1 ms, including millisecond/sample rounding.
Final duration tolerance is one frame + one 48 kHz AAC packet + 1 ms (55.7 ms).
Deterministic bytes are scoped to the same FFmpeg build and platform.
"""
from dataclasses import dataclass
from fractions import Fraction
import hashlib
import json
import math
import os
from pathlib import Path
import re
import shutil
import subprocess
import tempfile

from history_studio.media.audio import measure_wav_duration
from history_studio.models import ArtifactReference, GenerationMethod, MediaAssetReference, MediaPackage, MediaType, SubtitleCue, TimelinePlan
from history_studio.storage import MediaStore

FPS = 30
SAMPLE_RATE = 48000
AUDIO_SYNC_SECONDS = 0.001
DURATION_TOLERANCE_SECONDS = 1 / FPS + 1024 / SAMPLE_RATE + AUDIO_SYNC_SECONDS


class RenderError(RuntimeError):
    """Rendering, preflight or completed output validation failed."""


class RenderConflictError(RenderError):
    """Different bytes already occupy an immutable publication path."""


@dataclass(frozen=True)
class RenderResult:
    """Service result, independent of durable artifacts and workflow state."""

    output_path: Path
    sha256: str
    duration_seconds: float
    frame_count: int
    ffmpeg_version: str
    subtitle_path: Path | None = None


def frame_boundary(milliseconds: int) -> int:
    """Nearest absolute frame, ties up; integer arithmetic only."""
    return (milliseconds * FPS + 500) // 1000


def _digest(path: Path) -> str:
    with path.open("rb") as stream:
        return hashlib.file_digest(stream, "sha256").hexdigest()


class FFmpegRenderer:
    """Explicit controlled local render root, like MediaStore's root policy.

    v1 accepts PCM WAV narration, PNG static images and MP4 video. Inputs are
    authenticated through MediaStore and snapshotted before any decoder runs.
    Only fixed demuxers, local protocols and internally built filters are used.
    Caller must not concurrently rearrange root directories; this is not an OS
    sandbox. Publications use flushed temporary files + exclusive hard links.
    """

    def __init__(self, render_root: Path, *, ffmpeg: str = "ffmpeg", ffprobe: str = "ffprobe",
                 timeout_seconds: float = 300) -> None:
        supplied = Path(render_root).absolute()
        if supplied.anchor.startswith(("\\\\", "//")):
            raise RenderError("Render root must be a local directory")
        self._reject_links(supplied)
        self.render_root = supplied.resolve()
        self.ffmpeg = ffmpeg
        self.ffprobe = ffprobe
        if not math.isfinite(timeout_seconds) or timeout_seconds <= 0:
            raise ValueError("Render timeout must be positive and finite")
        self.timeout_seconds = timeout_seconds

    @staticmethod
    def _reject_links(path: Path) -> None:
        for component in (path, *path.parents):
            if component.is_symlink() or component.is_junction():
                raise RenderError("Render paths must not contain symbolic links or junctions")

    def _output(self, value: Path, suffix: str) -> Path:
        value = Path(value)
        if value.is_absolute():
            try:
                value = value.relative_to(self.render_root)
            except ValueError as exc:
                raise RenderError("Output path must stay inside render root") from exc
        try:
            MediaAssetReference.safe_relative_path(value.as_posix())
        except ValueError as exc:
            raise RenderError("Unsafe render path") from exc
        path = self.render_root / value
        self._reject_links(path)
        if not path.resolve().is_relative_to(self.render_root) or path.suffix.lower() != suffix:
            raise RenderError(f"Render path must stay inside render root and end in {suffix}")
        return path

    def _run(self, args: list[str], *, cwd: Path | None = None) -> str:
        try:
            completed = subprocess.run(args, cwd=cwd, shell=False, check=True,
                                       capture_output=True, text=True, encoding="utf-8",
                                       errors="replace", timeout=self.timeout_seconds)
            return completed.stdout
        except (OSError, subprocess.SubprocessError) as exc:
            detail = getattr(exc, "stderr", "") or str(exc)
            raise RenderError(f"{Path(args[0]).name} failed: {detail[-2000:]}") from exc

    def _tools(self) -> tuple[str, str, str]:
        paths = []
        for name in (self.ffmpeg, self.ffprobe):
            found = shutil.which(name)
            if not found:
                raise RenderError(f"Missing executable {name}; install FFmpeg with ffprobe or supply local executable paths")
            paths.append(str(Path(found).resolve()))
        versions = [self._run([path, "-version"]) for path in paths]
        if not all(version.strip() for version in versions):
            raise RenderError("FFmpeg/ffprobe returned no version information")
        return paths[0], paths[1], versions[0].splitlines()[0]

    def _probe(self, executable: str, path: Path, *, demuxer: str = "mov") -> dict:
        raw = self._run([executable, "-v", "error", "-protocol_whitelist", "file,pipe",
                         "-f", demuxer, "-show_streams", "-show_format", "-of", "json", str(path)])
        try:
            value = json.loads(raw)
            if not isinstance(value, dict) or not isinstance(value.get("streams"), list):
                raise ValueError("Missing stream information")
            return value
        except (ValueError, TypeError) as exc:
            raise RenderError("ffprobe returned malformed output") from exc

    @staticmethod
    def _duration(value: object) -> float:
        try:
            number = float(value)
            if not math.isfinite(number) or number <= 0:
                raise ValueError("Not positive finite duration")
            return number
        except (ValueError, TypeError) as exc:
            raise RenderError("Media must have positive finite measured duration") from exc

    @staticmethod
    def _match(timeline: TimelinePlan, package: MediaPackage, media_input_ref: ArtifactReference) -> None:
        if media_input_ref != timeline.media_input_ref or package.storyboard_input_ref != timeline.storyboard_input_ref:
            raise RenderError("Rendering requires exact timeline Media/Storyboard provenance")
        if [(item.segment_id, item.narration_asset) for item in timeline.segments] != [
                (item.segment_id, item.asset) for item in package.narration_assets]:
            raise RenderError("Narration identities, coverage and order must match timeline")
        by_shot = {item.shot_id: item for item in package.visual_assets}
        if set(by_shot) != {item.shot_id for item in timeline.shots}:
            raise RenderError("Visual coverage must match timeline")
        for shot in timeline.shots:
            asset = by_shot[shot.shot_id]
            if asset.source_segment_id != shot.source_segment_id or asset.asset != shot.visual_asset:
                raise RenderError("Visual identity and ownership must match timeline")
            expected = MediaType.IMAGE if asset.generation_method == GenerationMethod.STATIC_IMAGE else MediaType.VIDEO
            if asset.asset.media_type != expected:
                raise RenderError("Final visual type must match generation method")

    @staticmethod
    def _srt(content: bytes, timeline: TimelinePlan) -> None:
        """Validate P8-A canonical SRT timings without rewriting any bytes/text."""
        try:
            text = content.decode("utf-8")
            if "\r" in text or not text.endswith("\n\n"):
                raise ValueError("Expected P8-A LF endings")
            blocks = text[:-2].split("\n\n")
            if len(blocks) != len(timeline.segments):
                raise ValueError("One cue per represented segment required")
            pattern = re.compile(r"(\d{2,}):(\d{2}):(\d{2}),(\d{3}) --> (\d{2,}):(\d{2}):(\d{2}),(\d{3})")
            for index, (block, segment) in enumerate(zip(blocks, timeline.segments), 1):
                lines = block.split("\n")
                match = pattern.fullmatch(lines[1])
                if lines[0] != str(index) or match is None or len(lines) < 3 or not "\n".join(lines[2:]).strip():
                    raise ValueError("Invalid cue")
                numbers = [int(item) for item in match.groups()]
                if any(numbers[i] >= 60 for i in (1, 2, 5, 6)):
                    raise ValueError("Invalid timestamp")
                times = [((numbers[i] * 60 + numbers[i + 1]) * 60 + numbers[i + 2]) * 1000 + numbers[i + 3]
                         for i in (0, 4)]
                if times != [segment.start_ms, segment.end_ms]:
                    raise ValueError("Cue timing differs from timeline")
                SubtitleCue(index=index, start_ms=times[0], end_ms=times[1], text="\n".join(lines[2:]))
        except (ValueError, UnicodeError, IndexError, AttributeError) as exc:
            raise RenderError("SRT must be precomputed P8-A UTF-8 content with exact segment timing") from exc

    def _publish(self, temporary: Path, destination: Path) -> None:
        destination = self._output(destination, destination.suffix.lower())
        destination.parent.mkdir(parents=True, exist_ok=True)
        destination = self._output(destination, destination.suffix.lower())
        # Windows FlushFileBuffers requires a handle opened for writing.
        with temporary.open("r+b") as stream:
            stream.flush()
            os.fsync(stream.fileno())
        try:
            os.link(temporary, destination)
        except FileExistsError:
            if _digest(destination) != _digest(temporary):
                raise RenderConflictError(f"Different immutable output already exists: {destination}") from None
        except OSError as exc:
            raise RenderError("Atomic publication requires a filesystem supporting hard links") from exc

    def _validate_output(self, probe: dict, *, frames: int, planned: float, audio_duration: float) -> float:
        streams = probe["streams"]
        videos = [stream for stream in streams if stream.get("codec_type") == "video"]
        audios = [stream for stream in streams if stream.get("codec_type") == "audio"]
        if len(streams) != 2 or len(videos) != 1 or len(audios) != 1:
            raise RenderError("Output must contain exactly one video and one narration stream")
        video, audio = videos[0], audios[0]
        try:
            correct = (video["codec_name"] == "h264" and video["width"] == 1280 and video["height"] == 720
                       and video["pix_fmt"] == "yuv420p" and video["sample_aspect_ratio"] == "1:1"
                       and Fraction(video["avg_frame_rate"]) == FPS and int(video["nb_frames"]) == frames
                       and audio["codec_name"] == "aac" and int(audio["sample_rate"]) == SAMPLE_RATE
                       and audio["channels"] == 2)
            duration = self._duration(probe["format"]["duration"])
            video_duration = self._duration(video["duration"])
            measured_audio = self._duration(audio["duration"])
            starts = [float(stream["start_time"]) for stream in (video, audio)]
        except (KeyError, ValueError, TypeError, ZeroDivisionError) as exc:
            raise RenderError("Incomplete output codec/timing metadata") from exc
        if not correct:
            raise RenderError("Output codec, dimensions, frame count/rate or audio format differ from render contract")
        if (abs(duration - planned) > DURATION_TOLERANCE_SECONDS
                or any(not math.isfinite(start) or abs(start) > AUDIO_SYNC_SECONDS for start in starts)
                or abs(video_duration - frames / FPS) > AUDIO_SYNC_SECONDS
                or measured_audio < audio_duration - AUDIO_SYNC_SECONDS
                or abs(measured_audio - audio_duration) > 1024 / SAMPLE_RATE + AUDIO_SYNC_SECONDS):
            raise RenderError("Output duration deviates from timeline or truncates narration")
        return duration

    def render(self, timeline: TimelinePlan, media_package: MediaPackage, media_store: MediaStore,
               output_path: Path, *, media_input_ref: ArtifactReference,
               srt_content: bytes | None = None, srt_path: Path | None = None) -> RenderResult:
        """Render authenticated exact inputs; never discover artifacts or overwrite.

        Optional SRT bytes are preserved independently as output_path.with_suffix
        ('.srt'). A supplied SRT path must also reside inside the render root.
        Subtitle text authentication remains the caller's P8-A responsibility.
        """
        timeline = TimelinePlan.model_validate(timeline.model_dump())
        media_package = MediaPackage.model_validate(media_package.model_dump())
        media_input_ref = ArtifactReference.model_validate(media_input_ref.model_dump())
        self._match(timeline, media_package, media_input_ref)
        destination = self._output(output_path, ".mp4")
        counts = [frame_boundary(shot.end_ms) - frame_boundary(shot.start_ms) for shot in timeline.shots]
        if any(count <= 0 for count in counts):
            raise RenderError("A shot occupies no frame on the 30 fps grid; timeline cannot be rendered faithfully")
        if srt_path is not None:
            if srt_content is not None:
                raise RenderError("Supply SRT content or path, not both")
            srt_content = self._output(srt_path, ".srt").read_bytes()
        subtitle_path = self._output(destination.with_suffix(".srt"), ".srt") if srt_content is not None else None
        if srt_content is not None:
            self._srt(srt_content, timeline)
            if subtitle_path.exists() and subtitle_path.read_bytes() != srt_content:
                raise RenderConflictError("Different immutable subtitles already exist")
        ffmpeg, ffprobe, version = self._tools()
        self.render_root.mkdir(parents=True, exist_ok=True)
        self._reject_links(self.render_root)
        with tempfile.TemporaryDirectory(prefix=".render-", dir=self.render_root) as temporary:
            work = Path(temporary)
            staged = {}
            # Read through MediaStore once, freezing exactly the verified bytes;
            # FFmpeg never reopens a potentially changed original asset path.
            references = [segment.narration_asset for segment in timeline.segments] + [shot.visual_asset for shot in timeline.shots]
            for index, reference in enumerate(references):
                data = media_store.read_bytes(reference)
                extension = {MediaType.AUDIO: ".wav", MediaType.IMAGE: ".png", MediaType.VIDEO: ".mp4"}[reference.media_type]
                path = work / f"input-{index:06d}{extension}"
                path.write_bytes(data)
                staged[reference.asset_id] = path
            audio_elapsed = 0.0
            for segment in timeline.segments:
                data = staged[segment.narration_asset.asset_id].read_bytes()
                measured = measure_wav_duration(data)
                asset = next(item for item in media_package.narration_assets if item.segment_id == segment.segment_id)
                if abs(measured - asset.duration_seconds) > 1e-9:
                    raise RenderError("Narration manifest duration differs from measured WAV")
                if abs(audio_elapsed - segment.start_ms / 1000) > AUDIO_SYNC_SECONDS:
                    raise RenderError("Cumulative narration timing differs from authoritative timeline")
                audio_elapsed += measured
                if abs(audio_elapsed - segment.end_ms / 1000) > AUDIO_SYNC_SECONDS:
                    raise RenderError("Cumulative narration timing differs from authoritative timeline")
            # Preflight every visual before running any encoding command.
            for shot in timeline.shots:
                image = shot.visual_asset.media_type == MediaType.IMAGE
                probe = self._probe(ffprobe, staged[shot.visual_asset.asset_id], demuxer="png_pipe" if image else "mov")
                streams = [stream for stream in probe["streams"] if stream.get("codec_type") == "video"]
                if (len(streams) != 1 or not streams[0].get("codec_name")
                        or (image and streams[0].get("codec_name") != "png")):
                    raise RenderError("Visual must be a decodable PNG or single-video-stream MP4")
                if not image:
                    self._duration(probe.get("format", {}).get("duration"))
            common = [ffmpeg, "-hide_banner", "-loglevel", "error", "-nostdin", "-n", "-xerror",
                      "-filter_threads", "1", "-filter_complex_threads", "1"]
            clips = []
            for index, (shot, count) in enumerate(zip(timeline.shots, counts)):
                image = shot.visual_asset.media_type == MediaType.IMAGE
                clip = f"shot-{index:06d}.mp4"
                source = staged[shot.visual_asset.asset_id]
                input_args = ["-loop", "1", "-framerate", str(FPS), "-f", "image2", "-pattern_type", "none"] if image else ["-f", "mov"]
                filters = ("setpts=PTS-STARTPTS,scale=iw*sar:ih,setsar=1,"
                           "scale=1280:720:force_original_aspect_ratio=decrease:force_divisible_by=2,"
                           "pad=1280:720:(ow-iw)/2:(oh-ih)/2,setsar=1,fps=30,format=yuv420p,"
                           f"tpad=stop_mode=clone:stop_duration={count / FPS:.9f},setpts=N/(30*TB)")
                self._run(common + ["-protocol_whitelist", "file,pipe", *input_args, "-i", str(source),
                                    "-map", "0:v:0", "-an", "-sn", "-dn", "-vf", filters,
                                    "-frames:v", str(count), "-c:v", "libx264", "-preset", "medium", "-crf", "18",
                                    "-threads", "1", "-x264-params", "threads=1:bframes=0:scenecut=0:keyint=30:min-keyint=30",
                                    "-pix_fmt", "yuv420p", "-video_track_timescale", "30000", "-map_metadata", "-1",
                                    "-fflags", "+bitexact", "-flags:v", "+bitexact", clip], cwd=work)
                clips.append(clip)
            # Normalize full narration to identical PCM stream parameters, no
            # duration limiter or silence insertion. Check cumulative resampling
            # rounding again before concatenating the normalized WAVs.
            audio_clips = []
            elapsed = 0.0
            for index, segment in enumerate(timeline.segments):
                clip = f"narration-{index:06d}.wav"
                self._run(common + ["-protocol_whitelist", "file,pipe", "-f", "wav", "-i",
                                    str(staged[segment.narration_asset.asset_id]), "-map", "0:a:0", "-vn",
                                    "-ac", "2", "-ar", str(SAMPLE_RATE), "-c:a", "pcm_s16le", "-threads", "1",
                                    "-map_metadata", "-1", "-fflags", "+bitexact", "-flags:a", "+bitexact", clip], cwd=work)
                elapsed += measure_wav_duration((work / clip).read_bytes())
                if abs(elapsed - segment.end_ms / 1000) > AUDIO_SYNC_SECONDS:
                    raise RenderError("Resampled narration boundaries differ from authoritative timeline")
                audio_clips.append(clip)
            for filename, names in (("visuals.txt", clips), ("narration.txt", audio_clips)):
                (work / filename).write_text("".join(f"file '{name}'\n" for name in names), encoding="utf-8", newline="\n")
            self._run(common + ["-protocol_whitelist", "file,pipe", "-f", "concat", "-safe", "1", "-i", "narration.txt",
                                "-map", "0:a:0", "-c:a", "copy", "-map_metadata", "-1", "narration.wav"], cwd=work)
            actual_audio = measure_wav_duration((work / "narration.wav").read_bytes())
            if abs(actual_audio - elapsed) > AUDIO_SYNC_SECONDS:
                raise RenderError("Concatenated narration duration differs from complete PCM segments")
            # MP4 clip headers have millisecond duration rounding. Restamp every
            # H.264 packet from its global frame index so concat header errors
            # cannot accumulate. With bframes=0 each packet is one ordered frame.
            # No -shortest or -t: never discard the last narration samples.
            self._run(common + ["-protocol_whitelist", "file,pipe", "-f", "concat", "-safe", "1", "-i", "visuals.txt",
                                "-protocol_whitelist", "file,pipe", "-f", "wav", "-i", "narration.wav",
                                "-map", "0:v:0", "-map", "1:a:0", "-c:v", "copy",
                                "-bsf:v", "setts=pts=N*1000:dts=N*1000:duration=1000:time_base=1/30000",
                                "-c:a", "aac", "-b:a", "192k",
                                "-ac", "2", "-ar", str(SAMPLE_RATE), "-threads", "1", "-map_metadata", "-1",
                                "-fflags", "+bitexact", "-flags:a", "+bitexact", "-movflags", "+faststart",
                                "-video_track_timescale", "30000", "completed.mp4"], cwd=work)
            completed = work / "completed.mp4"
            measured = self._validate_output(self._probe(ffprobe, completed), frames=sum(counts),
                                             planned=timeline.total_duration_ms / 1000, audio_duration=actual_audio)
            digest = _digest(completed)
            # Reject all known conflicts before publishing either independent
            # output. Exclusive links also handle concurrent publication safely.
            destination = self._output(destination, ".mp4")
            if destination.exists() and _digest(destination) != digest:
                raise RenderConflictError("Different immutable MP4 already exists")
            if subtitle_path is not None:
                sidecar = work / "completed.srt"
                sidecar.write_bytes(srt_content)
                self._publish(sidecar, subtitle_path)
            self._publish(completed, destination)
            if _digest(destination) != digest:
                raise RenderError("Published MP4 digest differs from validated output")
            return RenderResult(destination, digest, measured, sum(counts), version, subtitle_path)
