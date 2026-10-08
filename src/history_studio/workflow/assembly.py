"""One-writer durable P8-C publication; orphan files never confer authority."""
from collections.abc import Callable
import hashlib
from pathlib import Path
from typing import Protocol

from pydantic import ConfigDict, field_validator

from history_studio.assembly import RenderResult, build_subtitle_cues, build_timeline_plan, serialize_srt
from history_studio.assembly.rendering import DURATION_TOLERANCE_SECONDS
from history_studio.media import measure_wav_duration, validate_media_package
from history_studio.models import (
    ArtifactReference, AssemblyPackage, MediaAssetReference, MediaPackage, MediaType,
    ProjectConfig, ScriptPackage, StoryboardPackage, TimelinePlan,
)
from history_studio.models.base import Contract, PositiveSeconds, Text
from history_studio.storage import ArtifactStore, MediaIntegrityError, MediaStore
from history_studio.storage.artifact_store import safe_component, write_json

from .state_machine import InvalidTransitionError, ProjectStateMachine
from .states import ProjectState as S, RuntimeState


class AssemblyConfiguration(Contract):
    """Explicit local settings; relative render roots resolve beside the CLI config."""
    model_config = ConfigDict(frozen=True)

    render_root: Path
    ffmpeg: Text
    ffprobe: Text
    timeout_seconds: PositiveSeconds = 300

    @field_validator("render_root", mode="before")
    @classmethod
    def explicit_root(cls, value):
        if isinstance(value, str) and not value.strip():
            raise ValueError("Assembly render root must be explicit and nonempty")
        return value


class AssemblyRenderer(Protocol):
    def render(self, timeline: TimelinePlan, media_package: MediaPackage, media_store: MediaStore,
               output_path: Path, *, media_input_ref: ArtifactReference, srt_content: bytes) -> RenderResult: ...


class AssemblyWorkflowOutcome(Contract):
    state: RuntimeState
    package: AssemblyPackage | None = None
    error_type: Text | None = None
    error_message: Text | None = None


class AssemblyWorkflow:
    """Only exact bound media and exact bound assembly are workflow authorities.

    Output paths depend on project + bound media version, never retry/manifest
    version. Retrying renders that same immutable destination and saves a fresh
    manifest using ArtifactStore's normal version policy, without reading orphans.
    COMPLETE restarts verify exact bound content without invoking the factory.
    A hard crash leaves ASSEMBLING for retry; caught errors retain media in FAILED.
    Recovery of FAILED interrupted in ASSEMBLING requires explicit recover=True.
    """

    def __init__(self, *, render_root: Path, renderer_factory: Callable[[], AssemblyRenderer]) -> None:
        supplied = Path(render_root).absolute()
        if supplied.anchor.startswith(("\\\\", "//")):
            raise ValueError("Assembly render root must be a local directory")
        for path in (supplied, *supplied.parents):
            if path.is_symlink() or path.is_junction():
                raise ValueError("Assembly render root must not contain links or junctions")
        self.render_root = supplied.resolve()
        self.renderer_factory = renderer_factory

    def run(self, project: ProjectConfig, store: ArtifactStore, *, media_store: MediaStore | None = None,
            recover: bool = False) -> AssemblyWorkflowOutcome:
        if project.project_id != store.project_dir.name:
            raise ValueError("Project ID must match artifact directory")
        state_path = store.project_dir / ".runtime/state.json"
        state = RuntimeState.model_validate_json(state_path.read_text(encoding="utf-8"))
        active = state.failed_state if state.current_state == S.FAILED else state.current_state
        if active not in (S.ASSEMBLING, S.COMPLETE):
            raise InvalidTransitionError("Assembly requires ASSEMBLING or authenticated bound COMPLETE")
        if state.current_state == S.FAILED and not recover:
            raise InvalidTransitionError("Failed assembly requires explicit recovery authorization (--recover)")
        if active == S.ASSEMBLING and state.last_successful_state != S.STORYBOARD_APPROVED:
            raise InvalidTransitionError("Assembly requires the STORYBOARD_APPROVED checkpoint")
        binaries = media_store if media_store is not None else MediaStore(store.project_dir / "media")
        final_store = MediaStore(self.render_root)

        def inputs():
            media_ref = state.require_media_ref(project.project_id)
            package = store.load("media", media_ref.version, MediaPackage)
            board_ref = package.storyboard_input_ref
            if board_ref != state.require_approved_storyboard_ref(project.project_id):
                raise ValueError("Bound media must identify the exact workflow approved Storyboard")
            board = store.load("storyboard", board_ref.version, StoryboardPackage)
            script_ref = board.script_input_ref
            if state.artifacts.approved_script is not None and script_ref != state.artifacts.approved_script:
                raise ValueError("Storyboard must identify the exact workflow approved Script")
            script = store.load("script", script_ref.version, ScriptPackage)
            report = validate_media_package(storyboard_input_ref=board_ref, storyboard=board,
                script_input_ref=script_ref, script=script, package=package)
            if not report.is_valid:
                raise ValueError("Invalid assembly inputs: " + "; ".join(issue.code.value for issue in report.issues))
            # Verify all final P7 bytes and actual WAV measurements before planning.
            for narration in package.narration_assets:
                if measure_wav_duration(binaries.read_bytes(narration.asset)) != narration.duration_seconds:
                    raise MediaIntegrityError("Narration duration differs from actual persisted WAV")
            for visual in package.visual_assets:
                binaries.verify(visual.asset)
            plan = build_timeline_plan(media_input_ref=media_ref, package=package,
                storyboard_input_ref=board_ref, storyboard=board, script_input_ref=script_ref, script=script)
            srt = serialize_srt(build_subtitle_cues(plan=plan, script_input_ref=script_ref, script=script))
            return media_ref, package, plan, srt

        def destinations(media_ref):
            base = Path("final") / safe_component(project.project_id) / f"media-v{media_ref.version}"
            return base / "documentary.mp4", base / "documentary.srt"

        def authenticate(package, media_ref, plan, srt):
            video_path, srt_path = destinations(media_ref)
            if (package.media_input_ref != media_ref or package.timeline_duration_ms != plan.total_duration_ms
                    or abs(package.measured_duration_seconds - plan.total_duration_ms / 1000) > DURATION_TOLERANCE_SECONDS):
                raise ValueError("Assembly must match exact bound media, timeline and measured duration tolerance")
            if (package.final_video.relative_path != video_path.as_posix()
                    or package.subtitles.relative_path != srt_path.as_posix()):
                raise ValueError("Assembly must identify the fixed immutable final-output location")
            final_store.verify(package.final_video)
            persisted_srt = final_store.read_bytes(package.subtitles)
            if persisted_srt != srt:
                raise MediaIntegrityError("Published subtitles differ from exact Script narration and timeline")

        if active == S.COMPLETE:
            # Never mutate a completed checkpoint, even if any bound bytes fail.
            media_ref, _, plan, srt = inputs()
            assembly_ref = state.require_assembly_ref(project.project_id)
            package = store.load("assembly", assembly_ref.version, AssemblyPackage)
            authenticate(package, media_ref, plan, srt)
            return AssemblyWorkflowOutcome(state=state, package=package)

        machine = ProjectStateMachine(state)
        ready = None
        try:
            media_ref, media, plan, srt = inputs()
            if state.current_state == S.FAILED:
                machine.recover()
                write_json(state_path, machine.state, replace=True)
            video_path, srt_path = destinations(media_ref)
            result = self.renderer_factory().render(plan, media, binaries, video_path,
                media_input_ref=media_ref, srt_content=srt)
            if (not isinstance(result, RenderResult) or result.output_path != self.render_root / video_path
                    or result.subtitle_path != self.render_root / srt_path):
                raise ValueError("Renderer must publish both outputs at the configured immutable destination")
            package = AssemblyPackage(media_input_ref=media_ref,
                final_video=MediaAssetReference(asset_id=f"assembly-media-v{media_ref.version}-video",
                    relative_path=video_path.as_posix(), media_type=MediaType.VIDEO, sha256=result.sha256),
                subtitles=MediaAssetReference(asset_id=f"assembly-media-v{media_ref.version}-subtitles",
                    relative_path=srt_path.as_posix(), media_type=MediaType.SUBTITLE,
                    sha256=hashlib.sha256(srt).hexdigest()),
                timeline_duration_ms=plan.total_duration_ms, measured_duration_seconds=result.duration_seconds,
                ffmpeg_version=result.ffmpeg_version)
            authenticate(package, media_ref, plan, srt)
            version = store.save("assembly", package)
            durable = store.load("assembly", version, AssemblyPackage)
            if durable != package:
                raise ValueError("Reloaded exact AssemblyPackage differs from accepted publication")
            authenticate(durable, media_ref, plan, srt)
            ready = ProjectStateMachine(machine.state)
            ready.complete_assembly(ArtifactReference(project_id=project.project_id, artifact_type="assembly", version=version),
                                    project_id=project.project_id)
            # Binding and COMPLETE are one atomic runtime-state publication.
            write_json(state_path, ready.state, replace=True)
            return AssemblyWorkflowOutcome(state=ready.state, package=durable)
        except Exception as exc:
            # If atomic replacement committed before an error was reported, do
            # not downgrade that verified checkpoint. A rerun authenticates it.
            if ready is not None:
                persisted = RuntimeState.model_validate_json(state_path.read_text(encoding="utf-8"))
                if persisted == ready.state:
                    raise
            if machine.state.current_state != S.FAILED:
                machine.fail(f"assembly_execution_failed:{type(exc).__name__[:120]}")
                write_json(state_path, machine.state, replace=True)
            return AssemblyWorkflowOutcome(state=machine.state, error_type=type(exc).__name__[:120],
                                           error_message=str(exc) or type(exc).__name__)
