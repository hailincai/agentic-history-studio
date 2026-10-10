"""Journaled Phase 7 execution; immutable binaries/manifests are not transactions."""
from collections.abc import Callable
from dataclasses import dataclass

from history_studio.media import generate_narration_assets, generate_visual_assets, measure_wav_duration
from history_studio.media.image import ImageProvider
from history_studio.media.tts import TTSProvider
from history_studio.media.video import VideoProvider
from history_studio.media.validation import MediaIntegrityReport, validate_media_package
from history_studio.media.recovery import AssetRecovery, RecoveryScope, provider_identity, configuration_digest
import hashlib
from history_studio.budget import ledger_lock
from history_studio.models import ArtifactReference, GenerationMethod, MediaPackage, ProjectConfig, ScriptPackage, StoryboardPackage
from history_studio.models.base import Contract, Text
from history_studio.storage import ArtifactStore, MediaIntegrityError, MediaStore
from history_studio.storage.artifact_store import write_json

from .states import RuntimeState, ProjectState as S
from .state_machine import ProjectStateMachine, InvalidTransitionError


@dataclass(frozen=True)
class MediaProviders:
    tts: TTSProvider
    image: ImageProvider | None = None
    video: VideoProvider | None = None
    # Supplementary scope configuration; never substitutes for each custom
    # provider's complete recovery_identity() declaration. Persisted as a digest.
    recovery_identity: dict | None = None


class MediaWorkflowOutcome(Contract):
    state: RuntimeState
    package: MediaPackage | None = None
    error_type: Text | None = None
    validation_report: MediaIntegrityReport | None = None


class MediaWorkflow:
    """Only exact approved inputs and bound outputs confer workflow authority.

    Retry reuses only exact verified completed journal entries, never orphans.
    Binary/manifest/state publication is not a transaction. Failures before binding
    retain upstream approval and may leave orphans. A hard crash leaves GENERATING_MEDIA
    and durable dispatch evidence. ASSEMBLING reruns authenticate exact bound media
    without providers or recovery-journal discovery.
    """

    def __init__(self, *, provider_factory: Callable[[StoryboardPackage], MediaProviders]) -> None:
        self.provider_factory = provider_factory

    def run(self, project: ProjectConfig, store: ArtifactStore, *,
            media_store: MediaStore | None = None) -> MediaWorkflowOutcome:
        if project.project_id != store.project_dir.name:
            raise ValueError("Project ID must match artifact directory")
        path = store.project_dir / ".runtime/state.json"
        state = RuntimeState.model_validate_json(path.read_text(encoding="utf-8"))
        active = state.failed_state if state.current_state == S.FAILED else state.current_state
        if active not in (S.STORYBOARD_APPROVED, S.GENERATING_MEDIA, S.ASSEMBLING):
            raise InvalidTransitionError("Media requires STORYBOARD_APPROVED, GENERATING_MEDIA or bound ASSEMBLING")
        if state.current_state == S.FAILED and active == S.STORYBOARD_APPROVED:
            raise InvalidTransitionError("Only interrupted media generation or assembly can resume media")
        if state.last_successful_state != S.STORYBOARD_APPROVED:
            raise InvalidTransitionError("Media execution requires the STORYBOARD_APPROVED checkpoint")
        reference = state.require_approved_storyboard_ref(project.project_id)
        binaries = media_store if media_store is not None else MediaStore(store.project_dir / "media")

        def sources():
            storyboard = store.load("storyboard", reference.version, StoryboardPackage)
            if storyboard.script_input_ref.project_id != project.project_id:
                raise ValueError("Storyboard Script provenance belongs to a different project")
            script_ref = storyboard.script_input_ref
            if state.artifacts.approved_script is not None and state.artifacts.approved_script != script_ref:
                raise ValueError("Storyboard Script provenance must match the exact workflow approved Script")
            script = store.load("script", script_ref.version, ScriptPackage)
            if script.story_input_ref.project_id != project.project_id:
                raise ValueError("Script provenance belongs to a different project")
            known = {segment.segment_id for section in script.sections for segment in section.segments}
            if any(shot.source_segment_id not in known for section in storyboard.sections for shot in section.shots):
                raise ValueError("Storyboard references an unknown Script segment")
            return storyboard, script_ref, script

        def authenticate(package, storyboard, script_ref, script):
            report = validate_media_package(storyboard_input_ref=reference, storyboard=storyboard,
                script_input_ref=script_ref, script=script, package=package)
            if report.is_valid:
                for narration in package.narration_assets:
                    data = binaries.read_bytes(narration.asset)
                    if measure_wav_duration(data) != narration.duration_seconds:
                        raise MediaIntegrityError("Narration duration must match actual persisted WAV audio")
                for visual in package.visual_assets:
                    binaries.verify(visual.asset)
            return report

        if active == S.ASSEMBLING:
            media_ref = state.require_media_ref(project.project_id)
            storyboard, script_ref, script = sources()
            package = store.load("media", media_ref.version, MediaPackage)
            report = authenticate(package, storyboard, script_ref, script)
            if not report.is_valid:
                raise ValueError("Bound media integrity is invalid")
            if state.current_state == S.FAILED:
                state = ProjectStateMachine(state).recover()
                write_json(path, state, replace=True)
            return MediaWorkflowOutcome(state=state, package=package, validation_report=report)

        machine = ProjectStateMachine(state)
        if state.current_state == S.FAILED:
            machine.recover()
        machine.begin_media(project_id=project.project_id)
        report = None
        state_lock = store.project_dir / ".runtime/media_state.lock"
        try:
            with ledger_lock(state_lock):
                current = RuntimeState.model_validate_json(path.read_text(encoding="utf-8"))
                if current.artifacts != machine.state.artifacts:
                    raise ValueError("Concurrent media authority changed; no dispatch authorized")
                write_json(path, machine.state, replace=True)
            storyboard, script_ref, script = sources()
            providers = self.provider_factory(StoryboardPackage.model_validate(storyboard.model_dump(mode="json")))
            if not isinstance(providers, MediaProviders) or providers.tts is None:
                raise ValueError("Media factory must return MediaProviders with a TTS provider")
            methods = {shot.generation_method for section in storyboard.sections for shot in section.shots}
            if methods & {GenerationMethod.STATIC_IMAGE, GenerationMethod.IMAGE_TO_VIDEO} and providers.image is None:
                raise ValueError("Approved Storyboard requires an image provider before narration execution")
            if methods & {GenerationMethod.TEXT_TO_VIDEO, GenerationMethod.IMAGE_TO_VIDEO} and providers.video is None:
                raise ValueError("Approved Storyboard requires a video provider before narration execution")
            identities = dict(tts=provider_identity(providers.tts), image=provider_identity(providers.image),
                              video=provider_identity(providers.video),
                              explicit=(configuration_digest(providers.recovery_identity)
                                        if providers.recovery_identity is not None else None))
            config_path = store.project_dir / ".runtime/media_config.json"
            if config_path.exists():
                identities["media_config_sha256"] = hashlib.sha256(config_path.read_bytes()).hexdigest()
            recovery = AssetRecovery(store.project_dir, RecoveryScope(project_id=project.project_id,
                storyboard_ref=reference, script_ref=script_ref,
                storyboard_sha256=hashlib.sha256((store.project_dir / "storyboard" / f"storyboard_v{reference.version}.json").read_bytes()).hexdigest(),
                script_sha256=hashlib.sha256((store.project_dir / "script" / f"script_v{script_ref.version}.json").read_bytes()).hexdigest(),
                providers=identities, media_root=str(binaries.root)))
            narration = generate_narration_assets(storyboard_input_ref=reference, artifact_store=store,
                                                  provider=providers.tts, media_store=binaries, recovery=recovery)
            visuals = generate_visual_assets(storyboard_input_ref=reference, artifact_store=store, media_store=binaries,
                                              image_provider=providers.image, video_provider=providers.video, recovery=recovery)
            package = MediaPackage(storyboard_input_ref=reference, title=storyboard.title,
                                   narration_assets=narration, visual_assets=visuals)
            report = authenticate(package, storyboard, script_ref, script)
            if not report.is_valid:
                raise ValueError("Media integrity validation failed")
            version = store.save("media", package)
            durable = store.load("media", version, MediaPackage)
            if durable != package:
                raise ValueError("Published media must equal accepted manifest")
            report = authenticate(durable, storyboard, script_ref, script)
            if not report.is_valid:
                raise ValueError("Published media integrity is invalid")
            ready = ProjectStateMachine(machine.state)
            ready.complete_media(ArtifactReference(project_id=project.project_id, artifact_type="media", version=version),
                                 project_id=project.project_id)
            with ledger_lock(state_lock):
                current = RuntimeState.model_validate_json(path.read_text(encoding="utf-8"))
                if current.artifacts != machine.state.artifacts:
                    raise ValueError("Concurrent media authority changed; no alternate binding authorized")
                write_json(path, ready.state, replace=True)
            return MediaWorkflowOutcome(state=ready.state, package=durable, validation_report=report)
        except Exception as exc:
            with ledger_lock(state_lock):
                current = RuntimeState.model_validate_json(path.read_text(encoding="utf-8"))
                if current.artifacts == machine.state.artifacts:
                    machine.fail(f"media_execution_failed:{type(exc).__name__[:120]}")
                    write_json(path, machine.state, replace=True)
                    current = machine.state
            return MediaWorkflowOutcome(state=current, error_type=type(exc).__name__[:120], validation_report=report)
