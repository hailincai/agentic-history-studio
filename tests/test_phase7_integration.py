"""P7-Z durable integration closure, written for manual execution by the user.

Only external media providers are fake. Workflows, execution services, stores,
contracts, validation, approval setup and runtime publication use production code.
No production edits or pytest execution accompany this test-only patch.
"""
import hashlib
import io
import struct
import wave
import zlib
from datetime import datetime, timezone

import pytest

from history_studio.cli import read_project
from history_studio.media import ImageGenerationResult, TTSResult, VideoGenerationResult, validate_media_package
from history_studio.models import (
    ArtifactReference, GenerationMetadata, MediaAssetReference, MediaPackage, MediaType,
    NarrationAsset, ProjectConfig, ScriptPackage, StoryboardPackage, VisualAsset,
)
from history_studio.storage import ArtifactStore, MediaIntegrityError, MediaStore
from history_studio.storage.artifact_store import write_json
from history_studio.workflow import ApprovalRecord, ProjectState as S, RuntimeState, WorkflowArtifactBindings
from history_studio.workflow.media import MediaProviders, MediaWorkflow
from history_studio.workflow.storyboard_review import apply_storyboard_review


@pytest.fixture(autouse=True)
def fake_providers_only(monkeypatch):
    """Keep conftest's socket guard and also reject accidental SDK client creation."""
    from openai import AsyncOpenAI, OpenAI
    import history_studio.research.openai_provider as transport

    def forbidden(*args, **kwargs):
        pytest.fail("P7-Z uses directly injected fake media providers; no OpenAI client or paid API")

    monkeypatch.setattr(OpenAI, "__init__", forbidden)
    monkeypatch.setattr(AsyncOpenAI, "__init__", forbidden)
    monkeypatch.setattr(transport, "create_client", forbidden)


def ref(kind, version=1):
    return ArtifactReference(project_id="phase7", artifact_type=kind, version=version)


def wav_payload(text, treatment="official"):
    """Deterministic PCM audio; duration is encoded in actual frames, not metadata."""
    frames = 400 + 37 * len(text.encode("utf-8"))
    sample = hashlib.sha256((treatment + text).encode("utf-8")).digest()[:2]
    buffer = io.BytesIO()
    with wave.open(buffer, "wb") as audio:
        audio.setnchannels(1)
        audio.setsampwidth(2)
        audio.setframerate(8000)
        audio.writeframes(sample * frames)
    return buffer.getvalue()


def png_payload(prompt, treatment="official"):
    """Valid one-pixel RGB PNG, sufficient for storage/integrity integration."""
    def chunk(kind, data):
        return struct.pack(">I", len(data)) + kind + data + struct.pack(">I", zlib.crc32(kind + data))
    color = hashlib.sha256((treatment + prompt).encode("utf-8")).digest()[:3]
    return (b"\x89PNG\r\n\x1a\n" + chunk(b"IHDR", struct.pack(">IIBBBBB", 1, 1, 8, 2, 0, 0, 0))
            + chunk(b"IDAT", zlib.compress(b"\x00" + color)) + chunk(b"IEND", b""))


def video_payload(prompt, image=None, treatment="official"):
    """Deterministic ISO BMFF container fixture; playback decoding is outside P7-Z."""
    def box(kind, data):
        return struct.pack(">I", len(data) + 8) + kind + data
    identity = treatment.encode() + prompt.encode("utf-8") + (image or b"")
    return box(b"ftyp", b"isom\x00\x00\x02\x00isommp42") + box(b"free", hashlib.sha256(identity).digest())


def actual_wav_duration(data):
    """Independent standard-library timing check on persisted PCM frames."""
    with wave.open(io.BytesIO(data), "rb") as audio:
        frames, rate = audio.getnframes(), audio.getframerate()
        assert len(audio.readframes(frames)) == frames * audio.getnchannels() * audio.getsampwidth()
        assert frames > 0 and rate > 0
        return frames / rate


class FakeTTS:
    def __init__(self, fail_at=None):
        self.fail_at = fail_at
        self.texts = []

    def recovery_identity(self):
        return dict(version=1, complete=True, implementation="phase7-pcm-v1",
                    settings={})

    def synthesize(self, *, text):
        self.texts.append(text)
        if len(self.texts) == self.fail_at:
            raise RuntimeError("deterministic TTS failure after earlier publication")
        return TTSResult(audio_bytes=wav_payload(text), provider="fake-tts", model="phase7-pcm")


class FakeImage:
    def __init__(self):
        self.prompts = []

    def recovery_identity(self):
        return dict(version=1, complete=True, implementation="phase7-png-v1",
                    settings={})

    def generate(self, *, prompt):
        self.prompts.append(prompt)
        return ImageGenerationResult(image_bytes=png_payload(prompt), provider="fake-image", model="phase7-png")


class FakeVideo:
    def __init__(self, media_root):
        self.media_root = media_root
        self.text_prompts = []
        self.image_calls = []

    def recovery_identity(self):
        return dict(version=1, complete=True, implementation="phase7-mp4-v1",
                    settings={})

    def generate_from_text(self, *, prompt):
        self.text_prompts.append(prompt)
        return VideoGenerationResult(video_bytes=video_payload(prompt), provider="fake-video", model="phase7-mp4")

    def generate_from_image(self, *, image, prompt):
        # This provider observes an already durable intermediate, not just transient
        # generated bytes. The service's real MediaStore read verifies its hash.
        matches = [path for path in self.media_root.rglob("*.png")
                   if "intermediate" in path.parts and path.read_bytes() == image]
        assert len(matches) == 1
        self.image_calls.append((prompt, image))
        return VideoGenerationResult(video_bytes=video_payload(prompt, image), provider="fake-video", model="phase7-mp4")


@pytest.fixture
def approved_project(tmp_path):
    """Persist exact inputs and reach STORYBOARD_APPROVED through the real P6 gate."""
    project = ProjectConfig(project_id="phase7", topic="Phase 7 integration")
    store = ArtifactStore(tmp_path / project.project_id)
    script = ScriptPackage.model_validate(dict(story_input_ref=ref("story", 3), title="Approved history plan",
        sections=[dict(section_id="section-z", title="Opening", segments=[
            dict(segment_id="seg-z", kind="HISTORICAL", narration="The accounts describe a journey — 李白。",
                 grounding=dict(story_beat_id="beat-z", research_fact_ids=["fact-z"])),
            dict(segment_id="seg-a", kind="STRUCTURAL", narration="Now we turn\nto the next part."),
        ]), dict(section_id="section-a", title="Closing", segments=[
            dict(segment_id="seg-m", kind="STRUCTURAL", narration="The closing narration retains  inner spaces."),
        ])]))
    script_version = store.save("script", script)

    def shot(shot_id, segment_id, kind, method):
        return dict(shot_id=shot_id, source_segment_id=segment_id, kind=kind,
            visual_description="A description that must not replace the approved prompt",
            generation_prompt=f"Approved {shot_id} prompt — 李白\nwith  inner spaces.",
            generation_method=method, framing="WIDE", camera_motion="NONE", estimated_duration_seconds=9)

    storyboard = StoryboardPackage.model_validate(dict(script_input_ref=ref("script", script_version), title=script.title,
        sections=[dict(section_id="section-z", title="Opening", shots=[
            shot("shot-z-1", "seg-z", "HISTORICAL", "STATIC_IMAGE"),
            shot("shot-z-2", "seg-z", "HISTORICAL", "IMAGE_TO_VIDEO"),
            shot("shot-a", "seg-a", "STRUCTURAL", "TEXT_TO_VIDEO"),
        ]), dict(section_id="section-a", title="Closing", shots=[
            shot("shot-m", "seg-m", "STRUCTURAL", "STATIC_IMAGE"),
        ])]))
    storyboard_version = store.save("storyboard", storyboard)
    waiting = RuntimeState(current_state=S.WAITING_STORYBOARD_APPROVAL, last_successful_state=S.WAITING_STORYBOARD_APPROVAL,
        artifacts=WorkflowArtifactBindings(research=ref("research"), verification=ref("verification", 2),
            approved_verification=ref("verification", 2), story=ref("story", 3), approved_story=ref("story", 3),
            script=ref("script", script_version), approved_script=ref("script", script_version),
            storyboard=ref("storyboard", storyboard_version)))
    write_json(store.project_dir / "project.json", project)
    write_json(store.project_dir / ".runtime/state.json", waiting)
    approved = apply_storyboard_review(store.project_dir, ApprovalRecord(project_id=project.project_id,
        stage="storyboard", artifact_type="storyboard", artifact_version=storyboard_version, decision="APPROVED",
        decision_source="human", decided_by="Integration reviewer", feedback="Fixture attestation of the exact plan",
        decided_at=datetime(2026, 10, 6, 15, tzinfo=timezone.utc)))
    assert approved.current_state == S.STORYBOARD_APPROVED
    assert approved.artifacts.approved_storyboard == approved.artifacts.storyboard
    return project, store, script, storyboard, approved


def protect_phase7_boundary(monkeypatch, approved, *, bound_media=None, forbidden_media_versions=()):
    """Observe real exact loads; prohibit latest/upstream semantics without replacing execution."""
    load, save, versions = ArtifactStore.load, ArtifactStore.save, ArtifactStore.list_versions
    loads = []
    def exact_load(self, kind, version, model):
        assert kind in ("storyboard", "script", "media"), "Phase 7 must not restart upstream semantic stages"
        if kind == "storyboard":
            assert version == approved.artifacts.approved_storyboard.version
        elif kind == "script":
            assert version == approved.artifacts.approved_script.version
        else:
            assert version not in forbidden_media_versions, "An orphan cannot become authority by discovery"
            if bound_media is not None:
                assert version == bound_media.version
        loads.append((kind, version))
        return load(self, kind, version, model)
    def official_save(self, kind, model):
        assert kind == "media", "Phase 7 may publish only its MediaPackage, not Phase 8 outputs"
        return save(self, kind, model)
    def allocation_only(self, kind):
        assert kind in ("media", "approvals"), "No latest Storyboard or Script version discovery"
        return versions(self, kind)
    def forbidden(*args, **kwargs):
        pytest.fail("Latest artifact discovery cannot establish Phase 7 authority")
    monkeypatch.setattr(ArtifactStore, "load", exact_load)
    monkeypatch.setattr(ArtifactStore, "save", official_save)
    monkeypatch.setattr(ArtifactStore, "list_versions", allocation_only)
    monkeypatch.setattr(ArtifactStore, "load_latest", forbidden)
    return loads


def run_media(project, store, *, fail_tts_at=None):
    providers = MediaProviders(FakeTTS(fail_tts_at), FakeImage(), FakeVideo(store.project_dir / "media"))
    selected = []
    def factory(storyboard):
        state = read_project(store.project_dir)[1]
        assert state.current_state == S.GENERATING_MEDIA and state.artifacts.media is None
        selected.append(storyboard)
        return providers
    outcome = MediaWorkflow(provider_factory=factory).run(project, store, media_store=MediaStore(store.project_dir / "media"))
    return outcome, providers, selected


def files_under(root):
    return {path.relative_to(root).as_posix(): path.read_bytes() for path in root.rglob("*") if path.is_file()}


def assert_complete(project, store, script, storyboard, approved, outcome):
    state = read_project(store.project_dir)[1]
    assert state == outcome.state and state.current_state == S.ASSEMBLING
    assert state.last_successful_state == S.STORYBOARD_APPROVED
    assert state.artifacts.media is not None
    for field in type(approved.artifacts).model_fields:
        if field != "media":
            assert getattr(state.artifacts, field) == getattr(approved.artifacts, field)
    package = store.load("media", state.artifacts.media.version, MediaPackage)
    assert package == outcome.package and package.title == storyboard.title
    assert package.storyboard_input_ref == approved.artifacts.approved_storyboard
    assert storyboard.script_input_ref == approved.artifacts.approved_script
    assert validate_media_package(storyboard_input_ref=approved.artifacts.approved_storyboard, storyboard=storyboard,
        script_input_ref=approved.artifacts.approved_script, script=script, package=package).is_valid
    shots = [shot for section in storyboard.sections for shot in section.shots]
    represented = {shot.source_segment_id for shot in shots}
    segments = [segment for section in script.sections for segment in section.segments if segment.segment_id in represented]
    assert [asset.segment_id for asset in package.narration_assets] == [segment.segment_id for segment in segments]
    assert len(package.narration_assets) == len(represented) == 3
    assert [asset.shot_id for asset in package.visual_assets] == [shot.shot_id for shot in shots]
    assert len(package.visual_assets) == len(shots) == 4
    media_store = MediaStore(store.project_dir / "media")
    finals = [item.asset for item in (*package.narration_assets, *package.visual_assets)]
    assert len({asset.asset_id for asset in finals}) == len(finals)
    for asset in finals:
        path = media_store.root / asset.relative_path
        assert path.is_file() and path.resolve().is_relative_to(media_store.root)
        assert media_store.verify(asset)
        assert hashlib.sha256(path.read_bytes()).hexdigest() == asset.sha256
    for asset, segment in zip(package.narration_assets, segments):
        data = media_store.read_bytes(asset.asset)
        assert asset.asset.media_type == MediaType.AUDIO and data == wav_payload(segment.narration)
        assert asset.duration_seconds == actual_wav_duration(data) > 0
        assert asset.duration_seconds != 9  # Every shot's planning estimate is deliberately different.
    for asset, shot in zip(package.visual_assets, shots):
        assert asset.source_segment_id == shot.source_segment_id and asset.generation_method == shot.generation_method
        expected_type = MediaType.IMAGE if shot.generation_method == "STATIC_IMAGE" else MediaType.VIDEO
        assert asset.asset.media_type == expected_type
    assert not hasattr(state.artifacts, "approved_media")
    assert not {"assembly", "timeline", "subtitles", "final"} & {path.name for path in store.project_dir.iterdir()}
    return package


def seed_orphan(store, script, storyboard, approved):
    """A complete valid manifest and real binaries, deliberately never workflow-bound."""
    media_store = MediaStore(store.project_dir / "media")
    metadata = GenerationMetadata(provider="fake-orphan", model="unbound-fixture")
    shots = [shot for section in storyboard.sections for shot in section.shots]
    represented = {shot.source_segment_id for shot in shots}
    narration, visuals = [], []
    for section in script.sections:
        for segment in section.segments:
            if segment.segment_id not in represented:
                continue
            data = wav_payload(segment.narration, "orphan")
            asset = media_store.save_bytes(asset_id=f"orphan-n-{segment.segment_id}",
                relative_path=f"orphan/audio/{segment.segment_id}.wav", media_type=MediaType.AUDIO, data=data)
            narration.append(NarrationAsset(segment_id=segment.segment_id, asset=asset,
                duration_seconds=actual_wav_duration(data), generation_metadata=metadata))
    for shot in shots:
        is_image = shot.generation_method == "STATIC_IMAGE"
        data = png_payload(shot.generation_prompt, "orphan") if is_image else video_payload(shot.generation_prompt, treatment="orphan")
        asset = media_store.save_bytes(asset_id=f"orphan-v-{shot.shot_id}",
            relative_path=f"orphan/visual/{shot.shot_id}.{'png' if is_image else 'mp4'}",
            media_type=MediaType.IMAGE if is_image else MediaType.VIDEO, data=data)
        visuals.append(VisualAsset(shot_id=shot.shot_id, source_segment_id=shot.source_segment_id,
            generation_method=shot.generation_method, asset=asset, generation_metadata=metadata))
    package = MediaPackage(storyboard_input_ref=approved.artifacts.approved_storyboard, title=storyboard.title,
                           narration_assets=narration, visual_assets=visuals)
    assert validate_media_package(storyboard_input_ref=approved.artifacts.approved_storyboard, storyboard=storyboard,
        script_input_ref=approved.artifacts.approved_script, script=script, package=package).is_valid
    for item in (*package.narration_assets, *package.visual_assets):
        assert media_store.verify(item.asset)
    version = store.save("media", package)
    return ref("media", version), package


def test_phase7_complete_happy_path(approved_project, monkeypatch):
    project, store, script, storyboard, approved = approved_project
    audits = files_under(store.project_dir / "approvals")
    protect_phase7_boundary(monkeypatch, approved)
    outcome, providers, selected = run_media(project, store)
    package = assert_complete(project, store, script, storyboard, approved, outcome)
    assert selected == [storyboard]
    assert providers.tts.texts == [segment.narration for section in script.sections for segment in section.segments]
    shots = [shot for section in storyboard.sections for shot in section.shots]
    assert providers.image.prompts == [shot.generation_prompt for shot in shots if shot.generation_method != "TEXT_TO_VIDEO"]
    assert providers.video.text_prompts == [shots[2].generation_prompt]
    assert providers.video.image_calls == [(shots[1].generation_prompt, png_payload(shots[1].generation_prompt))]
    assert len(package.visual_assets) == 4  # The durable intermediate is not a second final output.
    assert files_under(store.project_dir / "approvals") == audits  # No Phase 7 Human Gate.


def test_phase7_restart_authenticates_exact_bound_media_without_providers(approved_project, monkeypatch):
    project, store, script, storyboard, approved = approved_project
    first, _, _ = run_media(project, store)
    package = assert_complete(project, store, script, storyboard, approved, first)
    before = files_under(store.project_dir)
    bound = first.state.artifacts.media
    fresh = ArtifactStore(store.project_dir)
    fresh_media = MediaStore(fresh.project_dir / "media")
    loads = protect_phase7_boundary(monkeypatch, approved, bound_media=bound)
    verified = []
    read_bytes = MediaStore.read_bytes
    def verified_read(self, reference):
        data = read_bytes(self, reference)
        verified.append(reference)
        return data
    def forbidden(*args, **kwargs):
        pytest.fail("Successful restart must not discover versions, save outputs or construct/call providers")
    monkeypatch.setattr(MediaStore, "read_bytes", verified_read)
    monkeypatch.setattr(ArtifactStore, "list_versions", forbidden)
    monkeypatch.setattr(ArtifactStore, "save", forbidden)
    restarted = MediaWorkflow(provider_factory=forbidden).run(project, fresh, media_store=fresh_media)
    assert restarted.state == first.state and restarted.package == package
    assert ("media", bound.version) in loads
    assert set(verified) == {item.asset for item in (*package.narration_assets, *package.visual_assets)}
    assert files_under(store.project_dir) == before


def test_phase7_valid_looking_orphan_is_history_not_authority(approved_project, monkeypatch):
    project, store, script, storyboard, approved = approved_project
    orphan_ref, orphan = seed_orphan(store, script, storyboard, approved)
    orphan_path = store.project_dir / "media" / f"media_v{orphan_ref.version}.json"
    orphan_bytes = orphan_path.read_bytes()
    assert read_project(store.project_dir)[1].artifacts.media is None
    protect_phase7_boundary(monkeypatch, approved, forbidden_media_versions={orphan_ref.version})
    outcome, providers, selected = run_media(project, ArtifactStore(store.project_dir))
    official = assert_complete(project, store, script, storyboard, approved, outcome)
    assert selected == [storyboard] and providers.tts.texts and providers.image.prompts and providers.video.image_calls
    assert outcome.state.artifacts.media != orphan_ref and official != orphan
    assert {item.asset.asset_id for item in (*official.narration_assets, *official.visual_assets)}.isdisjoint(
        item.asset.asset_id for item in (*orphan.narration_assets, *orphan.visual_assets))
    assert all(not item.asset.relative_path.startswith("orphan/") for item in (*official.narration_assets, *official.visual_assets))
    assert orphan_path.read_bytes() == orphan_bytes
    assert MediaPackage.model_validate_json(orphan_bytes) == orphan
    for item in (*orphan.narration_assets, *orphan.visual_assets):
        assert MediaStore(store.project_dir / "media").verify(item.asset)


def test_phase7_newer_unapproved_storyboard_cannot_drive_media(approved_project, monkeypatch):
    project, store, script, storyboard, approved = approved_project
    data = storyboard.model_dump(mode="json")
    data["title"] = "Unapproved replacement plan"
    for section in data["sections"]:
        for shot in section["shots"]:
            shot["shot_id"] = "unapproved-" + shot["shot_id"]
            shot["generation_prompt"] = "Unapproved replacement prompt"
    newer_version = store.save("storyboard", StoryboardPackage.model_validate(data))
    assert newer_version != approved.artifacts.approved_storyboard.version
    assert read_project(store.project_dir)[1] == approved  # Neither approved nor workflow-bound.
    protect_phase7_boundary(monkeypatch, approved)
    outcome, providers, selected = run_media(project, store)
    package = assert_complete(project, store, script, storyboard, approved, outcome)
    assert selected == [storyboard] and package.storyboard_input_ref == approved.artifacts.approved_storyboard
    prompts = providers.image.prompts + providers.video.text_prompts + [prompt for prompt, _ in providers.video.image_calls]
    assert "Unapproved replacement prompt" not in prompts
    assert all(not asset.shot_id.startswith("unapproved-") for asset in package.visual_assets)
    assert providers.tts.texts == [segment.narration for section in script.sections for segment in section.segments]


def test_phase7_exact_script_provenance_wins_over_newer_script(approved_project, monkeypatch):
    project, store, script, storyboard, approved = approved_project
    data = script.model_dump(mode="json")
    for section in data["sections"]:
        for segment in section["segments"]:
            segment["narration"] = "Unapproved newer Script narration"
    newer_version = store.save("script", ScriptPackage.model_validate(data))
    assert newer_version != storyboard.script_input_ref.version
    protect_phase7_boundary(monkeypatch, approved)
    outcome, providers, _ = run_media(project, store)
    assert_complete(project, store, script, storyboard, approved, outcome)
    assert providers.tts.texts == [segment.narration for section in script.sections for segment in section.segments]
    assert "Unapproved newer Script narration" not in providers.tts.texts


@pytest.mark.parametrize("target", ["narration", "visual"])
def test_phase7_tampered_final_media_restart_fails_closed(approved_project, monkeypatch, target):
    project, store, script, storyboard, approved = approved_project
    outcome, _, _ = run_media(project, store)
    package = assert_complete(project, store, script, storyboard, approved, outcome)
    alternate_ref, _ = seed_orphan(store, script, storyboard, approved)  # Valid newer media cannot hide corruption.
    asset = package.narration_assets[0].asset if target == "narration" else package.visual_assets[0].asset
    binary_path = store.project_dir / "media" / asset.relative_path
    binary_path.write_bytes(binary_path.read_bytes() + b"tampered")
    assert hashlib.sha256(binary_path.read_bytes()).hexdigest() != asset.sha256
    state_bytes = (store.project_dir / ".runtime/state.json").read_bytes()
    protect_phase7_boundary(monkeypatch, approved, bound_media=outcome.state.artifacts.media,
                            forbidden_media_versions={alternate_ref.version})
    def forbidden(*args, **kwargs):
        pytest.fail("Exact-bound corruption must not trigger regeneration, discovery or replacement")
    monkeypatch.setattr(ArtifactStore, "list_versions", forbidden)
    monkeypatch.setattr(ArtifactStore, "save", forbidden)
    with pytest.raises(MediaIntegrityError, match="SHA-256"):
        MediaWorkflow(provider_factory=forbidden).run(project, ArtifactStore(store.project_dir),
                                                      media_store=MediaStore(store.project_dir / "media"))
    # Production ASSEMBLING authentication raises; it does not publish a new FAILED
    # snapshot or erase/replace the old binding. Protect those existing semantics.
    assert (store.project_dir / ".runtime/state.json").read_bytes() == state_bytes
    assert read_project(store.project_dir)[1] == outcome.state


def test_phase7_partial_generation_durable_binary_does_not_create_authority(approved_project, monkeypatch):
    project, store, _, _, approved = approved_project
    protect_phase7_boundary(monkeypatch, approved)
    outcome, providers, selected = run_media(project, store, fail_tts_at=2)
    assert len(selected) == 1 and len(providers.tts.texts) == 2
    assert outcome.error_type == "RuntimeError"
    assert outcome.state.current_state == S.FAILED and outcome.state.failed_state == S.GENERATING_MEDIA
    assert outcome.state.last_successful_state == S.STORYBOARD_APPROVED
    assert outcome.state.artifacts == approved.artifacts and outcome.state.artifacts.media is None
    assert read_project(store.project_dir)[1] == outcome.state
    assert store.list_versions("media") == [] and outcome.package is None
    media_store = MediaStore(store.project_dir / "media")
    binaries = [path for path in media_store.root.rglob("*") if path.is_file()]
    assert len(binaries) == 1 and binaries[0].suffix == ".wav"
    data = binaries[0].read_bytes()
    assert data == wav_payload(providers.tts.texts[0]) and actual_wav_duration(data) > 0
    witness = MediaAssetReference(asset_id="orphan-witness", relative_path=binaries[0].relative_to(media_store.root).as_posix(),
                                  media_type=MediaType.AUDIO, sha256=hashlib.sha256(data).hexdigest())
    assert media_store.verify(witness)  # Complete durable bytes exist, but no accepted MediaPackage does.
    assert providers.image.prompts == [] and providers.video.text_prompts == [] and providers.video.image_calls == []
