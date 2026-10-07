"""P7-B local-only binary storage, WAV timing and exact narration execution."""
import hashlib
import io
import json
import os
import struct
import wave
from concurrent.futures import ThreadPoolExecutor
from dataclasses import fields
from pathlib import Path

import pytest
from pydantic import ValidationError

from history_studio.media import (
    AudioDurationError, TTSResult, generate_narration_assets, measure_wav_duration,
)
from history_studio.models import (
    ArtifactReference, MediaAssetReference, MediaType, NarrationAsset,
    ScriptPackage, StoryboardPackage,
)
from history_studio.storage import (
    ArtifactNotFoundError, ArtifactStore, MediaAssetConflictError, MediaIntegrityError, MediaStore,
)


def wav_bytes(*, frames=2000, rate=8000, channels=1, width=2):
    buffer = io.BytesIO()
    with wave.open(buffer, "wb") as audio:
        audio.setnchannels(channels)
        audio.setsampwidth(width)
        audio.setframerate(rate)
        audio.writeframes(b"\x00" * frames * channels * width)
    return buffer.getvalue()


class FakeTTSProvider:
    """Only generates deterministic PCM bytes; exposes no duration or LLM API."""

    def __init__(self, *, data=None, fail_at=None):
        self.data = wav_bytes() if data is None else data
        self.fail_at = fail_at
        self.texts = []

    def synthesize(self, *, text):
        self.texts.append(text)
        if len(self.texts) == self.fail_at:
            raise RuntimeError("simulated provider failure")
        return TTSResult(audio_bytes=self.data, provider="fake", model="pcm-v1")


def save(store, **changes):
    return store.save_bytes(**(dict(asset_id="audio-01", relative_path="audio/nested/seg-01.wav",
                                   media_type=MediaType.AUDIO, data=wav_bytes()) | changes))


def test_save_exact_bytes_hash_nested_path_read_and_verify(tmp_path):
    store = MediaStore(tmp_path / "media")
    reference = save(store)
    path = store.root / reference.relative_path
    assert path.read_bytes() == wav_bytes()
    assert reference.relative_path == "audio/nested/seg-01.wav"
    assert reference.sha256 == hashlib.sha256(path.read_bytes()).hexdigest()
    assert reference.media_type == MediaType.AUDIO
    assert store.read_bytes(reference) == wav_bytes()
    assert store.verify(reference) is True
    assert json.loads(reference.model_dump_json()) == dict(
        asset_id="audio-01", relative_path="audio/nested/seg-01.wav", media_type="AUDIO",
        sha256=hashlib.sha256(wav_bytes()).hexdigest())
    assert not list(store.root.rglob(".pending-*"))


@pytest.mark.parametrize("path", ["/absolute.wav", "C:/absolute.wav", "C:\\file.wav",
                                      "../escape.wav", "audio/../../escape.wav", "audio\\file.wav"])
def test_invalid_paths_fail_before_writes(tmp_path, path):
    store = MediaStore(tmp_path / "media")
    with pytest.raises(ValidationError):
        save(store, relative_path=path)
    assert not store.root.exists()


@pytest.mark.parametrize("data", [b"", None, "audio", bytearray(b"audio")])
def test_store_requires_nonempty_immutable_bytes(tmp_path, data):
    with pytest.raises(ValueError, match="non-empty bytes"):
        save(MediaStore(tmp_path), data=data)


def test_tampered_file_and_wrong_expected_hash_fail_closed(tmp_path):
    store = MediaStore(tmp_path)
    reference = save(store)
    wrong = MediaAssetReference.model_validate(reference.model_dump() | {"sha256": "0" * 64})
    with pytest.raises(MediaIntegrityError, match="SHA-256"):
        store.read_bytes(wrong)
    (store.root / reference.relative_path).write_bytes(b"tampered")
    with pytest.raises(MediaIntegrityError):
        store.verify(reference)


def test_missing_exact_asset_and_unvalidated_reference(tmp_path):
    store = MediaStore(tmp_path)
    reference = save(store)
    (store.root / reference.relative_path).unlink()
    with pytest.raises(FileNotFoundError):
        store.read_bytes(reference)
    with pytest.raises(ValidationError):
        store.read_bytes(reference.model_copy(update={"relative_path": "../escape.wav"}))


def test_idempotent_save_and_conflict_preserve_destination(tmp_path):
    store = MediaStore(tmp_path)
    first = save(store)
    assert save(store) == first
    with pytest.raises(MediaAssetConflictError):
        save(store, data=wav_bytes(frames=3000))
    assert store.read_bytes(first) == wav_bytes()
    assert not list(tmp_path.rglob(".pending-*"))


@pytest.mark.parametrize("operation", ["link", "fsync"])
def test_failed_atomic_write_does_not_publish_and_cleans_temp(tmp_path, monkeypatch, operation):
    def fail(*args):
        raise OSError("simulated atomic write failure")
    monkeypatch.setattr(f"history_studio.storage.media_store.os.{operation}", fail)
    with pytest.raises(OSError):
        save(MediaStore(tmp_path))
    assert not list(tmp_path.rglob("*.wav"))
    assert not list(tmp_path.rglob(".pending-*"))


def test_concurrent_identical_publications_are_idempotent(tmp_path):
    store = MediaStore(tmp_path)
    with ThreadPoolExecutor(max_workers=4) as executor:
        references = list(executor.map(lambda _: save(store), range(8)))
    assert len(set(references)) == 1
    assert store.read_bytes(references[0]) == wav_bytes()


@pytest.mark.skipif(os.name != "nt", reason="Windows extended-path representation")
def test_windows_extended_resolved_path_is_same_contained_path(tmp_path, monkeypatch):
    store = MediaStore(tmp_path)
    original_resolve = Path.resolve
    def extended_resolve(path, *args, **kwargs):
        resolved = original_resolve(path, *args, **kwargs)
        return Path("\\\\?\\" + str(resolved))
    monkeypatch.setattr(Path, "resolve", extended_resolve)
    reference = save(store)
    assert store.read_bytes(reference) == wav_bytes()


def test_concurrent_conflicting_publications_never_overwrite(tmp_path):
    store = MediaStore(tmp_path)
    def publish(frames):
        try:
            return save(store, data=wav_bytes(frames=frames))
        except MediaAssetConflictError:
            return None
    with ThreadPoolExecutor(max_workers=2) as executor:
        references = list(executor.map(publish, [2000, 3000]))
    successful = [reference for reference in references if reference is not None]
    assert len(successful) == 1
    assert store.verify(successful[0])
    assert not list(tmp_path.rglob(".pending-*"))


@pytest.mark.parametrize("link_directory", [True, False])
def test_symlink_paths_fail_closed(tmp_path, link_directory):
    outside = tmp_path / "outside"
    outside.mkdir()
    root = tmp_path / "media"
    root.mkdir()
    relative_path = "linked/file.wav" if link_directory else "linked.wav"
    target = outside if link_directory else outside / "file.wav"
    if not link_directory:
        target.write_bytes(wav_bytes())
    link = root / ("linked" if link_directory else "linked.wav")
    try:
        link.symlink_to(target, target_is_directory=link_directory)
    except OSError:
        pytest.skip("OS does not permit local test symlink creation")
    store = MediaStore(root)
    with pytest.raises(ValueError, match="links or junctions"):
        save(store, relative_path=relative_path)
    reference = MediaAssetReference(asset_id="audio", relative_path=relative_path,
                                    media_type="AUDIO", sha256=hashlib.sha256(wav_bytes()).hexdigest())
    with pytest.raises(ValueError, match="links or junctions"):
        store.read_bytes(reference)
    assert list(outside.iterdir()) == ([] if link_directory else [target])


def test_fake_provider_is_deterministic_audio_without_duration():
    provider = FakeTTSProvider()
    first = provider.synthesize(text="Exact narration")
    assert first == provider.synthesize(text="Exact narration")
    assert {field.name for field in fields(TTSResult)} == {"audio_bytes", "provider", "model", "audio_format"}
    assert measure_wav_duration(first.audio_bytes) == 0.25


@pytest.mark.parametrize("frames,rate,channels,width", [(1000, 8000, 1, 1), (12345, 44100, 2, 2), (16000, 16000, 1, 3)])
def test_actual_pcm_frame_duration(frames, rate, channels, width):
    duration = measure_wav_duration(wav_bytes(frames=frames, rate=rate, channels=channels, width=width))
    assert duration == frames / rate > 0


@pytest.mark.parametrize("data", [b"", b"not audio", b"RIFF" + b"\x00" * 20,
                                      wav_bytes(frames=0), wav_bytes()[:-1]])
def test_invalid_empty_zero_or_truncated_audio(data):
    with pytest.raises(AudioDurationError):
        measure_wav_duration(data)


def test_declared_frame_count_cannot_supply_missing_actual_audio():
    data = bytearray(wav_bytes())
    struct.pack_into("<I", data, 40, len(data) * 2)
    with pytest.raises(AudioDurationError, match="truncated or incomplete"):
        measure_wav_duration(bytes(data))


def test_partial_pcm_frame_rejected():
    data = bytearray(wav_bytes())
    data.append(0)
    struct.pack_into("<I", data, 4, len(data) - 8)
    struct.pack_into("<I", data, 40, len(data) - 44)
    with pytest.raises(AudioDurationError, match="incomplete"):
        measure_wav_duration(bytes(data))


def script_package():
    return ScriptPackage.model_validate(dict(
        story_input_ref=dict(project_id="project", artifact_type="story", version=3), title="Documentary",
        sections=[dict(section_id="first", title="First", segments=[
            dict(segment_id="SEG-Z", kind="STRUCTURAL", narration="Exact first narration — 李白。"),
            dict(segment_id="unused", kind="STRUCTURAL", narration="Not represented"),
        ]), dict(section_id="second", title="Second", segments=[
            dict(segment_id="SEG-A", kind="STRUCTURAL", narration="Second\nline of narration."),
        ])],
    ))


def storyboard_package(*, script_version=1, project_id="project", segment_ids=("SEG-A", "SEG-Z", "SEG-Z")):
    return StoryboardPackage.model_validate(dict(
        script_input_ref=dict(project_id=project_id, artifact_type="script", version=script_version),
        title="Documentary", sections=[dict(section_id="visuals", title="Visuals", shots=[
            dict(shot_id=f"SHOT-{index}", kind="STRUCTURAL", source_segment_id=segment_id,
                 visual_description="River", generation_prompt="River", generation_method="STATIC_IMAGE",
                 framing="WIDE", camera_motion="NONE", estimated_duration_seconds=99)
            for index, segment_id in enumerate(segment_ids)
        ])],
    ))


@pytest.fixture
def execution(tmp_path):
    store = ArtifactStore(tmp_path / "project")
    store.save("script", script_package())
    store.save("storyboard", storyboard_package())
    reference = ArtifactReference(project_id="project", artifact_type="storyboard", version=1)
    return dict(storyboard_input_ref=reference, artifact_store=store,
                provider=FakeTTSProvider(), media_store=MediaStore(store.project_dir / "media"))


def test_generation_once_per_segment_in_script_order_with_exact_text_and_actual_duration(execution):
    assets = generate_narration_assets(**execution)
    assert [asset.segment_id for asset in assets] == ["SEG-Z", "SEG-A"]
    assert execution["provider"].texts == ["Exact first narration — 李白。", "Second\nline of narration."]
    assert len(assets) == 2  # Three shots, two represented segments, one unrepresented segment.
    for asset in assets:
        assert asset.asset.media_type == MediaType.AUDIO
        assert asset.duration_seconds == 0.25 != 99
        assert asset.generation_metadata.provider == "fake"
        assert asset.generation_metadata.model == "pcm-v1"
        assert NarrationAsset.model_validate_json(asset.model_dump_json()) == asset
        assert measure_wav_duration(execution["media_store"].read_bytes(asset.asset)) == asset.duration_seconds
    assert len({asset.asset.asset_id for asset in assets}) == 2


def test_single_represented_segment(execution):
    store = execution["artifact_store"]
    version = store.save("storyboard", storyboard_package(segment_ids=("SEG-A",)))
    execution["storyboard_input_ref"] = execution["storyboard_input_ref"].model_copy(update={"version": version})
    assets = generate_narration_assets(**execution)
    assert [asset.segment_id for asset in assets] == ["SEG-A"]


def test_deterministic_identity_path_and_idempotent_retry(execution):
    first = generate_narration_assets(**execution)
    second = generate_narration_assets(**execution)
    assert second == first
    assert len(execution["provider"].texts) == 4  # Retry is explicit resynthesis, not discovery.
    for asset in first:
        identity = json.dumps([
            execution["storyboard_input_ref"].model_dump(mode="json"),
            dict(project_id="project", artifact_type="script", version=1), asset.segment_id,
        ], sort_keys=True, separators=(",", ":"))
        digest = hashlib.sha256(identity.encode("utf-8")).hexdigest()
        assert asset.asset.asset_id == f"nar-{digest}"
        assert asset.asset.relative_path == f"audio/storyboard-v1/{digest}.wav"


def test_changed_retry_bytes_fail_without_overwrite(execution):
    first = generate_narration_assets(**execution)
    execution["provider"] = FakeTTSProvider(data=wav_bytes(frames=4000))
    with pytest.raises(MediaAssetConflictError):
        generate_narration_assets(**execution)
    assert execution["media_store"].read_bytes(first[0].asset) == wav_bytes()


def test_exact_versions_no_latest_substitution_or_workflow_mutation(execution, monkeypatch):
    store = execution["artifact_store"]
    newer = script_package().model_dump(mode="json")
    newer["sections"][0]["segments"][0]["narration"] = "Newer forbidden text"
    store.save("script", ScriptPackage.model_validate(newer))
    store.save("storyboard", storyboard_package(script_version=2))
    state_path = store.project_dir / ".runtime/state.json"
    state_path.parent.mkdir()
    state_path.write_bytes(b"unchanged runtime state")
    snapshots = {path: path.read_bytes() for path in store.project_dir.rglob("*.json")}
    original_load = store.load
    loaded = []
    def exact_load(artifact_type, version, model_type):
        loaded.append((artifact_type, version))
        return original_load(artifact_type, version, model_type)
    def forbidden(*args, **kwargs):
        raise AssertionError("Latest/discovery/LLM API must never run")
    monkeypatch.setattr(store, "load", exact_load)
    monkeypatch.setattr(store, "load_latest", forbidden)
    monkeypatch.setattr(store, "list_versions", forbidden)
    monkeypatch.setattr("history_studio.model_io.ModelProvider.decide", forbidden)
    generate_narration_assets(**execution)
    assert loaded == [("storyboard", 1), ("script", 1)]
    assert "Newer forbidden text" not in execution["provider"].texts
    assert {path: path.read_bytes() for path in store.project_dir.rglob("*.json")} == snapshots
    assert {path.suffix for path in execution["media_store"].root.rglob("*") if path.is_file()} == {".wav"}


@pytest.mark.parametrize("changes", [dict(project_id="other"), dict(artifact_type="script"), dict(version=999)])
def test_wrong_reference_or_missing_exact_storyboard_fails_before_provider(execution, changes):
    execution["storyboard_input_ref"] = execution["storyboard_input_ref"].model_copy(update=changes)
    with pytest.raises((ValueError, ArtifactNotFoundError)):
        generate_narration_assets(**execution)
    assert execution["provider"].texts == []


@pytest.mark.parametrize("board", [
    storyboard_package(project_id="other"), storyboard_package(script_version=999),
    storyboard_package(segment_ids=("unknown", "SEG-Z")),
])
def test_bad_script_provenance_missing_version_or_unknown_segment(execution, board):
    version = execution["artifact_store"].save("storyboard", board)
    execution["storyboard_input_ref"] = execution["storyboard_input_ref"].model_copy(update={"version": version})
    with pytest.raises((ValueError, ArtifactNotFoundError)):
        generate_narration_assets(**execution)
    assert execution["provider"].texts == []
    assert not execution["media_store"].root.exists()


@pytest.mark.parametrize("corruption", ["duplicate_segment", "empty_narration", "wrong_project", "duplicate_shot"])
def test_corrupt_stored_contracts_fail_before_tts(execution, corruption):
    store = execution["artifact_store"]
    artifact_type = "storyboard" if corruption == "duplicate_shot" else "script"
    path = store.project_dir / artifact_type / f"{artifact_type}_v1.json"
    data = json.loads(path.read_text(encoding="utf-8"))
    if corruption == "duplicate_segment":
        data["sections"][1]["segments"][0]["segment_id"] = "SEG-Z"
    elif corruption == "empty_narration":
        data["sections"][0]["segments"][0]["narration"] = " "
    elif corruption == "wrong_project":
        data["story_input_ref"]["project_id"] = "other"
    else:
        data["sections"][0]["shots"][1]["shot_id"] = "SHOT-0"
    path.write_text(json.dumps(data), encoding="utf-8")
    with pytest.raises(ValueError):
        generate_narration_assets(**execution)
    assert execution["provider"].texts == []


@pytest.mark.parametrize("data", [b"", b"malformed", wav_bytes(frames=0)])
def test_invalid_provider_audio_never_publishes_asset(execution, data):
    execution["provider"] = FakeTTSProvider(data=data)
    with pytest.raises(AudioDurationError):
        generate_narration_assets(**execution)
    assert not execution["media_store"].root.exists()


@pytest.mark.parametrize("fail_at", [1, 2])
def test_provider_failure_propagates_without_partial_success(execution, fail_at):
    execution["provider"] = FakeTTSProvider(fail_at=fail_at)
    with pytest.raises(RuntimeError, match="simulated provider failure"):
        generate_narration_assets(**execution)
    files = list(execution["media_store"].root.rglob("*.wav"))
    assert len(files) == fail_at - 1  # Explicit orphan publication, never a successful partial tuple.


@pytest.mark.parametrize("result", [
    TTSResult(audio_bytes=wav_bytes(), provider="fake", model="pcm", audio_format="mp3"),
    TTSResult(audio_bytes=wav_bytes(), provider="", model="pcm"), None,
])
def test_bad_provider_result_or_metadata_fails_before_publication(execution, result):
    class BadProvider:
        def synthesize(self, *, text):
            return result
    execution["provider"] = BadProvider()
    with pytest.raises(ValueError):
        generate_narration_assets(**execution)
    assert not execution["media_store"].root.exists()


def test_saved_audio_integrity_is_checked_before_success(execution, monkeypatch):
    media_store = execution["media_store"]
    original = media_store.save_bytes
    def tamper(**kwargs):
        reference = original(**kwargs)
        (media_store.root / reference.relative_path).write_bytes(b"tampered after publication")
        return reference
    monkeypatch.setattr(media_store, "save_bytes", tamper)
    with pytest.raises(MediaIntegrityError):
        generate_narration_assets(**execution)


def test_new_storyboard_version_gets_distinct_immutable_identity(execution):
    first = generate_narration_assets(**execution)
    version = execution["artifact_store"].save("storyboard", storyboard_package())
    execution["storyboard_input_ref"] = execution["storyboard_input_ref"].model_copy(update={"version": version})
    second = generate_narration_assets(**execution)
    assert {asset.asset.asset_id for asset in first}.isdisjoint(asset.asset.asset_id for asset in second)
    assert {asset.asset.relative_path for asset in first}.isdisjoint(asset.asset.relative_path for asset in second)
