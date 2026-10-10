"""P7-C offline visual execution and immutable binary publication tests."""
import hashlib
import json
import struct
import zlib
from dataclasses import fields

import pytest
from pydantic import ValidationError

from history_studio.media import ImageGenerationResult, VideoGenerationResult, generate_visual_assets
from history_studio.models import ArtifactReference, GenerationMethod, MediaType, StoryboardPackage, VisualAsset
from history_studio.storage import ArtifactNotFoundError, ArtifactStore, MediaAssetConflictError, MediaIntegrityError, MediaStore


def png_bytes(color=b"\x12\x34\x56"):
    """Valid deterministic one-pixel RGB PNG, using standard-library encoding."""
    def chunk(kind, data):
        return struct.pack(">I", len(data)) + kind + data + struct.pack(">I", zlib.crc32(kind + data))
    return (b"\x89PNG\r\n\x1a\n" + chunk(b"IHDR", struct.pack(">IIBBBBB", 1, 1, 8, 2, 0, 0, 0))
            + chunk(b"IDAT", zlib.compress(b"\x00" + color)) + chunk(b"IEND", b""))


def mp4_bytes(payload=b"deterministic fixture"):
    """Minimal ISO BMFF container fixture; tests inspect bytes, not video playback."""
    def box(kind, data):
        return struct.pack(">I", len(data) + 8) + kind + data
    return box(b"ftyp", b"isom\x00\x00\x02\x00isommp42") + box(b"free", payload)


class FakeImageProvider:
    def __init__(self, events, *, data=None, fail_at=None):
        self.events = events
        self.data = png_bytes() if data is None else data
        self.fail_at = fail_at
        self.prompts = []

    def recovery_identity(self):
        return dict(version=1, complete=True, implementation="png-v1",
                    settings={"data_sha256": hashlib.sha256(self.data).hexdigest()})

    def generate(self, *, prompt):
        self.prompts.append(prompt)
        self.events.append(("image", prompt))
        if len(self.prompts) == self.fail_at:
            raise RuntimeError("image provider failed")
        return ImageGenerationResult(image_bytes=self.data, provider="fake-image", model="png-v1")


class FakeVideoProvider:
    def __init__(self, events, *, data=None, fail_at=None):
        self.events = events
        self.data = mp4_bytes() if data is None else data
        self.fail_at = fail_at
        self.calls = []

    def recovery_identity(self):
        return dict(version=1, complete=True, implementation="mp4-v1",
                    settings={"data_sha256": hashlib.sha256(self.data).hexdigest()})

    def _generate(self, kind, prompt, image=None):
        self.calls.append((kind, prompt, image))
        self.events.append((kind, prompt, image))
        if len(self.calls) == self.fail_at:
            raise RuntimeError("video provider failed")
        return VideoGenerationResult(video_bytes=self.data, provider="fake-video", model="mp4-v1")

    def generate_from_text(self, *, prompt):
        return self._generate("text-video", prompt)

    def generate_from_image(self, *, image, prompt):
        return self._generate("image-video", prompt, image)


def shot(shot_id="SHOT-Z", method="STATIC_IMAGE", **changes):
    return dict(shot_id=shot_id, kind="STRUCTURAL", source_segment_id="SEG-01",
                visual_description="Description must never be the prompt", generation_prompt=f"Exact {shot_id} prompt — 李白\nwith  inner spaces.",
                generation_method=method, framing="WIDE", camera_motion="NONE",
                estimated_duration_seconds=77) | changes


def board(shots=None):
    return StoryboardPackage.model_validate(dict(
        script_input_ref=dict(project_id="project", artifact_type="script", version=9),
        title="Documentary", sections=[dict(section_id="first", title="First", shots=shots or [shot()])],
    ))


@pytest.fixture
def execution(tmp_path):
    store = ArtifactStore(tmp_path / "project")
    store.save("storyboard", board())
    events = []
    return dict(storyboard_input_ref=ArtifactReference(project_id="project", artifact_type="storyboard", version=1),
                artifact_store=store, media_store=MediaStore(store.project_dir / "media"),
                image_provider=FakeImageProvider(events), video_provider=FakeVideoProvider(events))


def select_board(execution, package):
    version = execution["artifact_store"].save("storyboard", package)
    execution["storyboard_input_ref"] = execution["storyboard_input_ref"].model_copy(update={"version": version})


def test_static_image_uses_exact_prompt_persists_image_and_metadata(execution):
    execution["video_provider"] = None
    assets = generate_visual_assets(**execution)
    assert len(assets) == 1
    final = assets[0]
    assert execution["image_provider"].prompts == [shot()["generation_prompt"]]
    assert shot()["visual_description"] not in execution["image_provider"].prompts
    assert final.shot_id == "SHOT-Z"
    assert final.source_segment_id == "SEG-01"
    assert final.generation_method == GenerationMethod.STATIC_IMAGE
    assert final.asset.media_type == MediaType.IMAGE
    assert execution["media_store"].read_bytes(final.asset) == png_bytes()
    assert final.generation_metadata.provider == "fake-image"
    assert final.generation_metadata.model == "png-v1"
    assert VisualAsset.model_validate_json(final.model_dump_json()) == final


def test_text_video_uses_exact_prompt_without_image_or_intermediate(execution):
    select_board(execution, board([shot(method="TEXT_TO_VIDEO")]))
    image_provider = execution["image_provider"]
    execution["image_provider"] = None
    assets = generate_visual_assets(**execution)
    assert image_provider.prompts == []
    assert execution["video_provider"].calls == [("text-video", shot()["generation_prompt"], None)]
    assert len(assets) == 1
    assert assets[0].asset.media_type == MediaType.VIDEO
    assert assets[0].generation_method == GenerationMethod.TEXT_TO_VIDEO
    assert execution["media_store"].read_bytes(assets[0].asset) == mp4_bytes()
    assert assets[0].generation_metadata.provider == "fake-video"
    assert not list(execution["media_store"].root.rglob("*.png"))


def test_image_video_persists_and_verifies_image_before_video_then_returns_one_final(execution, monkeypatch):
    select_board(execution, board([shot(method="IMAGE_TO_VIDEO")]))
    media_store = execution["media_store"]
    events = execution["image_provider"].events
    original_save = media_store.save_bytes
    original_read = media_store.read_bytes
    saved = []
    def save(**kwargs):
        reference = original_save(**kwargs)
        saved.append(reference)
        events.append(("persisted", reference.media_type))
        return reference
    def read(reference):
        data = original_read(reference)
        events.append(("verified", reference.media_type))
        return data
    monkeypatch.setattr(media_store, "save_bytes", save)
    monkeypatch.setattr(media_store, "read_bytes", read)
    assets = generate_visual_assets(**execution)
    assert len(assets) == 1
    assert [reference.media_type for reference in saved] == [MediaType.IMAGE, MediaType.VIDEO]
    intermediate = saved[0]
    assert intermediate.asset_id.startswith("img-")
    assert "/intermediate/" in intermediate.relative_path
    assert original_read(intermediate) == png_bytes()
    assert execution["video_provider"].calls == [("image-video", shot()["generation_prompt"], original_read(intermediate))]
    video_index = next(index for index, event in enumerate(events) if event[0] == "image-video")
    persisted_index = events.index(("persisted", MediaType.IMAGE))
    assert events[0] == ("image", shot()["generation_prompt"])
    assert persisted_index < video_index
    assert ("verified", MediaType.IMAGE) in events[persisted_index + 1:video_index]
    final = assets[0]
    assert final.asset == saved[1]
    assert final.asset.media_type == MediaType.VIDEO
    assert final.generation_method == GenerationMethod.IMAGE_TO_VIDEO
    assert final.generation_metadata.provider == "fake-video"
    assert final.generation_metadata.model == "mp4-v1"
    assert original_read(final.asset) == mp4_bytes()


def test_all_shots_and_sections_preserve_presentation_order_and_kind(execution):
    data = board([shot("z", "STATIC_IMAGE", kind="HISTORICAL"), shot("a", "TEXT_TO_VIDEO")]).model_dump(mode="json")
    data["sections"].append(dict(section_id="second", title="Second", shots=[shot("b", "IMAGE_TO_VIDEO")]))
    select_board(execution, StoryboardPackage.model_validate(data))
    assets = generate_visual_assets(**execution)
    assert [asset.shot_id for asset in assets] == ["z", "a", "b"]
    assert [asset.asset.media_type for asset in assets] == [MediaType.IMAGE, MediaType.VIDEO, MediaType.VIDEO]
    assert len({asset.asset.asset_id for asset in assets}) == 3
    assert len(list(execution["media_store"].root.rglob("*.png"))) == 2
    assert len(list(execution["media_store"].root.rglob("*.mp4"))) == 2


@pytest.mark.parametrize("method", list(GenerationMethod))
def test_deterministic_identity_path_hash_and_idempotent_retry(execution, method):
    select_board(execution, board([shot(method=method)]))
    first = generate_visual_assets(**execution)
    assert generate_visual_assets(**execution) == first
    reference = execution["storyboard_input_ref"]
    identity = json.dumps([reference.model_dump(mode="json"),
                          dict(project_id="project", artifact_type="script", version=9), "SHOT-Z"],
                         sort_keys=True, separators=(",", ":"))
    digest = hashlib.sha256(identity.encode("utf-8")).hexdigest()
    asset = first[0].asset
    assert asset.asset_id == f"vis-{digest}"
    is_image = method == GenerationMethod.STATIC_IMAGE
    directory, extension = ("images", "png") if is_image else ("video", "mp4")
    assert asset.relative_path == f"{directory}/storyboard-v{reference.version}/{digest}.{extension}"
    assert not asset.relative_path.startswith(("/", "\\")) and ":" not in asset.relative_path
    assert asset.sha256 == hashlib.sha256(png_bytes() if is_image else mp4_bytes()).hexdigest()
    if method == GenerationMethod.IMAGE_TO_VIDEO:
        intermediate = execution["media_store"].root / f"images/storyboard-v{reference.version}/intermediate/{digest}.png"
        assert intermediate.read_bytes() == png_bytes()
    assert len(execution["image_provider"].prompts) == (0 if method == GenerationMethod.TEXT_TO_VIDEO else 2)


@pytest.mark.parametrize("method", list(GenerationMethod))
def test_conflicting_retry_fails_and_original_bytes_survive(execution, method):
    select_board(execution, board([shot(method=method)]))
    final = generate_visual_assets(**execution)[0]
    if method == GenerationMethod.STATIC_IMAGE:
        execution["image_provider"].data = png_bytes(b"\x00\x00\x00")
    else:
        execution["video_provider"].data = mp4_bytes(b"different bytes")
    with pytest.raises(MediaAssetConflictError):
        generate_visual_assets(**execution)
    assert execution["media_store"].verify(final.asset)


def test_intermediate_conflicting_retry_blocks_video(execution):
    select_board(execution, board([shot(method="IMAGE_TO_VIDEO")]))
    generate_visual_assets(**execution)
    execution["image_provider"].data = png_bytes(b"\x00\x00\x00")
    execution["video_provider"].calls.clear()
    with pytest.raises(MediaAssetConflictError):
        generate_visual_assets(**execution)
    assert execution["video_provider"].calls == []


def test_exact_storyboard_only_no_upstream_latest_tts_or_workflow_mutation(execution, monkeypatch):
    store = execution["artifact_store"]
    store.save("storyboard", board([shot(generation_prompt="Forbidden latest prompt")]))
    state_path = store.project_dir / ".runtime/state.json"
    state_path.parent.mkdir()
    state_path.write_bytes(b"unchanged workflow state")
    original_bytes = {path: path.read_bytes() for path in store.project_dir.rglob("*.json")}
    original_load = store.load
    loaded = []
    def load(artifact_type, version, model_type):
        loaded.append((artifact_type, version))
        assert (artifact_type, version, model_type) == ("storyboard", 1, StoryboardPackage)
        return original_load(artifact_type, version, model_type)
    def forbidden(*args, **kwargs):
        raise AssertionError("Discovery, LLM, TTS and workflow behavior are forbidden")
    monkeypatch.setattr(store, "load", load)
    monkeypatch.setattr(store, "load_latest", forbidden)
    monkeypatch.setattr(store, "list_versions", forbidden)
    monkeypatch.setattr(store, "save", forbidden)
    monkeypatch.setattr("history_studio.model_io.ModelProvider.decide", forbidden)
    monkeypatch.setattr("history_studio.media.narration.generate_narration_assets", forbidden)
    monkeypatch.setattr("history_studio.storage.artifact_store.write_json", forbidden)
    generate_visual_assets(**execution)
    assert loaded == [("storyboard", 1)]
    assert execution["image_provider"].prompts == [shot()["generation_prompt"]]
    assert {path: path.read_bytes() for path in store.project_dir.rglob("*.json")} == original_bytes


@pytest.mark.parametrize("changes", [dict(project_id="other"), dict(artifact_type="script"), dict(version=999), dict(version=True)])
def test_wrong_or_missing_exact_reference_fails_before_generation(execution, changes):
    execution["storyboard_input_ref"] = execution["storyboard_input_ref"].model_copy(update=changes)
    with pytest.raises((ValueError, ArtifactNotFoundError)):
        generate_visual_assets(**execution)
    assert execution["image_provider"].events == []


@pytest.mark.parametrize("corruption", ["empty_prompt", "unsupported_method", "duplicate_shot", "wrong_project"])
def test_all_input_validation_precedes_providers(execution, corruption):
    store = execution["artifact_store"]
    path = store.project_dir / "storyboard/storyboard_v1.json"
    data = board([shot(), shot("later")]).model_dump(mode="json")
    later = data["sections"][0]["shots"][1]
    if corruption == "empty_prompt":
        later["generation_prompt"] = " \n "
    elif corruption == "unsupported_method":
        later["generation_method"] = "UNKNOWN"
    elif corruption == "duplicate_shot":
        later["shot_id"] = "SHOT-Z"
    else:
        data["script_input_ref"]["project_id"] = "other"
    path.write_text(json.dumps(data), encoding="utf-8")
    with pytest.raises(ValueError):
        generate_visual_assets(**execution)
    assert execution["image_provider"].events == []
    assert not execution["media_store"].root.exists()


@pytest.mark.parametrize("method,missing", [("STATIC_IMAGE", "image_provider"), ("TEXT_TO_VIDEO", "video_provider"),
                                            ("IMAGE_TO_VIDEO", "image_provider"), ("IMAGE_TO_VIDEO", "video_provider")])
def test_missing_required_provider_fails_before_any_generation(execution, method, missing):
    select_board(execution, board([shot(method=method)]))
    events = execution["image_provider"].events
    execution[missing] = None
    with pytest.raises(ValueError, match="require.*provider"):
        generate_visual_assets(**execution)
    assert events == []


@pytest.mark.parametrize("method,provider", [("STATIC_IMAGE", "image_provider"), ("TEXT_TO_VIDEO", "video_provider"),
                                             ("IMAGE_TO_VIDEO", "image_provider"), ("IMAGE_TO_VIDEO", "video_provider")])
def test_empty_provider_bytes_fail_without_success(execution, method, provider):
    select_board(execution, board([shot(method=method)]))
    execution[provider].data = b""
    with pytest.raises(ValueError, match="non-empty"):
        generate_visual_assets(**execution)
    assert not list(execution["media_store"].root.rglob("*.mp4"))
    expected_intermediate = method == "IMAGE_TO_VIDEO" and provider == "video_provider"
    assert len(list(execution["media_store"].root.rglob("*.png"))) == int(expected_intermediate)


@pytest.mark.parametrize("method,provider", [("STATIC_IMAGE", "image_provider"), ("TEXT_TO_VIDEO", "video_provider"),
                                             ("IMAGE_TO_VIDEO", "image_provider"), ("IMAGE_TO_VIDEO", "video_provider")])
def test_provider_failure_propagates(execution, method, provider):
    select_board(execution, board([shot(method=method)]))
    execution[provider].fail_at = 1
    with pytest.raises(RuntimeError, match="provider failed"):
        generate_visual_assets(**execution)


@pytest.mark.parametrize("corruption", ["tamper", "delete"])
def test_intermediate_integrity_or_missing_file_fails_before_video(execution, monkeypatch, corruption):
    select_board(execution, board([shot(method="IMAGE_TO_VIDEO")]))
    media_store = execution["media_store"]
    original_save = media_store.save_bytes
    def save(**kwargs):
        reference = original_save(**kwargs)
        path = media_store.root / reference.relative_path
        if corruption == "tamper":
            path.write_bytes(b"corrupted image")
        else:
            path.unlink()
        return reference
    monkeypatch.setattr(media_store, "save_bytes", save)
    with pytest.raises(MediaIntegrityError if corruption == "tamper" else FileNotFoundError):
        generate_visual_assets(**execution)
    assert execution["video_provider"].calls == []


@pytest.mark.parametrize("method,provider", [("STATIC_IMAGE", "image_provider"), ("TEXT_TO_VIDEO", "video_provider")])
def test_later_shot_failure_retains_earlier_orphan_without_partial_success(execution, method, provider):
    select_board(execution, board([shot("first", method), shot("later", method)]))
    execution[provider].fail_at = 2
    with pytest.raises(RuntimeError):
        generate_visual_assets(**execution)
    files = [path for path in execution["media_store"].root.rglob("*") if path.is_file()]
    assert len(files) == 1
    assert files[0].read_bytes() == (png_bytes() if method == "STATIC_IMAGE" else mp4_bytes())


@pytest.mark.parametrize("method,result", [
    ("STATIC_IMAGE", VideoGenerationResult(video_bytes=mp4_bytes(), provider="fake", model="mp4")),
    ("TEXT_TO_VIDEO", ImageGenerationResult(image_bytes=png_bytes(), provider="fake", model="png")),
    ("STATIC_IMAGE", ImageGenerationResult(image_bytes=png_bytes(), provider="fake", model="png", image_format="jpeg")),
    ("TEXT_TO_VIDEO", VideoGenerationResult(video_bytes=mp4_bytes(), provider="fake", model="mp4", video_format="webm")),
    ("STATIC_IMAGE", ImageGenerationResult(image_bytes=png_bytes(), provider="", model="png")),
    ("TEXT_TO_VIDEO", VideoGenerationResult(video_bytes=mp4_bytes(), provider="fake", model="")),
    ("STATIC_IMAGE", ImageGenerationResult(image_bytes="not bytes", provider="fake", model="png")),
    ("TEXT_TO_VIDEO", VideoGenerationResult(video_bytes=bytearray(b"not immutable"), provider="fake", model="mp4")),
])
def test_bad_result_media_type_format_metadata_or_bytes(execution, method, result):
    select_board(execution, board([shot(method=method)]))
    class BadProvider:
        def generate(self, *, prompt):
            return result
        def generate_from_text(self, *, prompt):
            return result
    execution["image_provider" if method == "STATIC_IMAGE" else "video_provider"] = BadProvider()
    with pytest.raises(ValueError):
        generate_visual_assets(**execution)
    assert not execution["media_store"].root.exists()


@pytest.mark.parametrize("method", ["STATIC_IMAGE", "TEXT_TO_VIDEO"])
def test_wrong_stored_media_type_fails(execution, monkeypatch, method):
    select_board(execution, board([shot(method=method)]))
    original_save = execution["media_store"].save_bytes
    def save(**kwargs):
        return original_save(**kwargs).model_copy(update={"media_type": MediaType.AUDIO})
    monkeypatch.setattr(execution["media_store"], "save_bytes", save)
    with pytest.raises(ValueError, match="media type"):
        generate_visual_assets(**execution)


def test_unsafe_storage_path_failure_propagates(execution, monkeypatch):
    original_save = execution["media_store"].save_bytes
    def save(**kwargs):
        return original_save(**(kwargs | {"relative_path": "../outside.png"}))
    monkeypatch.setattr(execution["media_store"], "save_bytes", save)
    with pytest.raises(ValidationError):
        generate_visual_assets(**execution)
    assert not execution["media_store"].root.exists()


def test_results_have_no_provider_hash_and_final_has_no_binary_prompt_or_timing():
    assert {field.name for field in fields(ImageGenerationResult)} == {"image_bytes", "image_format", "provider", "model"}
    assert {field.name for field in fields(VideoGenerationResult)} == {"video_bytes", "video_format", "provider", "model"}
    assert set(VisualAsset.model_fields) == {"shot_id", "source_segment_id", "asset", "generation_method", "generation_metadata"}


def test_new_storyboard_version_has_distinct_asset_identity(execution):
    first = generate_visual_assets(**execution)[0]
    select_board(execution, board())
    second = generate_visual_assets(**execution)[0]
    assert first.asset.asset_id != second.asset.asset_id
    assert first.asset.relative_path != second.asset.relative_path


def test_case_distinct_shots_have_filesystem_safe_distinct_paths(execution):
    select_board(execution, board([shot("SHOT-A"), shot("shot-a")]))
    assets = generate_visual_assets(**execution)
    assert len({asset.asset.relative_path.lower() for asset in assets}) == 2
