import json

import pytest
from pydantic import ValidationError

from history_studio.models import (
    ArtifactReference, GenerationMetadata, GenerationMethod, MediaAssetReference,
    MediaPackage, MediaType, NarrationAsset, StoryboardShot, VisualAsset,
)


def asset(asset_id="audio_01", media_type="AUDIO", **changes):
    return MediaAssetReference(**(dict(asset_id=asset_id, relative_path="audio/seg-07.mp3",
                                      media_type=media_type, sha256="ab" * 32) | changes))


def narration(segment_id="segment_01", **changes):
    return NarrationAsset(**(dict(segment_id=segment_id, asset=asset(),
                                 duration_seconds=6.125) | changes))


def visual(shot_id="shot_01", **changes):
    return VisualAsset(**(dict(shot_id=shot_id, source_segment_id="segment_01",
                              asset=asset("image_01", "IMAGE", relative_path="images/shot-01.png"),
                              generation_method="STATIC_IMAGE") | changes))


def package(**changes):
    return MediaPackage(**(dict(
        storyboard_input_ref=ArtifactReference(project_id="project", artifact_type="storyboard", version=7),
        title="Documentary", narration_assets=[narration()], visual_assets=[visual()],
    ) | changes))


@pytest.mark.parametrize("media_type", list(MediaType))
def test_asset_media_types(media_type):
    assert asset(media_type=media_type).media_type == media_type


@pytest.mark.parametrize("value", ["", " ", "bad id", "-bad", "a" * 81])
def test_invalid_asset_identity(value):
    with pytest.raises(ValidationError):
        asset(asset_id=value)


@pytest.mark.parametrize("path", [
    "", "/outside.png", "C:/Users/file.png", "D:\\projects\\file.png",
    "C:file.png", "//server/share/file.png", "\\server\\file.png",
    "../../outside.png", "images/../outside.png", "images/./shot.png",
    "images//shot.png", "images/", "images\\shot.png", "images/shot\x00.png",
    "images/NUL.png", "images/shot.png.", "images/shot.png ", "https://host/shot.png",
])
def test_unsafe_paths(path):
    with pytest.raises(ValidationError, match="safe POSIX relative path"):
        asset(relative_path=path)


@pytest.mark.parametrize("path", ["audio/seg-07.mp3", "images/nested/shot-021.png", "video/shot-022.mp4"])
def test_safe_paths_preserved_without_file_access(path):
    assert asset(relative_path=path).relative_path == path


@pytest.mark.parametrize("digest", ["", "a" * 63, "a" * 65, "g" * 64, "AB" * 32, " " + "a" * 64, "a" * 64 + "\n"])
def test_noncanonical_digest(digest):
    with pytest.raises(ValidationError):
        asset(sha256=digest)


def test_narration_actual_duration_and_metadata():
    value = narration(generation_metadata=dict(provider="provider", model="model"))
    assert value.asset.media_type == MediaType.AUDIO
    assert value.generation_metadata == GenerationMetadata(provider="provider", model="model")
    estimate = StoryboardShot(
        shot_id="shot", kind="STRUCTURAL", source_segment_id="segment_01",
        visual_description="River", generation_prompt="River", generation_method="STATIC_IMAGE",
        framing="WIDE", camera_motion="NONE", estimated_duration_seconds=4.5,
    )
    assert value.duration_seconds == 6.125 != estimate.estimated_duration_seconds
    assert NarrationAsset.model_validate_json(value.model_dump_json()).duration_seconds == 6.125


@pytest.mark.parametrize("media_type", ["IMAGE", "VIDEO"])
def test_narration_requires_audio(media_type):
    with pytest.raises(ValidationError, match="AUDIO"):
        narration(asset=asset(media_type=media_type))


@pytest.mark.parametrize("duration", [0, -1, float("nan"), float("inf"), -float("inf")])
def test_invalid_actual_duration(duration):
    with pytest.raises(ValidationError):
        narration(duration_seconds=duration)


@pytest.mark.parametrize("media_type,method", [("IMAGE", "STATIC_IMAGE"), ("VIDEO", "TEXT_TO_VIDEO"), ("VIDEO", "IMAGE_TO_VIDEO")])
def test_valid_visual(media_type, method):
    value = visual(asset=asset("visual", media_type), generation_method=method)
    assert value.generation_method == GenerationMethod(method)


def test_visual_rejects_audio():
    with pytest.raises(ValidationError, match="IMAGE or VIDEO"):
        visual(asset=asset())


@pytest.mark.parametrize("build,field", [(narration, "segment_id"), (visual, "shot_id"), (visual, "source_segment_id")])
def test_empty_source_identity(build, field):
    with pytest.raises(ValidationError):
        build(**{field: ""})


def test_exact_provenance_and_json_round_trip():
    value = package()
    data = json.loads(value.model_dump_json())
    assert MediaPackage.model_validate(data) == value
    assert MediaPackage.model_validate_json(value.model_dump_json()) == value
    assert data["storyboard_input_ref"] == dict(project_id="project", artifact_type="storyboard", version=7)
    assert data["schema_version"] == 1 != data["storyboard_input_ref"]["version"]
    assert data["narration_assets"][0]["duration_seconds"] == 6.125


@pytest.mark.parametrize("artifact_type", ["script", "story", "research", "Storyboard"])
def test_storyboard_type_required(artifact_type):
    with pytest.raises(ValidationError, match="storyboard artifact"):
        package(storyboard_input_ref=dict(project_id="project", artifact_type=artifact_type, version=7))


@pytest.mark.parametrize("changes,label", [
    (dict(narration_assets=[narration(), narration(asset=asset("other"))]), "Narration segment IDs"),
    (dict(visual_assets=[visual(), visual(asset=asset("other", "IMAGE"))]), "Visual shot IDs"),
    (dict(visual_assets=[visual(asset=asset("audio_01", "IMAGE"))]), "Media asset IDs"),
    (dict(narration_assets=[narration(), narration("other_segment")]), "Media asset IDs"),
    (dict(visual_assets=[visual(), visual("other_shot")]), "Media asset IDs"),
])
def test_duplicate_identities(changes, label):
    with pytest.raises(ValidationError, match=label):
        package(**changes)


def test_distinct_assets_order_and_detached_collections():
    data = package(narration_assets=[narration("z"), narration("a", asset=asset("audio_02"))],
                   visual_assets=[visual("z"), visual("a", asset=asset("video_02", "VIDEO"))]).model_dump(mode="json")
    value = MediaPackage.model_validate(data)
    data["visual_assets"].clear()
    assert [item.segment_id for item in value.narration_assets] == ["z", "a"]
    assert [item.shot_id for item in value.visual_assets] == ["z", "a"]


def test_local_validation_does_not_authenticate_upstream_authority():
    assert package(narration_assets=[], visual_assets=[]).visual_assets == ()
    value = package(narration_assets=[narration("unknown_segment")], visual_assets=[
        visual("unknown_shot", source_segment_id="another_unknown", generation_method="TEXT_TO_VIDEO")])
    assert value.visual_assets[0].generation_method == GenerationMethod.TEXT_TO_VIDEO


def test_exact_field_shapes_exclude_binary_timeline_and_duplicated_authority():
    assert set(MediaAssetReference.model_fields) == {"asset_id", "relative_path", "media_type", "sha256"}
    assert set(NarrationAsset.model_fields) == {"segment_id", "asset", "duration_seconds", "generation_metadata"}
    assert set(VisualAsset.model_fields) == {"shot_id", "source_segment_id", "asset", "generation_method", "generation_metadata"}
    assert set(MediaPackage.model_fields) == {"schema_version", "storyboard_input_ref", "title", "narration_assets", "visual_assets"}
    assert set(GenerationMetadata.model_fields) == {"provider", "model"}


@pytest.mark.parametrize("build,field", [
    (asset, "base64"), (asset, "binary"), (narration, "start_time"), (narration, "end_time"),
    (visual, "duration_seconds"), (visual, "grounding"), (visual, "generation_prompt"),
    (package, "workflow_state"),
])
def test_extras_forbidden(build, field):
    with pytest.raises(ValidationError, match="Extra inputs"):
        build(**{field: "unexpected"})


@pytest.mark.parametrize("value,field", [
    (asset(), "relative_path"), (narration(), "duration_seconds"), (visual(), "shot_id"),
    (package(), "title"), (GenerationMetadata(provider="provider", model="model"), "provider"),
])
def test_frozen_contracts(value, field):
    with pytest.raises(ValidationError, match="frozen"):
        setattr(value, field, getattr(value, field))


def test_required_package_fields_and_schema_version():
    data = package().model_dump()
    for field in ("storyboard_input_ref", "title", "narration_assets", "visual_assets"):
        with pytest.raises(ValidationError):
            MediaPackage.model_validate({key: value for key, value in data.items() if key != field})
    with pytest.raises(ValidationError):
        package(schema_version=2)
