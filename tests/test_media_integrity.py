import pytest

from history_studio.media import validate_media_package
from history_studio.models import ArtifactReference, MediaAssetReference, MediaPackage, NarrationAsset, VisualAsset
from test_narration_media import script_package, storyboard_package


def reference(kind, version=1):
    return ArtifactReference(project_id="project", artifact_type=kind, version=version)


def binary(asset_id, media_type):
    return MediaAssetReference(asset_id=asset_id, relative_path=f"assets/{asset_id}.bin", media_type=media_type, sha256="a" * 64)


def inputs():
    script = script_package()
    storyboard = storyboard_package()
    data = storyboard.model_dump(mode="json")
    for shot, method in zip(data["sections"][0]["shots"], ["STATIC_IMAGE", "TEXT_TO_VIDEO", "IMAGE_TO_VIDEO"]):
        shot["generation_method"] = method
    storyboard = type(storyboard).model_validate(data)
    package = MediaPackage(storyboard_input_ref=reference("storyboard"), title=storyboard.title,
        narration_assets=[NarrationAsset(segment_id=segment_id, asset=binary(f"n-{segment_id}", "AUDIO"), duration_seconds=0.25)
                          for segment_id in ("SEG-Z", "SEG-A")],
        visual_assets=[VisualAsset(shot_id=shot.shot_id, source_segment_id=shot.source_segment_id,
            generation_method=shot.generation_method,
            asset=binary(f"v-{shot.shot_id}", "IMAGE" if shot.generation_method == "STATIC_IMAGE" else "VIDEO"))
            for shot in storyboard.sections[0].shots])
    return dict(storyboard_input_ref=reference("storyboard"), storyboard=storyboard,
                script_input_ref=reference("script"), script=script, package=package)


def codes(values):
    return {issue.code.value for issue in validate_media_package(**values).issues}


def test_complete_package_all_methods_structural_shots_and_no_filesystem():
    values = inputs()
    report = validate_media_package(**values)
    assert report.is_valid and report.model_dump(mode="json")["is_valid"] is True
    assert len(values["package"].visual_assets) == 3
    assert {shot.kind.value for shot in values["storyboard"].sections[0].shots} == {"STRUCTURAL"}


@pytest.mark.parametrize("field,kind,expected", [
    ("storyboard_input_ref", "storyboard", "STORYBOARD_PROVENANCE_MISMATCH"),
    ("script_input_ref", "script", "SCRIPT_PROVENANCE_MISMATCH"),
])
def test_exact_refs_cannot_be_substituted_even_with_identical_payload(field, kind, expected):
    values = inputs()
    values[field] = reference(kind, 2)
    assert expected in codes(values)


@pytest.mark.parametrize("change,expected", [
    (lambda data: data.update(title="Other"), "TITLE_MISMATCH"),
    (lambda data: data["narration_assets"].pop(), "NARRATION_MISSING"),
    (lambda data: data["narration_assets"].append(dict(segment_id="unused", asset=binary("extra", "AUDIO").model_dump(), duration_seconds=1)), "NARRATION_EXTRA"),
    (lambda data: data["narration_assets"][0].update(segment_id="unknown"), "NARRATION_UNKNOWN"),
    (lambda data: data["narration_assets"].reverse(), "NARRATION_ORDER_MISMATCH"),
    (lambda data: data["visual_assets"].pop(), "VISUAL_MISSING"),
    (lambda data: data["visual_assets"][0].update(shot_id="unknown"), "VISUAL_UNKNOWN"),
    (lambda data: data["visual_assets"][0].update(source_segment_id="SEG-Z"), "SOURCE_SEGMENT_MISMATCH"),
    (lambda data: data["visual_assets"].reverse(), "VISUAL_ORDER_MISMATCH"),
    (lambda data: data["visual_assets"][0].update(generation_method="TEXT_TO_VIDEO"), "GENERATION_METHOD_MISMATCH"),
])
def test_coverage_identity_order_and_method_failures(change, expected):
    values = inputs()
    data = values["package"].model_dump(mode="json")
    change(data)
    values["package"] = MediaPackage.model_validate(data)
    assert expected in codes(values)


@pytest.mark.parametrize("index,wrong_type", [(0, "VIDEO"), (1, "IMAGE"), (2, "IMAGE")])
def test_final_media_type_authenticates_approved_method(index, wrong_type):
    values = inputs()
    data = values["package"].model_dump(mode="json")
    data["visual_assets"][index]["asset"]["media_type"] = wrong_type
    values["package"] = MediaPackage.model_validate(data)
    assert "FINAL_MEDIA_TYPE_MISMATCH" in codes(values)


@pytest.mark.parametrize("field", ["narration_assets", "visual_assets"])
def test_duplicate_identities_in_unsafe_copies_report_contract_failure(field):
    values = inputs()
    assets = getattr(values["package"], field)
    values["package"] = values["package"].model_copy(update={field: (*assets, assets[0])})
    assert "INVALID_PACKAGE_STRUCTURE" in codes(values)


def test_wrong_audio_type_and_duplicate_asset_ids_report_contract_failure():
    for change in ("type", "asset_id"):
        values = inputs()
        narration = values["package"].narration_assets[0]
        modified = narration.asset.model_copy(update={"media_type": "IMAGE"} if change == "type" else {
            "asset_id": values["package"].visual_assets[0].asset.asset_id})
        values["package"] = values["package"].model_copy(update={"narration_assets": (
            narration.model_copy(update={"asset": modified}), values["package"].narration_assets[1])})
        assert "INVALID_PACKAGE_STRUCTURE" in codes(values)


def test_storyboard_unknown_source_and_foreign_script_lineage():
    values = inputs()
    data = values["storyboard"].model_dump(mode="json")
    data["sections"][0]["shots"][0]["source_segment_id"] = "unknown"
    values["storyboard"] = type(values["storyboard"]).model_validate(data)
    assert "SOURCE_SEGMENT_UNKNOWN" in codes(values)
    values = inputs()
    data = values["script"].model_dump(mode="python")
    data["story_input_ref"]["project_id"] = "foreign"
    values["script"] = type(values["script"]).model_validate(data)
    assert "SCRIPT_PROVENANCE_MISMATCH" in codes(values)
