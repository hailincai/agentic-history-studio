import pytest
from pydantic import ValidationError

from history_studio.models import (
    ArtifactReference, CameraMotion, GenerationMethod, Shot, ShotFraming, Storyboard,
    StoryboardPackage, StoryboardSection, StoryboardShot, StoryboardShotKind,
)


def shot(shot_id="shot_01", **changes):
    return StoryboardShot(**(dict(
        shot_id=shot_id, kind="HISTORICAL", source_segment_id="segment_01",
        visual_description="A traveler beside a river", generation_prompt="Cinematic river valley",
        generation_method="STATIC_IMAGE", framing="WIDE", camera_motion="NONE",
        estimated_duration_seconds=4.5,
    ) | changes))


def section(section_id="section_01", **changes):
    return StoryboardSection(**(dict(section_id=section_id, title="Opening", shots=[shot()]) | changes))


def package(**changes):
    return StoryboardPackage(**(dict(
        script_input_ref=ArtifactReference(project_id="project", artifact_type="script", version=7),
        title="Documentary", sections=[section()],
    ) | changes))


def test_minimal_package_exact_provenance_and_round_trip():
    value = package()
    restored = StoryboardPackage.model_validate_json(value.model_dump_json())
    assert restored == value
    assert restored.script_input_ref.model_dump() == dict(
        project_id="project", artifact_type="script", version=7)
    assert restored.schema_version == 1 != restored.script_input_ref.version
    assert set(StoryboardShot.model_fields) == {
        "shot_id", "kind", "source_segment_id", "visual_description", "generation_prompt",
        "generation_method", "framing", "camera_motion", "estimated_duration_seconds",
    }


def test_detached_validation_and_order_with_repeated_segment():
    data = package(sections=[section("z", shots=[shot("z"), shot("a")]),
                             section("a", shots=[shot("b", kind="STRUCTURAL")])]).model_dump(mode="json")
    value = StoryboardPackage.model_validate(data)
    data["sections"][0]["shots"][0]["generation_prompt"] = "Changed"
    data["sections"].clear()
    assert [item.section_id for item in value.sections] == ["z", "a"]
    assert [item.shot_id for item in value.sections[0].shots] == ["z", "a"]
    assert value.sections[0].shots[0].generation_prompt == "Cinematic river valley"
    assert {item.source_segment_id for sec in value.sections for item in sec.shots} == {"segment_01"}


@pytest.mark.parametrize("kind", list(StoryboardShotKind))
def test_declared_kind_without_cross_artifact_or_semantic_authentication(kind):
    assert shot(kind=kind, source_segment_id="unknown_segment",
                visual_description="An unchecked historical assertion",
                generation_prompt="Unsupported creative detail").kind == kind


@pytest.mark.parametrize("field,enum", [
    ("generation_method", GenerationMethod), ("framing", ShotFraming),
    ("camera_motion", CameraMotion), ("kind", StoryboardShotKind),
])
def test_enum_members_and_invalid_value(field, enum):
    for member in enum:
        assert getattr(shot(**{field: member.value}), field) == member
    with pytest.raises(ValidationError):
        shot(**{field: "UNKNOWN"})


@pytest.mark.parametrize("field", list(StoryboardShot.model_fields))
def test_every_shot_field_required(field):
    data = shot().model_dump()
    del data[field]
    with pytest.raises(ValidationError):
        StoryboardShot.model_validate(data)


@pytest.mark.parametrize("field", ["visual_description", "generation_prompt"])
@pytest.mark.parametrize("value", ["", " \n\t", None, 123])
def test_creative_text_constraints(field, value):
    with pytest.raises(ValidationError):
        shot(**{field: value})


def test_text_normalization():
    assert shot(generation_prompt="  Prompt  ").generation_prompt == "Prompt"


@pytest.mark.parametrize("duration", [0, -1, float("inf"), float("nan")])
def test_invalid_planning_duration(duration):
    with pytest.raises(ValidationError):
        shot(estimated_duration_seconds=duration)


def test_positive_fractional_planning_duration():
    assert shot(estimated_duration_seconds=0.1).estimated_duration_seconds == 0.1


@pytest.mark.parametrize("artifact_type", ["story", "research", "verification", "Script"])
def test_wrong_provenance_type(artifact_type):
    with pytest.raises(ValidationError, match="script artifact"):
        package(script_input_ref=dict(project_id="project", artifact_type=artifact_type, version=7))


def test_required_package_and_section_shape():
    for model, data, fields in (
        (StoryboardPackage, package().model_dump(), ["script_input_ref", "title", "sections"]),
        (StoryboardSection, section().model_dump(), ["section_id", "title", "shots"]),
    ):
        for field in fields:
            with pytest.raises(ValidationError):
                model.model_validate({key: value for key, value in data.items() if key != field})
    for build in (lambda: package(sections=[]), lambda: section(shots=[]),
                  lambda: package(title=" "), lambda: section(title=" "),
                  lambda: package(schema_version=2),
                  lambda: package(script_input_ref={"schema_version": 1})):
        with pytest.raises(ValidationError):
            build()


def test_duplicate_identifiers():
    for sections, message in (
        ([section(), section(shots=[shot("other")])], "Section IDs"),
        ([section(), section("other")], "Shot IDs"),
    ):
        with pytest.raises(ValidationError, match=message):
            package(sections=sections)
    with pytest.raises(ValidationError, match="Shot IDs"):
        section(shots=[shot(), shot()])


@pytest.mark.parametrize("identifier", ["", "bad id", "-bad", "a" * 81])
def test_identifier_constraints(identifier):
    for build in (lambda: shot(shot_id=identifier), lambda: shot(source_segment_id=identifier),
                  lambda: section(section_id=identifier)):
        with pytest.raises(ValidationError):
            build()


@pytest.mark.parametrize("field", [
    "story_beat_id", "research_fact_ids", "status", "historical_time", "evidence", "qualification",
])
def test_shot_forbids_duplicated_authority(field):
    with pytest.raises(ValidationError, match="Extra inputs"):
        shot(**{field: []})


@pytest.mark.parametrize("field", ["script_package", "target_duration_seconds", "workflow_state"])
def test_package_forbids_upstream_payload_and_runtime_metadata(field):
    with pytest.raises(ValidationError, match="Extra inputs"):
        package(**{field: {}})


def test_legacy_storyboard_remains_usable():
    legacy = Shot(shot_id="legacy", scene_id="scene", sequence=1, start_seconds=0,
                  duration_seconds=5, visual_description="River", location="China", period="Tang",
                  generation_method="TEXT_TO_VIDEO", camera_motion="slow arbitrary motion", prompt="River")
    value = Storyboard(shots=[legacy], estimated_media_cost_usd=0)
    assert Storyboard.model_validate_json(value.model_dump_json()) == value
    assert value.shots[0].camera_motion == "slow arbitrary motion"
