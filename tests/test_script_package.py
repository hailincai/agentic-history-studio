import pytest
from pydantic import ValidationError

from history_studio.models import (
    ArtifactReference, ScriptGrounding, ScriptPackage, ScriptSection, ScriptSegment,
    ScriptSegmentKind,
)


def grounding(**changes):
    return ScriptGrounding(**(dict(story_beat_id="beat_01", research_fact_ids=["f1"]) | changes))


def segment(segment_id="segment_01", **changes):
    return ScriptSegment(**(dict(segment_id=segment_id, kind="HISTORICAL",
                                narration="Audience-facing narration", grounding=grounding()) | changes))


def section(section_id="section_01", **changes):
    return ScriptSection(**(dict(section_id=section_id, title="Opening", segments=[segment()]) | changes))


def package(**changes):
    return ScriptPackage(**(dict(story_input_ref=ArtifactReference(
        project_id="project", artifact_type="story", version=7),
        title="Documentary", sections=[section()]) | changes))


def test_minimal_historical_segment():
    value = segment()
    assert value.kind == ScriptSegmentKind.HISTORICAL
    assert value.grounding.story_beat_id == "beat_01"
    assert value.grounding.research_fact_ids == ("f1",)


def test_minimal_structural_segment():
    value = ScriptSegment(segment_id="transition", kind="STRUCTURAL", narration="Next chapter")
    assert value.kind == ScriptSegmentKind.STRUCTURAL
    assert value.grounding is None


def test_historical_requires_grounding():
    for changes in ({"grounding": None},):
        with pytest.raises(ValidationError, match="Historical segment requires grounding"):
            segment(**changes)
    with pytest.raises(ValidationError, match="Historical segment requires grounding"):
        ScriptSegment(segment_id="s1", kind="HISTORICAL", narration="Narration")


def test_grounding_requires_fact_ids():
    with pytest.raises(ValidationError):
        grounding(research_fact_ids=[])
    with pytest.raises(ValidationError):
        ScriptGrounding(story_beat_id="b1")


def test_duplicate_grounding_fact_ids():
    with pytest.raises(ValidationError, match="Grounding fact IDs must be unique"):
        grounding(research_fact_ids=["f1", "f1"])


def test_structural_forbids_grounding():
    with pytest.raises(ValidationError, match="Structural segment cannot carry grounding"):
        segment(kind="STRUCTURAL")


@pytest.mark.parametrize("narration", ["", " \n\t", None, 123])
def test_invalid_narration(narration):
    with pytest.raises(ValidationError):
        segment(narration=narration)


def test_text_normalization_and_invalid_kind():
    assert segment(narration="  Narration  ").narration == "Narration"
    with pytest.raises(ValidationError):
        segment(kind="UNKNOWN")


@pytest.mark.parametrize("identifier", ["", " ", "bad id", "-bad", "a" * 81])
def test_identifier_rules(identifier):
    for build in (lambda: segment(segment_id=identifier),
                  lambda: section(section_id=identifier),
                  lambda: grounding(story_beat_id=identifier),
                  lambda: grounding(research_fact_ids=[identifier])):
        with pytest.raises(ValidationError):
            build()


def test_valid_section():
    assert section().segments == (segment(),)


def test_section_requires_segments():
    with pytest.raises(ValidationError):
        section(segments=[])
    with pytest.raises(ValidationError):
        ScriptSection(section_id="s1", title="Opening")


def test_duplicate_segments_within_section():
    with pytest.raises(ValidationError, match="Segment IDs must be unique"):
        section(segments=[segment(), segment()])


def test_package_preserves_exact_story_provenance_and_round_trips():
    value = package()
    restored = ScriptPackage.model_validate_json(value.model_dump_json())
    assert restored == value
    assert restored.story_input_ref == value.story_input_ref
    assert restored.story_input_ref.model_dump() == dict(
        project_id="project", artifact_type="story", version=7)
    assert restored.schema_version == 1
    assert restored.schema_version != restored.story_input_ref.version


@pytest.mark.parametrize("artifact_type", ["research", "verification", "script", "Story"])
def test_wrong_input_artifact_type(artifact_type):
    with pytest.raises(ValidationError, match="story artifact"):
        package(story_input_ref=ArtifactReference(
            project_id="project", artifact_type=artifact_type, version=7))


def test_package_requires_explicit_reference_and_sections():
    data = package().model_dump()
    for field in ("story_input_ref", "sections"):
        with pytest.raises(ValidationError):
            ScriptPackage.model_validate({key: value for key, value in data.items() if key != field})
    with pytest.raises(ValidationError):
        package(sections=[])
    with pytest.raises(ValidationError):
        package(story_input_ref={"schema_version": 1})
    with pytest.raises(ValidationError):
        package(schema_version=2)


def test_duplicate_section_ids():
    with pytest.raises(ValidationError, match="Section IDs must be unique"):
        package(sections=[section(), section(segments=[segment("segment_02")])])


def test_duplicate_segment_ids_across_sections():
    with pytest.raises(ValidationError, match="Segment IDs must be unique"):
        package(sections=[section(), section("section_02")])


def test_same_beat_reused_with_different_fact_subsets():
    first = segment(grounding=grounding(research_fact_ids=["f1", "f2"]))
    second = segment("segment_02", grounding=grounding(research_fact_ids=["f2"]))
    value = package(sections=[section(segments=[first, second])])
    assert [item.grounding.story_beat_id for item in value.sections[0].segments] == ["beat_01", "beat_01"]
    assert [item.grounding.research_fact_ids for item in value.sections[0].segments] == [("f1", "f2"), ("f2",)]


def test_grounding_contains_only_identifiers_without_membership_authentication():
    value = grounding(story_beat_id="unknown_beat", research_fact_ids=["unknown_fact"])
    assert value.model_dump() == dict(story_beat_id="unknown_beat", research_fact_ids=("unknown_fact",))
    # Structural shape cannot prove narration truth or detect hidden historical claims.
    assert segment(grounding=value).grounding == value
    assert segment(kind="STRUCTURAL", grounding=None, narration="An unchecked assertion").grounding is None


@pytest.mark.parametrize("field,value", [
    ("status", "VERIFIED"), ("qualification", "Qualified"),
    ("historical_time", {}), ("fact_snapshots", []), ("evidence", []),
])
def test_grounding_forbids_duplicate_authority(field, value):
    with pytest.raises(ValidationError, match="Extra inputs"):
        grounding(**{field: value})


@pytest.mark.parametrize("field", ["research_package", "verification_package", "target_duration_seconds"])
def test_package_forbids_upstream_payloads_and_duration(field):
    with pytest.raises(ValidationError, match="Extra inputs"):
        package(**{field: {}})


@pytest.mark.parametrize("build", [lambda: section(title=" "), lambda: package(title=" ")])
def test_titles_follow_text_rules(build):
    with pytest.raises(ValidationError):
        build()
