import pytest
from pydantic import ValidationError

from history_studio.models import StoryboardPackage
from history_studio.visual_director import (
    StoryboardIntegrityReport, finalize_storyboard_submission, validate_storyboard_integrity,
)
from test_visual_director_preparation import context
from test_storyboard_submission import submission


def package():
    return finalize_storyboard_submission(context(), submission())


def altered_shot(index=0, **changes):
    original = package()
    section = original.sections[0]
    shots = list(section.shots)
    shots[index] = shots[index].model_copy(update=changes)
    return original.model_copy(update={"sections": (
        section.model_copy(update={"shots": tuple(shots)}), original.sections[1])})


def codes(output):
    return [issue.code.value for issue in validate_storyboard_integrity(context(), output).issues]


def test_valid_deterministic_detached_round_trip_and_nonmutating():
    source, output = context(), package()
    before = source.model_dump_json(), output.model_dump_json()
    report = validate_storyboard_integrity(source, output)
    assert report.is_valid and report.issues == ()
    assert validate_storyboard_integrity(source, output) == report
    assert validate_storyboard_integrity(source, StoryboardPackage.model_validate_json(output.model_dump_json())) == report
    assert before == (source.model_dump_json(), output.model_dump_json())
    assert report.model_dump(mode="json") == {"issues": [], "is_valid": True}


@pytest.mark.parametrize("changes", [{"version": 8}, {"project_id": "other"}, {"artifact_type": "story"}])
def test_exact_provenance_mismatch(changes):
    original = package()
    output = original.model_copy(update={"script_input_ref": original.script_input_ref.model_copy(update=changes)})
    assert "SCRIPT_PROVENANCE_MISMATCH" in codes(output)


def test_canonical_package_and_section_titles():
    original = package()
    assert codes(original.model_copy(update={"title": "Different"})) == ["TITLE_MISMATCH"]
    section = original.sections[0].model_copy(update={"title": "Different"})
    assert codes(original.model_copy(update={"sections": (section, original.sections[1])})) == ["SECTION_IDENTITY_MISMATCH"]


def test_missing_unknown_reordered_and_duplicate_sections():
    original = package()
    assert codes(original.model_copy(update={"sections": original.sections[:1]})) == ["SECTION_MISSING", "SEGMENT_UNCOVERED"]
    unknown = original.sections[0].model_copy(update={"section_id": "unknown"})
    report = codes(original.model_copy(update={"sections": (unknown, original.sections[1])}))
    assert "SECTION_UNKNOWN" in report and "SECTION_MISSING" in report
    assert codes(original.model_copy(update={"sections": tuple(reversed(original.sections))})) == ["SECTION_ORDER_MISMATCH"]
    duplicated = original.model_copy(update={"sections": (*original.sections, original.sections[0])})
    report = codes(duplicated)
    assert "INVALID_PACKAGE_STRUCTURE" in report and "SECTION_ID_DUPLICATE" in report
    assert "SHOT_ID_DUPLICATE" in report


def test_uncovered_segment():
    original = package()
    section = original.sections[0].model_copy(update={"shots": original.sections[0].shots[:1]})
    report = validate_storyboard_integrity(context(), original.model_copy(update={"sections": (section, original.sections[1])}))
    assert [i.code for i in report.issues] == ["SEGMENT_UNCOVERED"]
    assert report.issues[0].section_id == "z" and report.issues[0].source_segment_id == "a"


@pytest.mark.parametrize("segment,expected", [("unknown", "SOURCE_SEGMENT_UNKNOWN"), ("closing", "SOURCE_SEGMENT_WRONG_SECTION")])
def test_source_authorization_without_cascading_kind_or_order(segment, expected):
    report = validate_storyboard_integrity(context(), altered_shot(source_segment_id=segment, kind="STRUCTURAL"))
    assert [i.code for i in report.issues] == [expected, "SEGMENT_UNCOVERED"]
    issue = report.issues[0]
    assert (issue.section_id, issue.shot_id, issue.source_segment_id) == ("z", "shot1", segment)


@pytest.mark.parametrize("index,kind", [(0, "STRUCTURAL"), (1, "HISTORICAL")])
def test_kind_compatibility(index, kind):
    assert codes(altered_shot(index=index, kind=kind)) == ["SHOT_KIND_MISMATCH"]


def test_multiple_shots_preserve_agent_order_and_regression_reported():
    original = package()
    section = original.sections[0]
    repeated = section.shots[0].model_copy(update={"shot_id": "extra"})
    valid = section.model_copy(update={"shots": (section.shots[0], repeated, section.shots[1])})
    assert codes(original.model_copy(update={"sections": (valid, original.sections[1])})) == []
    backward = section.model_copy(update={"shots": (*section.shots, repeated)})
    assert codes(original.model_copy(update={"sections": (backward, original.sections[1])})) == ["SEGMENT_ORDER_REGRESSION"]


def test_global_duplicate_shot_identity():
    assert codes(altered_shot(shot_id="shot3")) == ["INVALID_PACKAGE_STRUCTURE", "SHOT_ID_DUPLICATE"]


@pytest.mark.parametrize("changes", [
    {"estimated_duration_seconds": 0}, {"estimated_duration_seconds": float("inf")},
    {"generation_method": "UNKNOWN"}, {"generation_prompt": ""}, {"visual_description": ""},
])
def test_malformed_local_fields_report_structure(changes):
    assert codes(altered_shot(**changes)) == ["INVALID_PACKAGE_STRUCTURE"]


@pytest.mark.parametrize("unsafe", [{}, None, []])
def test_unsafe_nested_values_fail_safely(unsafe):
    original = package()
    section = original.sections[0].model_copy(update={"shots": (unsafe,)})
    assert codes(original.model_copy(update={"sections": (section,)})) == ["INVALID_PACKAGE_STRUCTURE"]
    assert codes(altered_shot(source_segment_id=unsafe)) == ["INVALID_PACKAGE_STRUCTURE"]


def test_invalid_authority_raises():
    source = context()
    source.sections[1].segments[0].segment_id = "z"
    with pytest.raises(ValidationError):
        validate_storyboard_integrity(source, package())
    source = context()
    invalid = source.script_input_ref.model_copy(update={"version": True})
    with pytest.raises(ValidationError):
        validate_storyboard_integrity(source.model_copy(update={"script_input_ref": invalid}), package())


def test_multiple_independent_issues_have_stable_order():
    original = altered_shot(kind="STRUCTURAL")
    output = original.model_copy(update={"title": "Different", "script_input_ref":
        original.script_input_ref.model_copy(update={"version": 8})})
    before = output.model_dump_json()
    report = validate_storyboard_integrity(context(), output)
    assert [i.code for i in report.issues] == ["SCRIPT_PROVENANCE_MISMATCH", "TITLE_MISMATCH", "SHOT_KIND_MISMATCH"]
    assert validate_storyboard_integrity(context(), output).model_dump_json() == report.model_dump_json()
    restored = StoryboardIntegrityReport.model_validate(report.model_dump(exclude={"is_valid"}))
    assert restored == report and not restored.is_valid
    assert output.model_dump_json() == before


@pytest.mark.parametrize("prose", ["In 9999, driven by ambition, a king met his secret brother.",
                                    "Invented costume, emotion, exact geography and dialogue"])
def test_no_historical_semantic_judgment(prose):
    for index in (0, 1):
        output = altered_shot(index=index, visual_description=prose, generation_prompt=prose)
        before = output.model_dump_json()
        assert validate_storyboard_integrity(context(), output).is_valid
        assert output.model_dump_json() == before


def test_no_external_access(monkeypatch):
    from history_studio.storage.artifact_store import ArtifactStore
    from openai.resources.responses import Responses

    def forbidden(*args, **kwargs):
        pytest.fail("Integrity validation must not access artifacts or models")

    for name in ("load", "load_latest", "list_versions", "save"):
        monkeypatch.setattr(ArtifactStore, name, forbidden)
    monkeypatch.setattr(Responses, "create", forbidden)
    assert validate_storyboard_integrity(context(), package()).is_valid


def test_validity_is_derived():
    report = StoryboardIntegrityReport()
    assert report.is_valid
    with pytest.raises(ValidationError):
        StoryboardIntegrityReport(is_valid=True)
    with pytest.raises(AttributeError):
        report.is_valid = False


@pytest.mark.parametrize("source,output", [(None, package()), (context(), {})])
def test_requires_typed_inputs(source, output):
    with pytest.raises(TypeError, match="VisualDirectorContext and StoryboardPackage"):
        validate_storyboard_integrity(source, output)
