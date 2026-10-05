import pytest
from pydantic import ValidationError

from history_studio.models import ScriptGrounding
from history_studio.script import (
    ScriptGroundingValidationReport, finalize_script_submission, validate_script_grounding,
)
from test_script_submission import extended_context, submission, section, segment


def package():
    return finalize_script_submission(extended_context(), submission())


def altered_segment(**changes):
    original = package()
    item = original.sections[0].segments[0].model_copy(update=changes)
    part = original.sections[0].model_copy(update={"segments": (item,)})
    return original.model_copy(update={"sections": (part,)})


def codes(report):
    return [issue.code.value for issue in report.issues]


def test_valid_deterministic_nonmutating_and_reloaded():
    source, output = extended_context(), package()
    before = source.model_dump_json(), output.model_dump_json()
    report = validate_script_grounding(source, output)
    assert report.is_valid and report.issues == ()
    assert validate_script_grounding(source, output) == report
    restored = type(output).model_validate_json(output.model_dump_json())
    assert validate_script_grounding(source, restored) == report
    assert before == (source.model_dump_json(), output.model_dump_json())


@pytest.mark.parametrize("changes", [{"version": 8}, {"project_id": "other"}, {"artifact_type": "research"}])
def test_exact_provenance_mismatch(changes):
    original = package()
    output = original.model_copy(update={"story_input_ref": original.story_input_ref.model_copy(update=changes)})
    assert "STORY_PROVENANCE_MISMATCH" in codes(validate_script_grounding(extended_context(), output))


@pytest.mark.parametrize("beat_id,fact_id,expected", [
    ("unknown", "f0", "UNKNOWN_STORY_BEAT"),
    ("beat_03", "f2", "BEAT_SECTION_MISMATCH"),
    ("transition_01", "f0", "NON_HISTORICAL_BEAT"),
    ("beat_01", "unknown", "UNKNOWN_FACT"),
    ("beat_01", "f2", "FACT_BEAT_MISMATCH"),
])
def test_beat_and_fact_authorization(beat_id, fact_id, expected):
    output = altered_segment(grounding=ScriptGrounding(story_beat_id=beat_id, research_fact_ids=[fact_id]))
    report = validate_script_grounding(extended_context(), output)
    assert expected in codes(report) and not report.is_valid
    issue = next(i for i in report.issues if i.code == expected)
    assert issue.section_id == "section_01" and issue.segment_id == "segment_01"
    assert issue.story_beat_id == beat_id


@pytest.mark.parametrize("changes,expected", [
    ({"grounding": None}, "MISSING_HISTORICAL_GROUNDING"),
    ({"kind": "STRUCTURAL"}, "STRUCTURAL_GROUNDING_PRESENT"),
    ({"grounding": ScriptGrounding.model_construct(story_beat_id="beat_01", research_fact_ids=())}, "EMPTY_FACT_SELECTION"),
    ({"grounding": ScriptGrounding.model_construct(story_beat_id="beat_01", research_fact_ids=("f0", "f0"))}, "DUPLICATE_FACT"),
])
def test_corrupted_shapes_are_reported(changes, expected):
    report = validate_script_grounding(extended_context(), altered_segment(**changes))
    assert "INVALID_PACKAGE_STRUCTURE" in codes(report)
    assert expected in codes(report)


def test_unknown_duplicate_section_and_global_segment_identity():
    original = package()
    unknown = original.sections[0].model_copy(update={"section_id": "unknown"})
    assert "UNKNOWN_SECTION" in codes(validate_script_grounding(
        extended_context(), original.model_copy(update={"sections": (unknown,)})))
    duplicated = original.model_copy(update={"sections": (original.sections[0], original.sections[0])})
    report = validate_script_grounding(extended_context(), duplicated)
    assert "DUPLICATE_SECTION" in codes(report) and "DUPLICATE_SEGMENT" in codes(report)


def test_section_order_subset_omission_and_repeated_beat_order():
    source = extended_context()
    later = segment("later", grounding=dict(story_beat_id="beat_02", research_fact_ids=["f2"]))
    structural = segment("transition", kind="STRUCTURAL", grounding=None)
    proposal = submission(sections=[section(segments=[segment(), segment("repeat"), structural, later]),
        section("section_02", [segment("last", grounding=dict(story_beat_id="beat_03", research_fact_ids=["f2"]))])])
    output = finalize_script_submission(source, proposal)
    assert validate_script_grounding(source, output).is_valid
    assert validate_script_grounding(source, output.model_copy(update={"sections": output.sections[1:]})).is_valid
    reversed_sections = output.model_copy(update={"sections": tuple(reversed(output.sections))})
    assert "SECTION_ORDER_VIOLATION" in codes(validate_script_grounding(source, reversed_sections))
    part = output.sections[0]
    reordered = part.model_copy(update={"segments": (part.segments[-1], part.segments[2], part.segments[0])})
    bad = output.model_copy(update={"sections": (reordered,)})
    assert "BEAT_ORDER_VIOLATION" in codes(validate_script_grounding(source, bad))


def test_multiple_issues_have_stable_presentation_order():
    output = altered_segment(grounding=ScriptGrounding(
        story_beat_id="beat_01", research_fact_ids=["missing", "f2"]))
    report = validate_script_grounding(extended_context(), output)
    assert codes(report) == ["UNKNOWN_FACT", "FACT_BEAT_MISMATCH"]
    assert [i.research_fact_id for i in report.issues] == ["missing", "f2"]
    assert validate_script_grounding(extended_context(), output) == report


@pytest.mark.parametrize("narration", ["In 9999, driven by ambition, he did it.",
    "可能 大约 据说 uncertain perhaps reportedly", "Unconditional claim with no qualification"])
def test_no_prose_date_qualification_or_psychology_judgment(narration):
    output = altered_segment(narration=narration, grounding=ScriptGrounding(
        story_beat_id="beat_01", research_fact_ids=["f1"]))
    before = output.model_dump_json()
    assert validate_script_grounding(extended_context(), output).is_valid
    assert output.model_dump_json() == before
    structural = altered_segment(kind="STRUCTURAL", grounding=None, narration=narration)
    assert validate_script_grounding(extended_context(), structural).is_valid


def test_no_external_dependencies(monkeypatch):
    from history_studio.storage.artifact_store import ArtifactStore
    from openai.resources.responses import Responses

    def forbidden(*args, **kwargs):
        pytest.fail("Validator must not access artifacts or models")

    for name in ("load", "load_latest", "list_versions", "save"):
        monkeypatch.setattr(ArtifactStore, name, forbidden)
    monkeypatch.setattr(Responses, "create", forbidden)
    assert validate_script_grounding(extended_context(), package()).is_valid


def test_report_validity_is_derived_and_cannot_be_authored():
    report = ScriptGroundingValidationReport()
    assert report.is_valid and report.model_dump()["is_valid"] is True
    with pytest.raises(ValidationError):
        ScriptGroundingValidationReport(is_valid=True)


def test_invalid_context_is_not_an_authority():
    source = extended_context()
    source.sections[0].beats[0].fact_refs[0].research_fact_id = "f1"
    with pytest.raises(ValidationError):
        validate_script_grounding(source, package())


def test_unsafe_nested_value_reports_shape_without_crashing():
    report = validate_script_grounding(extended_context(), altered_segment(grounding={}))
    assert codes(report) == ["INVALID_PACKAGE_STRUCTURE"]
