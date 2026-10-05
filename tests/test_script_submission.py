import pytest
from pydantic import ValidationError

from history_studio.models import ScriptContext, ScriptPackage
from history_studio.script import ScriptSubmission, finalize_script_submission
from test_script_writer import context


def segment(segment_id="segment_01", **changes):
    return dict(segment_id=segment_id, kind="HISTORICAL", narration="Approved narration",
                grounding=dict(story_beat_id="beat_01", research_fact_ids=["f0"])) | changes


def section(section_id="section_01", segments=None):
    return dict(section_id=section_id, title="Audience-facing section", segments=segments or [segment()])


def submission(**changes):
    return ScriptSubmission.model_validate(dict(title="Audience-facing title", sections=[section()]) | changes)


def extended_context():
    data = context().model_dump(mode="json")
    beat = data["sections"][0]["beats"][0]
    later = dict(beat, beat_id="beat_02", fact_refs=[beat["fact_refs"][2]],
                 fact_chronology=[beat["fact_chronology"][2]])
    # f2 is authorized only by the later beat.
    beat["fact_refs"] = beat["fact_refs"][:2]
    beat["fact_chronology"] = beat["fact_chronology"][:2]
    data["sections"][0]["beats"].append(later)
    data["sections"].append(dict(section_id="section_02", purpose="Later section", beats=[
        dict(later, beat_id="beat_03")]))
    return ScriptContext.model_validate(data)


def test_historical_finalization_exact_provenance_and_detachment():
    source, proposal = context(), submission()
    before = source.model_dump_json(), proposal.model_dump_json()
    result = finalize_script_submission(source, proposal)
    assert isinstance(result, ScriptPackage) and not isinstance(proposal, ScriptPackage)
    assert result.story_input_ref == source.story_input_ref
    assert result.story_input_ref is not source.story_input_ref
    assert result.story_input_ref.version == 7 and result.schema_version == 1
    accepted = result.sections[0].segments[0]
    assert accepted.grounding.story_beat_id == "beat_01"
    assert accepted.grounding.research_fact_ids == ("f0",)
    assert result.title == proposal.title and result.sections[0].title == proposal.sections[0].title
    assert before == (source.model_dump_json(), proposal.model_dump_json())
    assert finalize_script_submission(source, proposal) == result
    assert ScriptPackage.model_validate_json(result.model_dump_json()) == result
    accepted.narration = "Output-only wording"
    assert proposal.sections[0].segments[0].narration == "Approved narration"


def test_structural_narration_has_no_invented_grounding():
    proposal = submission(sections=[section(segments=[segment(kind="STRUCTURAL", grounding=None,
        narration="An unchecked historical assertion")])])
    accepted = finalize_script_submission(context(), proposal).sections[0].segments[0]
    assert accepted.grounding is None
    assert accepted.narration == "An unchecked historical assertion"


@pytest.mark.parametrize("beat_id,message", [
    ("unknown", "must exist"), ("transition_01", "historical Story beat"),
    ("beat_03", "submitted Story section"),
])
def test_beat_identity_kind_and_section_authentication(beat_id, message):
    proposal = submission(sections=[section(segments=[segment(grounding=dict(
        story_beat_id=beat_id, research_fact_ids=["f0"]))])])
    with pytest.raises(ValueError, match=message):
        finalize_script_submission(extended_context(), proposal)


@pytest.mark.parametrize("fact_id", ["unknown", "f2"])
def test_fact_must_be_authorized_by_target_beat(fact_id):
    proposal = submission(sections=[section(segments=[segment(grounding=dict(
        story_beat_id="beat_01", research_fact_ids=[fact_id]))])])
    with pytest.raises(ValueError, match="exact Story beat"):
        finalize_script_submission(extended_context(), proposal)


def test_subset_selection_preserved_without_expansion_or_rewriting():
    source = context()
    proposal = submission(sections=[section(segments=[segment(grounding=dict(
        story_beat_id="beat_01", research_fact_ids=["f1"]), narration="  原文，没有资格关键词。  ")])])
    result = finalize_script_submission(source, proposal)
    accepted = result.sections[0].segments[0]
    assert accepted.grounding.research_fact_ids == ("f1",)
    assert accepted.narration == proposal.sections[0].segments[0].narration == "原文，没有资格关键词。"
    assert source.sections[0].beats[0].fact_refs[1].qualification == "日期尚不确定"


@pytest.mark.parametrize("changes", [
    {"grounding": None}, {"kind": "STRUCTURAL"},
    {"grounding": {"story_beat_id": "beat_01", "research_fact_ids": []}},
    {"grounding": {"story_beat_id": "beat_01", "research_fact_ids": ["f0", "f0"]}},
    {"narration": " "}, {"segment_id": "../bad"},
    {"kind": "STRUCTURAL", "grounding": None, "story_beat_id": "transition_01"},
    {"kind": "STRUCTURAL", "grounding": None, "research_fact_ids": ["f0"]},
])
def test_invalid_proposal_structure(changes):
    with pytest.raises(ValidationError):
        submission(sections=[section(segments=[segment(**changes)])])


def test_duplicate_identity_rejected():
    for sections in ([section(), section()],
                     [section(), section("section_02")],
                     [section(segments=[segment(), segment()])]):
        with pytest.raises(ValidationError, match="must be unique"):
            submission(sections=sections)


def test_unknown_section_and_section_order_rejected():
    with pytest.raises(ValueError, match="section must exist"):
        finalize_script_submission(context(), submission(sections=[section("unknown")]))
    proposal = submission(sections=[section("section_02", [segment("s2", grounding=dict(
        story_beat_id="beat_03", research_fact_ids=["f2"]))]), section()])
    with pytest.raises(ValueError, match="section order"):
        finalize_script_submission(extended_context(), proposal)


def test_order_omission_and_repeat_segments_for_one_beat():
    source = extended_context()
    later = segment("s3", grounding=dict(story_beat_id="beat_02", research_fact_ids=["f2"]))
    structural = segment("s2", kind="STRUCTURAL", grounding=None)
    proposal = submission(sections=[section(segments=[segment(), structural, segment("s4"), later])])
    result = finalize_script_submission(source, proposal)
    assert [s.segment_id for s in result.sections[0].segments] == ["segment_01", "s2", "s4", "s3"]
    assert [s.section_id for s in result.sections] == ["section_01"]
    with pytest.raises(ValueError, match="beat order"):
        finalize_script_submission(source, submission(sections=[section(segments=[later, structural, segment()])]))
    # Neither complete section/beat coverage nor complete fact coverage is required.
    omitted = submission(sections=[section("section_02", [segment(grounding=dict(
        story_beat_id="beat_03", research_fact_ids=["f2"]))])])
    assert finalize_script_submission(source, omitted).sections[0].section_id == "section_02"


@pytest.mark.parametrize("level,field,value", [
    ("submission", "story_input_ref", {}), ("submission", "schema_version", 1),
    ("submission", "approval_identity", "approved"), ("submission", "version", 8),
    ("segment", "status", "VERIFIED"), ("segment", "historical_time", {}),
    ("grounding", "status", "VERIFIED"), ("grounding", "use", "AFFIRMATIVE"),
    ("grounding", "qualification", "perhaps"), ("grounding", "historical_time", {}),
    ("grounding", "source_id", "source"), ("grounding", "evidence", []),
])
def test_submission_cannot_author_authoritative_metadata(level, field, value):
    data = submission().model_dump(mode="json")
    target = data if level == "submission" else data["sections"][0]["segments"][0]
    if level == "grounding":
        target = target["grounding"]
    target[field] = value
    with pytest.raises(ValidationError, match="Extra inputs"):
        ScriptSubmission.model_validate(data)


def test_output_only_contains_p5a_fields_and_no_upstream_metadata():
    result = finalize_script_submission(context(), submission()).model_dump(mode="json")
    assert set(result) == {"schema_version", "story_input_ref", "title", "sections"}
    accepted = result["sections"][0]["segments"][0]
    assert set(accepted) == {"segment_id", "kind", "narration", "grounding"}
    assert set(accepted["grounding"]) == {"story_beat_id", "research_fact_ids"}


def test_no_artifact_provider_or_latest_access(monkeypatch):
    from history_studio.storage.artifact_store import ArtifactStore
    from openai.resources.responses import Responses

    def forbidden(*args, **kwargs):
        pytest.fail("Finalization must not access artifacts or models")

    for method in ("load", "load_latest", "list_versions", "save"):
        monkeypatch.setattr(ArtifactStore, method, forbidden)
    monkeypatch.setattr(Responses, "create", forbidden)
    source = context()
    assert finalize_script_submission(source, submission()).story_input_ref.version == 7


def test_unvalidated_copies_and_mutated_context_fail_closed():
    source, proposal = context(), submission()
    bad_segment = proposal.sections[0].segments[0].model_copy(update={"grounding": None})
    bad_section = proposal.sections[0].model_copy(update={"segments": (bad_segment,)})
    with pytest.raises(ValidationError):
        finalize_script_submission(source, proposal.model_copy(update={"sections": (bad_section,)}))
    invalid_ref = source.story_input_ref.model_copy(update={"version": True})
    with pytest.raises(ValidationError):
        finalize_script_submission(source.model_copy(update={"story_input_ref": invalid_ref}), proposal)
    source.sections[0].beats[0].fact_refs[0].research_fact_id = "f1"
    with pytest.raises(ValidationError, match="must be unique"):
        finalize_script_submission(source, proposal)


@pytest.mark.parametrize("source,proposal", [(None, submission()), (context(), {})])
def test_finalization_requires_typed_inputs(source, proposal):
    with pytest.raises(TypeError, match="ScriptContext and ScriptSubmission"):
        finalize_script_submission(source, proposal)
