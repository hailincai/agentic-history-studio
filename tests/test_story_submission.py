import pytest
from pydantic import ValidationError

from history_studio.models import HistoricalTime, StoryPackage
from history_studio.story import StorySubmission, finalize_story_submission
from test_story_architect import context


def selection(fact_id="RF-0", use="AFFIRMATIVE", qualification=None):
    return dict(research_fact_id=fact_id, use=use, qualification=qualification)


def beat(beat_id="beat_01", **changes):
    return dict(beat_id=beat_id, narrative_role="opening", summary="Grounded narrative decision",
                fact_proposals=[selection()]) | changes


def submission(**changes):
    return StorySubmission.model_validate(dict(title="Working title", narrative_thesis="Trace change",
        sections=[dict(section_id="section_01", purpose="Introduce", beats=[beat()])]) | changes)


def with_beats(beats):
    return submission(sections=[dict(section_id="section_01", purpose="Introduce", beats=beats)])


def test_valid_submission_exact_provenance_status_time_and_detached_output():
    source, proposal = context(), submission()
    before = (source.model_dump_json(), proposal.model_dump_json())
    result = finalize_story_submission(source, proposal)
    assert isinstance(result, StoryPackage) and not isinstance(proposal, StoryPackage)
    assert result.verification_input_ref == source.verification_input_ref
    assert result.verification_input_ref.version == 7
    accepted = result.plan.sections[0].beats[0]
    assert accepted.fact_refs[0].status == source.eligible_facts[0].status
    assert accepted.fact_refs[0].use == "AFFIRMATIVE"
    assert accepted.fact_chronology[0].historical_time == source.eligible_facts[0].historical_time
    assert before == (source.model_dump_json(), proposal.model_dump_json())
    assert finalize_story_submission(source, proposal) == result
    assert StoryPackage.model_validate_json(result.model_dump_json()) == result
    accepted.fact_chronology[0].historical_time.display = "Output-only change"
    accepted.summary = "Output-only summary"
    assert before == (source.model_dump_json(), proposal.model_dump_json())


@pytest.mark.parametrize("fact_id", ["unknown", "RF-3", "RF-4", "RF-pending"])
def test_unknown_excluded_and_pending_cannot_ground(fact_id):
    proposal = with_beats([beat(fact_proposals=[selection(fact_id)])])
    with pytest.raises(ValueError, match="eligible context"):
        finalize_story_submission(context(), proposal)


@pytest.mark.parametrize("fact_id,use,qualification,status", [
    ("RF-0", "AFFIRMATIVE", None, "VERIFIED"),
    ("RF-0", "QUALIFIED", "Explicit qualification", "VERIFIED"),
    ("RF-1", "QUALIFIED", "Date remains uncertain", "PARTIALLY_VERIFIED"),
    ("RF-2", "DISPUTE", "Competing accounts remain", "DISPUTED"),
])
def test_allowed_use_authoritative_status(fact_id, use, qualification, status):
    proposal = with_beats([beat(fact_proposals=[selection(fact_id, use, qualification)])])
    result = finalize_story_submission(context(), proposal)
    ref = result.plan.sections[0].beats[0].fact_refs[0]
    assert ref.status == status and ref.use == use and ref.qualification == qualification


@pytest.mark.parametrize("fact_id,use,qualification", [
    ("RF-1", "QUALIFIED", None), ("RF-1", "AFFIRMATIVE", None),
    ("RF-2", "AFFIRMATIVE", None), ("RF-2", "QUALIFIED", "A hedge"),
    ("RF-2", "DISPUTE", None), ("RF-0", "QUALIFIED", None),
])
def test_ineligible_use_or_missing_qualification_rejected(fact_id, use, qualification):
    proposal = with_beats([beat(fact_proposals=[selection(fact_id, use, qualification)])])
    with pytest.raises(ValidationError):
        finalize_story_submission(context(), proposal)


def test_multifact_chronology_preserved_independently_without_sorting():
    source = context()
    source.eligible_facts[0].historical_time = HistoricalTime(display="725", start_year=725, precision="YEAR")
    source.eligible_facts[1].historical_time = HistoricalTime(display="701", start_year=701, precision="YEAR")
    proposal = with_beats([beat(fact_proposals=[selection(), selection("RF-1", "QUALIFIED", "Qualification")])])
    result = finalize_story_submission(source, proposal)
    accepted = result.plan.sections[0].beats[0]
    assert [entry.research_fact_id for entry in accepted.fact_chronology] == ["RF-0", "RF-1"]
    assert [entry.historical_time.start_year for entry in accepted.fact_chronology] == [725, 701]
    assert [entry.historical_time for entry in accepted.fact_chronology] == [
        fact.historical_time for fact in source.eligible_facts[:2]]
    assert "historical_time" not in accepted.model_dump()


def test_structural_and_reused_fact_beats_preserve_agent_order():
    proposal = with_beats([beat("beat_02"), beat("transition", kind="structural", fact_proposals=[]), beat()])
    result = finalize_story_submission(context(), proposal)
    beats = result.plan.sections[0].beats
    assert [item.beat_id for item in beats] == ["beat_02", "transition", "beat_01"]
    assert beats[1].fact_refs == beats[1].fact_chronology == ()
    assert beats[0].fact_refs == beats[2].fact_refs


@pytest.mark.parametrize("changes", [
    {"fact_proposals": []}, {"kind": "structural"},
    {"fact_proposals": [selection(), selection()]}, {"beat_id": "../bad"},
])
def test_invalid_beat_shape(changes):
    with pytest.raises(ValidationError):
        with_beats([beat(**changes)])


def test_duplicate_section_and_global_beat_ids():
    original = submission().model_dump(mode="json")
    section = original["sections"][0]
    for sections in ([section, section], [section, section | {"section_id": "section_02"}]):
        with pytest.raises(ValidationError, match="must be unique"):
            submission(sections=sections)
    with pytest.raises(ValidationError, match="must be unique"):
        with_beats([beat(), beat()])


@pytest.mark.parametrize("level,field,value", [
    ("submission", "verification_input_ref", {}), ("submission", "project_id", "other"),
    ("submission", "schema_version", 1), ("fact", "status", "VERIFIED"),
    ("fact", "historical_time", {}), ("fact", "source_version", "version"),
    ("beat", "fact_chronology", []), ("beat", "historical_time", {}),
])
def test_model_cannot_inject_authoritative_metadata(level, field, value):
    data = submission().model_dump(mode="json")
    target = data if level == "submission" else data["sections"][0]["beats"][0]
    if level == "fact":
        target = target["fact_proposals"][0]
    target[field] = value
    with pytest.raises(ValidationError, match="Extra inputs"):
        StorySubmission.model_validate(data)


def test_finalization_revalidates_unvalidated_copies_and_context():
    source, proposal = context(), submission()
    bad_beat = proposal.sections[0].beats[0].model_copy(update={"fact_proposals": ()})
    bad_section = proposal.sections[0].model_copy(update={"beats": (bad_beat,)})
    bad = proposal.model_copy(update={"sections": (bad_section,)})
    with pytest.raises(ValidationError, match="fact grounding"):
        finalize_story_submission(source, bad)
    source.eligible_facts[0].research_fact_id = "RF-1"
    with pytest.raises(ValidationError, match="must be unique"):
        finalize_story_submission(source, proposal)


def test_structural_acceptance_does_not_claim_semantic_validation():
    proposal = with_beats([beat("structural", kind="structural", fact_proposals=[],
                                summary="An unsupported historical assertion")])
    result = finalize_story_submission(context(), proposal)
    assert result.plan.sections[0].beats[0].summary == "An unsupported historical assertion"
