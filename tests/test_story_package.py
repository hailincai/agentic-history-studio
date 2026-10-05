import pytest
from pydantic import ValidationError

from history_studio.models import (
    ArtifactReference, HistoricalTime, StoryFactReference, StoryFactChronology, StoryNarrativeBeat,
    StoryPackage, StorySection, StoryStructure, VerificationStatus,
)


def reference(fact_id="f1", status="VERIFIED", use="AFFIRMATIVE", **kwargs):
    return StoryFactReference(research_fact_id=fact_id, status=status, use=use, **kwargs)


def beat(beat_id="beat_01", **kwargs):
    return StoryNarrativeBeat(**(dict(
        beat_id=beat_id, narrative_role="opening", summary="Introduce the grounded claim",
        fact_refs=[reference()], fact_chronology=[chronology()],
    ) | kwargs))


def chronology(fact_id="f1", time=None):
    return StoryFactChronology(research_fact_id=fact_id, historical_time=time or
                              HistoricalTime(display="701", start_year=701, precision="YEAR"))


def plan(sections=None):
    return StoryStructure(title="Working title", narrative_thesis="Trace change chronologically",
                          sections=sections or [StorySection(section_id="section_01",
                                                             purpose="Introduce", beats=[beat()])])


def test_grounded_package_retains_exact_provenance_and_round_trips():
    source = ArtifactReference(project_id="project", artifact_type="verification", version=7)
    package = StoryPackage(verification_input_ref=source, plan=plan())
    restored = StoryPackage.model_validate_json(package.model_dump_json())
    assert restored == package
    assert restored.verification_input_ref == source
    assert restored.verification_input_ref.version == 7
    assert restored.schema_version == 1


def test_wrong_provenance_type():
    with pytest.raises(ValidationError, match="verification artifact"):
        StoryPackage(verification_input_ref=ArtifactReference(
            project_id="project", artifact_type="research", version=7), plan=plan())


def test_duplicate_section_and_global_beat_ids():
    first = StorySection(section_id="section_01", purpose="Opening", beats=[beat()])
    for sections in ([first, first], [first, StorySection(
            section_id="section_02", purpose="Ending", beats=[beat()])]):
        with pytest.raises(ValidationError, match="must be unique"):
            plan(sections)
    with pytest.raises(ValidationError, match="Beat IDs"):
        StorySection(section_id="section_01", purpose="Opening", beats=[beat(), beat()])


def test_historical_grounding_and_chronology_required():
    for changes, message in [({"fact_refs": []}, "fact grounding"),
                             ({"fact_chronology": []}, "exactly match")]:
        with pytest.raises(ValidationError, match=message):
            beat(**changes)


def test_structural_beat_policy():
    structural = StoryNarrativeBeat(beat_id="transition_01", kind="structural",
                                   narrative_role="transition", summary="Move to the next section")
    assert not structural.fact_refs and not structural.fact_chronology
    for changes in ({"fact_refs": [reference()]},
                    {"fact_chronology": [chronology()]}):
        with pytest.raises(ValidationError, match="Structural beat"):
            StoryNarrativeBeat.model_validate(structural.model_dump() | changes)


def test_multiple_facts_and_reuse_across_beats():
    first = beat(fact_refs=[reference(), reference("f2")],
                 fact_chronology=[chronology(), chronology("f2")])
    second = beat("beat_02", fact_refs=[reference()])
    result = plan([StorySection(section_id="section_01", purpose="Develop", beats=[first, second])])
    assert len(result.sections[0].beats[0].fact_refs) == 2
    assert result.sections[0].beats[1].fact_refs[0].research_fact_id == "f1"
    with pytest.raises(ValidationError, match="Beat fact references"):
        beat(fact_refs=[reference(), reference()])


@pytest.mark.parametrize("time", [
    HistoricalTime(display="1 BCE", start_year=0, precision="YEAR"),
    HistoricalTime(display="701–725", start_year=701, end_year=725, precision="RANGE"),
    HistoricalTime(display="circa the beginning", precision="APPROXIMATE"),
    HistoricalTime(display="Unknown date", precision="UNKNOWN"),
])
def test_chronology_round_trip_without_invented_bounds(time):
    original = beat(fact_chronology=[chronology(time=time)])
    assert StoryNarrativeBeat.model_validate_json(original.model_dump_json()).fact_chronology[0].historical_time == time


def test_invalid_chronology_rejected():
    with pytest.raises(ValidationError, match="chronological"):
        beat(fact_chronology=[dict(research_fact_id="f1", historical_time=dict(
            display="Backwards", start_year=725, end_year=701, precision="RANGE"))])


@pytest.mark.parametrize("status,use", [
    ("VERIFIED", "AFFIRMATIVE"), ("VERIFIED", "QUALIFIED"),
    ("PARTIALLY_VERIFIED", "QUALIFIED"), ("DISPUTED", "DISPUTE"),
])
def test_eligible_status_use_preserves_declared_verdict(status, use):
    ref = reference(status=status, use=use, qualification="Explicit uncertainty")
    assert ref.status == VerificationStatus(status)
    assert StoryFactReference.model_validate_json(ref.model_dump_json()) == ref


@pytest.mark.parametrize("status", ["REJECTED", "UNVERIFIED"])
@pytest.mark.parametrize("use", ["AFFIRMATIVE", "QUALIFIED", "DISPUTE"])
def test_ineligible_statuses_cannot_ground_narrative(status, use):
    with pytest.raises(ValidationError, match="not eligible"):
        reference(status=status, use=use, qualification="Uncertainty")


@pytest.mark.parametrize("status,use", [
    ("PARTIALLY_VERIFIED", "AFFIRMATIVE"), ("DISPUTED", "AFFIRMATIVE"),
    ("DISPUTED", "QUALIFIED"), ("VERIFIED", "DISPUTE"),
])
def test_status_cannot_be_silently_strengthened(status, use):
    with pytest.raises(ValidationError, match="not eligible"):
        reference(status=status, use=use, qualification="Uncertainty")


@pytest.mark.parametrize("status,use", [("PARTIALLY_VERIFIED", "QUALIFIED"), ("DISPUTED", "DISPUTE")])
def test_uncertain_use_requires_explicit_qualification(status, use):
    with pytest.raises(ValidationError, match="explicit qualification"):
        reference(status=status, use=use)


def test_structural_validation_does_not_claim_semantic_proof():
    # Unknown membership and unsuitable prose require the later loaded-package boundary.
    value = beat(summary="An unsupported assertion", fact_refs=[reference("unknown_fact")],
                 fact_chronology=[chronology("unknown_fact")])
    assert value.fact_refs[0].research_fact_id == "unknown_fact"


def test_distinct_fact_times_preserved_without_synthetic_beat_time_or_sorting():
    early = chronology("f1")
    late = chronology("f2", HistoricalTime(display="725", start_year=725, precision="YEAR"))
    value = beat(fact_refs=[reference(), reference("f2")], fact_chronology=[late, early])
    assert value.fact_chronology == (late, early)
    restored = StoryNarrativeBeat.model_validate_json(value.model_dump_json())
    assert restored == value
    assert [entry.historical_time.start_year for entry in restored.fact_chronology] == [725, 701]
    assert "historical_time" not in value.model_dump()
    with pytest.raises(ValidationError, match="Extra inputs"):
        StoryNarrativeBeat.model_validate(value.model_dump() | {"historical_time": early.historical_time})


@pytest.mark.parametrize("entries", [[], ["f1"], ["f1", "f2", "unrelated"], ["f1", "unrelated"]])
def test_chronology_membership_must_exactly_match_grounding(entries):
    with pytest.raises(ValidationError, match="exactly match"):
        beat(fact_refs=[reference(), reference("f2")],
             fact_chronology=[chronology(fact_id) for fact_id in entries])


def test_duplicate_chronology_identity_rejected():
    with pytest.raises(ValidationError, match="Beat chronology fact IDs must be unique"):
        beat(fact_chronology=[chronology(), chronology()])
