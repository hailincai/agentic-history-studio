import pytest
from pydantic import ValidationError

from history_studio.models import ArtifactReference, VerificationContext, build_verification_context
from history_studio.models.research import EvidenceReference, ResearchFact
from history_studio.models.research_package import ResearchPackage
from history_studio.models.verification import VerificationEvidence, VerificationResult
from history_studio.research.spans import make_spans


def research_ref(version=1):
    return ArtifactReference(project_id="test", artifact_type="research", version=version)


def source_data(source_id):
    return dict(source_id=source_id, title=f"Record {source_id}",
                url=f"https://example.org/{source_id}", source_type="PRIMARY_SOURCE",
                accessed_at="2026-01-01T00:00:00Z", notes="Original metadata")


def quotation(source_id, text):
    canonical = make_spans(source_id, text)
    return dict(source_id=source_id, source_version=canonical.source_version,
                span_id=next(iter(canonical.spans)), excerpt=canonical.text, locator="Entry 1")


def package_data():
    target = dict(fact_id="RF-target", claim="The record dates the event to 701.",
        historical_time={"display": "701", "precision": "YEAR", "start_year": 701},
        research_confidence=0.6, research_notes="Candidate interpretation, not verified.",
        people=["Historical subject"], location="Recorded locality", dispute_group_id="DG-1",
        evidence=[quotation("SRC-A", "The event occurred in 701."),
                  quotation("SRC-B", "A second original quotation."),
                  quotation("SRC-A", "A changed representation of the record.")])
    other = dict(target, fact_id="RF-other", claim="Another event occurred.",
                 evidence=[quotation("SRC-unrelated", "Unrelated quotation.")])
    return ResearchPackage(project_id="test", topic="Whole topic not projected",
        facts=[other, target], sources=[source_data(s) for s in
            ("SRC-B", "SRC-unrelated", "SRC-A", "SRC-discovered")], progress={"run_id": "run"})


def test_projection_preserves_complete_target_and_only_required_sources():
    package = package_data()
    before = package.model_dump_json()
    target = package.facts[1]
    context = build_verification_context(package, "RF-target", research_input_ref=research_ref())
    assert context.target_fact == target
    assert context.target_fact.evidence == target.evidence
    assert [s.source_id for s in context.sources] == ["SRC-B", "SRC-A"]
    assert context.sources == [package.sources[0], package.sources[2]]
    assert len(context.target_fact.evidence) == 3
    assert context.target_fact.evidence[0].source_version != context.target_fact.evidence[2].source_version
    assert set(VerificationContext.model_fields) == {"research_input_ref", "target_fact", "sources", "investigation_boundary",
        "original_evidence_role", "whole_topic_research_allowed"}
    assert "RF-other" not in context.model_dump_json()
    assert "SRC-unrelated" not in context.model_dump_json()
    assert "SRC-discovered" not in context.model_dump_json()
    assert package.model_dump_json() == before
    assert context.target_fact is not target
    assert context.target_fact.evidence[0] is not target.evidence[0]
    assert context.sources[0] is not package.sources[0]
    # Altering projected working input must not alter original research objects.
    context.target_fact.evidence[0].locator = "Context-only locator"
    context.target_fact.people.append("Context-only person")
    context.sources[0].notes = "Context-only note"
    assert package.model_dump_json() == before


def test_original_evidence_is_not_a_verification_result_or_verification_evidence():
    context = build_verification_context(package_data(), "RF-target", research_input_ref=research_ref())
    assert not isinstance(context, VerificationResult)
    assert all(type(e) is EvidenceReference and not isinstance(e, VerificationEvidence)
               for e in context.target_fact.evidence)
    assert "verification_evidence" not in context.model_dump()
    assert "status" not in context.model_dump()


def test_claim_boundary_and_json_round_trip():
    context = build_verification_context(package_data(), "RF-target", research_input_ref=research_ref())
    assert context.investigation_boundary == "TARGET_CLAIM_ONLY"
    assert context.original_evidence_role == "RESEARCH_INPUT_NOT_INDEPENDENT_VERIFICATION"
    assert context.whole_topic_research_allowed is False
    assert VerificationContext.model_validate_json(context.model_dump_json()) == context


@pytest.mark.parametrize("field, value", [
    ("investigation_boundary", "WHOLE_TOPIC"),
    ("original_evidence_role", "INDEPENDENT_VERIFICATION"),
    ("whole_topic_research_allowed", True),
])
def test_boundary_cannot_be_redefined(field, value):
    data = build_verification_context(package_data(), "RF-target", research_input_ref=research_ref()).model_dump()
    with pytest.raises(ValidationError):
        VerificationContext.model_validate(data | {field: value})


def test_unknown_target_fails_clearly():
    with pytest.raises(ValueError, match="Target research_fact_id not found"):
        build_verification_context(package_data(), "RF-missing", research_input_ref=research_ref())


def test_missing_source_metadata_in_mutated_package_is_not_silently_dropped():
    package = package_data()
    # In-place container changes bypass assignment validation; builder must still reject them.
    package.sources[:] = [s for s in package.sources if s.source_id != "SRC-A"]
    with pytest.raises(ValidationError, match="Missing source metadata"):
        build_verification_context(package, "RF-target", research_input_ref=research_ref())


def test_duplicate_target_and_source_metadata_fail_clearly():
    package = package_data()
    package.facts.append(package.facts[1].model_copy(deep=True))
    with pytest.raises(ValueError, match="not unique"):
        build_verification_context(package, "RF-target", research_input_ref=research_ref())
    package = package_data()
    package.sources.append(package.sources[0].model_copy(deep=True))
    with pytest.raises(ValidationError, match="Context source IDs must be unique"):
        build_verification_context(package, "RF-target", research_input_ref=research_ref())


def test_unrelated_metadata_rejected_in_direct_context():
    data = build_verification_context(package_data(), "RF-target", research_input_ref=research_ref()).model_dump()
    data["sources"].append(source_data("SRC-extra"))
    with pytest.raises(ValidationError, match="must be referenced"):
        VerificationContext.model_validate(data)


def test_valid_legacy_evidence_free_fact_as_standalone_context():
    # Standalone ResearchFact permits embedded sources; ResearchPackage forbids that form.
    fact = ResearchFact(fact_id="RF-legacy", claim="A legacy candidate claim.",
        time_period="Original textual date", research_confidence=0.5,
        sources=[source_data("SRC-legacy")])
    context = VerificationContext(target_fact=fact, research_input_ref=research_ref())
    assert context.target_fact.evidence == []
    assert context.target_fact.sources == fact.sources
    assert context.sources == []
    assert VerificationContext.model_validate_json(context.model_dump_json()) == context
