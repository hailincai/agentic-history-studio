"""Offline structural verification contracts, without semantic historical judgments."""
import pytest
from pydantic import ValidationError

from history_studio.models import VerificationEvidence, VerificationResult, VerificationStatus
from history_studio.models.research import ResearchFact
from history_studio.research.spans import make_spans


def evidence_data(text="A dated record supports the claim."):
    source = make_spans("SRC-record", text)
    return dict(source_id=source.source_id, source_version=source.source_version,
                span_id=next(iter(source.spans)), excerpt=source.text, locator="Record entry")


def result_data(status="VERIFIED", **changes):
    return dict(verification_id="VR-1", research_fact_id="RF-1",
                claim_snapshot="The record dates the event to 701.", status=status,
                verification_evidence=[evidence_data()],
                independence_note="The record was independently examined; independence is a semantic assessment.",
                rationale="Claim-specific investigation assessed the dated entry.") | changes


@pytest.mark.parametrize("status, changes", [
    ("VERIFIED", {}),
    ("PARTIALLY_VERIFIED", {"unresolved_issues": ["The location remains uncertain."]}),
    ("PARTIALLY_VERIFIED", {"contradiction_evidence": [evidence_data("Another record gives 702.")]}),
    ("DISPUTED", {"contradiction_evidence": [evidence_data("Another record gives 702.")]}),
    ("REJECTED", {"verification_evidence": [], "contradiction_evidence": [evidence_data("The date is 702.")]}),
    ("UNVERIFIED", {"verification_evidence": [], "unresolved_issues": ["No independent date could be established."]}),
])
def test_valid_terminal_outcomes_and_round_trip(status, changes):
    result = VerificationResult(**result_data(status, **changes))
    assert result.status == VerificationStatus(status)
    assert VerificationResult.model_validate_json(result.model_dump_json()) == result


@pytest.mark.parametrize("status, changes", [
    ("VERIFIED", {"verification_evidence": []}),
    ("PARTIALLY_VERIFIED", {}),
    ("PARTIALLY_VERIFIED", {"verification_evidence": [], "unresolved_issues": ["Unknown location"]}),
    ("DISPUTED", {}),
    ("DISPUTED", {"verification_evidence": [], "contradiction_evidence": [evidence_data()]}),
    ("REJECTED", {}),
    ("UNVERIFIED", {"verification_evidence": []}),
    ("UNVERIFIED", {"unresolved_issues": [" "]}),
])
def test_invalid_terminal_structure(status, changes):
    with pytest.raises(ValidationError):
        VerificationResult(**result_data(status, **changes))


@pytest.mark.parametrize("changes", [
    {"claim_snapshot": " "}, {"claim_snapshot": "x" * 601},
    {"verification_id": "../bad"}, {"research_fact_id": "../bad"},
    {"independence_note": " "}, {"rationale": " "}, {"unexpected": True},
])
def test_required_contract_shapes(changes):
    with pytest.raises(ValidationError):
        VerificationResult(**result_data(**changes))


def test_claim_snapshot_required_and_preserved_exactly():
    data = result_data()
    del data["claim_snapshot"]
    with pytest.raises(ValidationError):
        VerificationResult(**data)
    result = VerificationResult(**result_data(claim_snapshot="  Original claim text.  "))
    assert result.claim_snapshot == "  Original claim text.  "


@pytest.mark.parametrize("changes", [
    {"source_id": "../bad"}, {"excerpt": " "}, {"excerpt": "x" * 1201},
    {"span_id": None}, {"source_version": None}, {"arbitrary": "value"},
])
def test_verification_evidence_reuses_provenance_validation(changes):
    with pytest.raises(ValidationError):
        VerificationEvidence(**(evidence_data() | changes))


def test_research_and_verification_evidence_are_separate_without_automatic_copy():
    original = ResearchFact(fact_id="RF-1", claim="The record dates the event to 701.",
        historical_time={"display": "701", "start_year": 701, "precision": "YEAR"},
        evidence=[evidence_data("Original research quotation.")], research_confidence=0.6)
    before = original.model_dump_json()
    result = VerificationResult(**result_data(research_fact_id=original.fact_id,
        claim_snapshot=original.claim))
    assert isinstance(result.verification_evidence[0], VerificationEvidence)
    assert result.verification_evidence[0] is not original.evidence[0]
    assert result.verification_evidence[0].excerpt != original.evidence[0].excerpt
    result.verification_evidence[0].locator = "Independent locator"
    assert original.model_dump_json() == before
    # Insufficiency is UNVERIFIED, not rejection, and original evidence is not filled in.
    unverified = VerificationResult(**result_data("UNVERIFIED", verification_evidence=[],
        unresolved_issues=["Independent evidence could not be located."]))
    assert unverified.verification_evidence == []


def test_list_defaults_are_independent_and_assignment_is_validated():
    first = VerificationResult(**result_data())
    second = VerificationResult(**result_data())
    first.unresolved_issues.append("A detail remains unknown.")
    assert second.unresolved_issues == []
    assert first.contradiction_evidence is not second.contradiction_evidence
    with pytest.raises(ValidationError):
        second.status = "NOT_A_STATUS"
