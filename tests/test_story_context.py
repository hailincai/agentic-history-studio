import json

import pytest
from pydantic import ValidationError

from history_studio.models import ArtifactReference, StoryContext, StoryFactReference
from history_studio.models.verification_package import VerificationPackage, create_verification_package
from history_studio.models.verification import VerificationResult
from history_studio.storage.artifact_store import ArtifactStore, ArtifactNotFoundError
from history_studio.story import build_story_context
from test_verification_context import package_data, research_ref
from test_verification_models import result_data, evidence_data


def setup(tmp_path, status="VERIFIED"):
    store = ArtifactStore(tmp_path / "test")
    research = package_data()
    store.save("research", research)
    verification = create_verification_package(research, research_input_ref=research_ref())
    # Only target membership is approved; research-only facts must remain invisible.
    verification = VerificationPackage.model_validate(verification.model_dump(mode="json") | {
        "research_facts": [verification.research_facts[1].model_dump(mode="json")]})
    changes = {}
    if status in ("PARTIALLY_VERIFIED", "UNVERIFIED"):
        changes["unresolved_issues"] = ["Date remains uncertain"]
    if status in ("DISPUTED", "REJECTED"):
        changes["contradiction_evidence"] = [evidence_data("A competing date")]
    if status in ("REJECTED", "UNVERIFIED"):
        changes["verification_evidence"] = []
    result = VerificationResult(**result_data(status, research_fact_id="RF-target", **changes))
    verification = VerificationPackage.model_validate(verification.model_dump(mode="json") | {
        "results": [result.model_dump(mode="json")]})
    store.save("verification", verification)
    ref = ArtifactReference(project_id="test", artifact_type="verification", version=1)
    return store, ref, research, verification


@pytest.mark.parametrize("status,uses,qualified", [
    ("VERIFIED", ("AFFIRMATIVE", "QUALIFIED"), False),
    ("PARTIALLY_VERIFIED", ("QUALIFIED",), True),
    ("DISPUTED", ("DISPUTE",), True),
    ("REJECTED", (), False), ("UNVERIFIED", (), False),
])
def test_status_authority_and_exact_projection(tmp_path, status, uses, qualified):
    store, ref, research, verification = setup(tmp_path, status)
    before = (research.model_dump_json(), verification.model_dump_json())
    context = build_story_context(store, verification_input_ref=ref)
    assert context.verification_input_ref == ref
    assert context.research_input_ref == research_ref()
    fact = context.fact("RF-target")
    assert fact.status.value == status
    assert fact.allowed_uses == uses
    assert fact.qualification_required is qualified
    assert fact.claim_snapshot == verification.results[0].claim_snapshot == research.facts[1].claim
    assert fact.historical_time == research.facts[1].historical_time
    assert fact.verification_evidence == tuple(verification.results[0].verification_evidence)
    assert fact.contradiction_evidence == tuple(verification.results[0].contradiction_evidence)
    assert fact.unresolved_issues == tuple(verification.results[0].unresolved_issues)
    assert fact.rationale == verification.results[0].rationale
    assert bool(context.eligible_facts) == bool(uses)
    assert bool(context.excluded_facts) != bool(uses)
    assert StoryContext.model_validate_json(context.model_dump_json()) == context
    assert before == (research.model_dump_json(), verification.model_dump_json())
    payload = context.model_dump_json()
    for forbidden in ("RF-other", "research_confidence", "research_notes", "progress",
                      "sources", "independence_note", "verification_id", "ToolObservation"):
        assert forbidden not in payload
    fact.verification_evidence = ()
    assert context.fact("RF-target").verification_evidence == tuple(verification.results[0].verification_evidence)


def test_reference_checks(tmp_path):
    store, ref, _, _ = setup(tmp_path, "PARTIALLY_VERIFIED")
    context = build_story_context(store, verification_input_ref=ref)
    reference = StoryFactReference(research_fact_id="RF-target", status="PARTIALLY_VERIFIED",
                                   use="QUALIFIED", qualification="Date remains uncertain")
    context.validate_fact_reference(reference, expected_claim=context.fact("RF-target").claim_snapshot)
    with pytest.raises(ValueError, match="exactly match"):
        context.validate_fact_reference(reference, expected_claim="Different claim")
    with pytest.raises(ValueError, match="authoritative"):
        context.validate_fact_reference(StoryFactReference(
            research_fact_id="RF-target", status="VERIFIED", use="AFFIRMATIVE"))
    with pytest.raises(ValueError, match="no accepted result"):
        context.fact("RF-other")


@pytest.mark.parametrize("kind", ["type", "project", "version"])
def test_invalid_verification_reference(tmp_path, kind):
    store, ref, _, _ = setup(tmp_path)
    changes = {"type": {"artifact_type": "research"}, "project": {"project_id": "other"},
               "version": {"version": 99}}[kind]
    with pytest.raises((ValueError, ArtifactNotFoundError)):
        build_story_context(store, verification_input_ref=ArtifactReference.model_validate(ref.model_dump() | changes))


def overwrite(store, artifact_type, data):
    # Simulate corrupt persisted inputs that bypass model construction.
    path = store.project_dir / artifact_type / f"{artifact_type}_v1.json"
    path.write_text(json.dumps(data), encoding="utf-8")


@pytest.mark.parametrize("failure", ["claim", "missing", "duplicate", "invalid_id", "project"])
def test_research_identity_fails_closed(tmp_path, failure):
    store, ref, research, _ = setup(tmp_path)
    data = research.model_dump(mode="json")
    if failure == "claim":
        data["facts"][1]["claim"] = "A different claim"
    elif failure == "missing":
        data["facts"].pop()
    elif failure == "duplicate":
        data["facts"].append(data["facts"][1])
    elif failure == "invalid_id":
        data["facts"][1]["fact_id"] = "../bad"
    else:
        data["project_id"] = "other"
    overwrite(store, "research", data)
    with pytest.raises(ValueError):
        build_story_context(store, verification_input_ref=ref)


def test_exact_research_version_no_latest_fallback(tmp_path):
    store, ref, research, _ = setup(tmp_path)
    changed = research.model_copy(deep=True)
    changed.facts[1].historical_time.display = "Changed chronology in v2"
    assert store.save("research", changed) == 2
    context = build_story_context(store, verification_input_ref=ref)
    assert context.fact("RF-target").historical_time == research.facts[1].historical_time
    assert context.research_input_ref.version == 1
    # A matching-ID alternative version cannot substitute for the missing exact snapshot.
    (store.project_dir / "research" / "research_v1.json").unlink()
    with pytest.raises(ArtifactNotFoundError):
        build_story_context(store, verification_input_ref=ref)


@pytest.mark.parametrize("failure", ["duplicate", "claim", "research_type", "research_project"])
def test_verification_identity_fails_closed(tmp_path, failure):
    store, ref, _, verification = setup(tmp_path)
    data = verification.model_dump(mode="json")
    if failure == "duplicate":
        data["research_facts"] *= 2
    elif failure == "claim":
        data["results"][0]["claim_snapshot"] = "Different claim"
    else:
        field, value = ("artifact_type", "story") if failure == "research_type" else ("project_id", "other")
        data["research_input_ref"][field] = value
        data["results"][0]["research_input_ref"][field] = value
    overwrite(store, "verification", data)
    with pytest.raises(ValueError):
        build_story_context(store, verification_input_ref=ref)


def test_pending_membership_is_caution_without_invented_status(tmp_path):
    store, ref, _, verification = setup(tmp_path)
    overwrite(store, "verification", verification.model_dump(mode="json") | {"results": []})
    context = build_story_context(store, verification_input_ref=ref)
    assert not context.eligible_facts and not context.excluded_facts
    assert context.pending_claims[0].research_fact_id == "RF-target"
    assert "status" not in context.pending_claims[0].model_dump()
    with pytest.raises(ValueError, match="no accepted result"):
        context.fact("RF-target")


def test_context_rejects_excluded_verdict_in_eligible_group(tmp_path):
    store, ref, _, _ = setup(tmp_path, "REJECTED")
    context = build_story_context(store, verification_input_ref=ref)
    with pytest.raises(ValidationError, match="Ineligible"):
        StoryContext.model_validate(context.model_dump() | {
            "eligible_facts": context.excluded_facts, "excluded_facts": ()})
