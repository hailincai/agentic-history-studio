"""Deterministic accepted-knowledge aggregation; no stage execution or external calls."""
import json

import pytest
from pydantic import ValidationError

from history_studio.models import (
    ArtifactReference, ResearchFactSnapshot, VerificationPackage, VerificationResult,
    add_verification_result, create_verification_package,
)
from history_studio.models.research_package import ResearchPackage
from history_studio.storage.artifact_store import ArtifactStore
from test_verification_context import package_data, research_ref
from test_verification_models import evidence_data, result_data


def research_data():
    data = package_data().model_dump(mode="json")
    data["facts"].append(data["facts"][0] | {"fact_id": "RF-third", "claim": "A third event occurred."})
    return ResearchPackage.model_validate(data)


def empty_package():
    return create_verification_package(research_data(), research_input_ref=research_ref(4))


def result_for(package, index=0, status="VERIFIED", **changes):
    fact = package.research_facts[index]
    return VerificationResult(**result_data(status, verification_id=f"VR-{index}",
        research_input_ref=package.research_input_ref, research_fact_id=fact.research_fact_id,
        claim_snapshot=fact.claim_snapshot, **changes))


def test_creation_captures_compact_ordered_membership_and_exact_input_without_mutation():
    research, reference = research_data(), research_ref(4)
    before = research.model_dump_json()
    package = create_verification_package(research, research_input_ref=reference)
    assert package.research_input_ref == reference and package.research_input_ref is not reference
    assert package.schema_version == 1 and package.results == ()
    assert [f.research_fact_id for f in package.research_facts] == [f.fact_id for f in research.facts]
    assert [f.claim_snapshot for f in package.research_facts] == [f.claim for f in research.facts]
    assert set(ResearchFactSnapshot.model_fields) == {"research_fact_id", "claim_snapshot"}
    assert set(package.model_dump()) == {"schema_version", "research_input_ref", "research_facts", "results"}
    assert package.completed_fact_ids == []
    assert package.pending_fact_ids == [f.fact_id for f in research.facts]
    assert package.is_complete is False
    assert research.model_dump_json() == before
    research.facts[0].claim = "The caller changed its candidate assertion."
    assert package.research_facts[0].claim_snapshot != research.facts[0].claim


def test_creation_requires_explicit_version_and_rejects_project_or_type_mismatch():
    with pytest.raises(TypeError, match="research_input_ref"):
        create_verification_package(research_data())
    with pytest.raises(ValueError, match="project_id"):
        create_verification_package(research_data(), research_input_ref=
            ArtifactReference(project_id="other", artifact_type="research", version=4))
    with pytest.raises(ValidationError, match="research artifact"):
        create_verification_package(research_data(), research_input_ref=
            ArtifactReference(project_id="test", artifact_type="story", version=4))
    first = create_verification_package(research_data(), research_input_ref=research_ref(3))
    second = empty_package()
    assert first.research_facts == second.research_facts
    assert first.research_input_ref != second.research_input_ref
    assert research_data().schema_version not in (first.research_input_ref.version, second.research_input_ref.version)


def test_partial_complete_and_ordering_ignore_result_insertion_order():
    package = empty_package()
    third = add_verification_result(package, result_for(package, 2))
    partial = add_verification_result(third, result_for(package, 0))
    assert partial.completed_fact_ids == ["RF-other", "RF-third"]
    assert partial.pending_fact_ids == ["RF-target"] and partial.is_complete is False
    complete = add_verification_result(partial, result_for(package, 1))
    assert [r.research_fact_id for r in complete.results] == ["RF-third", "RF-other", "RF-target"]
    assert complete.completed_fact_ids == ["RF-other", "RF-target", "RF-third"]
    assert complete.pending_fact_ids == [] and complete.is_complete is True
    assert package.results == () and third.pending_fact_ids == ["RF-other", "RF-target"]
    assert partial.results == complete.results[:2]
    completed = complete.completed_fact_ids
    completed.clear()
    assert len(complete.completed_fact_ids) == 3


def test_copy_on_update_detaches_existing_and_new_results_without_modifying_callers():
    original = empty_package()
    result = result_for(original)
    before = result.model_dump_json()
    partial = add_verification_result(original, result)
    partial_before = partial.model_dump_json()
    next_result = result_for(original, 1)
    updated = add_verification_result(partial, next_result)
    assert original.results == () and result.model_dump_json() == before
    assert partial.model_dump_json() == partial_before
    assert updated.results[0] == result and updated.results[0] is not partial.results[0]
    assert updated.results[1] == next_result and updated.results[1] is not next_result
    result.verification_evidence[0].locator = "Caller-only locator"
    next_result.rationale = "Caller-only rationale"
    assert updated.results[0].verification_evidence[0].locator != result.verification_evidence[0].locator
    assert updated.results[1].rationale != next_result.rationale
    with pytest.raises(ValidationError):
        updated.research_input_ref = research_ref(3)
    with pytest.raises(ValidationError):
        updated.research_facts[0].claim_snapshot = "Changed membership"
    with pytest.raises(AttributeError):
        updated.results.append(result)


@pytest.mark.parametrize("failure, message", [
    ("snapshot", "research snapshot"), ("project", "research snapshot"),
    ("foreign", "captured research membership"), ("claim", "exactly match"),
    ("claim_whitespace", "exactly match"), ("duplicate", "must be unique"),
])
def test_invalid_result_rejected_by_update_and_direct_deserialization(failure, message):
    package = empty_package()
    data = result_for(package).model_dump(mode="json")
    if failure == "snapshot":
        data["research_input_ref"]["version"] = 3
    elif failure == "project":
        data["research_input_ref"]["project_id"] = "other"
    elif failure == "foreign":
        data["research_fact_id"] = "RF-foreign"
    elif failure == "claim":
        data["claim_snapshot"] = "A different assertion."
    elif failure == "claim_whitespace":
        data["claim_snapshot"] += " "
    else:
        package = add_verification_result(package, result_for(package))
        data["verification_id"] = "VR-another"
    result = VerificationResult.model_validate(data)
    before = package.model_dump_json()
    with pytest.raises(ValidationError, match=message):
        add_verification_result(package, result)
    with pytest.raises(ValidationError, match=message):
        VerificationPackage.model_validate_json(json.dumps(package.model_dump(mode="json") | {
            "results": [r.model_dump(mode="json") for r in package.results] + [data]}))
    assert package.model_dump_json() == before


@pytest.mark.parametrize("status, changes", [
    ("VERIFIED", {}),
    ("PARTIALLY_VERIFIED", {"unresolved_issues": ["Date remains uncertain"]}),
    ("DISPUTED", {"contradiction_evidence": [evidence_data("Another record disputes the date.")]}),
    ("REJECTED", {"verification_evidence": [], "contradiction_evidence": [evidence_data()]}),
    ("UNVERIFIED", {"verification_evidence": [], "unresolved_issues": ["No independent support"]}),
])
def test_every_status_and_canonical_evidence_role_preserved_through_round_trip(status, changes):
    package = empty_package()
    result = result_for(package, status=status, **changes)
    updated = add_verification_result(package, result)
    restored = VerificationPackage.model_validate_json(updated.model_dump_json())
    assert restored == updated
    assert restored.results[0].model_dump() == result.model_dump()
    assert restored.results[0].status.value == status
    assert restored.completed_fact_ids == ["RF-other"]
    assert restored.pending_fact_ids == ["RF-target", "RF-third"]
    assert restored.is_complete is False


@pytest.mark.parametrize("changes", [
    {"schema_version": 2}, {"is_complete": True}, {"pending_fact_ids": []},
    {"read_authorization": {}}, {"current_observation": {}},
])
def test_unknown_schema_or_transient_or_derived_fields_rejected(changes):
    with pytest.raises(ValidationError):
        VerificationPackage.model_validate(empty_package().model_dump(mode="json") | changes)


def test_schema_excludes_transient_execution_and_redundant_completion_state():
    schema = json.dumps(VerificationPackage.model_json_schema())
    for name in ("ToolObservation", "NativeToolCall", "ModelRequest", "ModelResponse",
                 "read_authorization", "provider", "transcript", "current_observation",
                 "completed_fact_ids", "pending_fact_ids", "is_complete"):
        assert name not in schema


def test_invalid_membership_and_reference_are_rejected_on_load():
    data = empty_package().model_dump(mode="json")
    with pytest.raises(ValidationError, match="must be unique"):
        VerificationPackage.model_validate(data | {"research_facts": data["research_facts"] * 2})
    with pytest.raises(ValidationError, match="research artifact"):
        VerificationPackage.model_validate(data | {"research_input_ref":
            dict(project_id="test", artifact_type="story", version=4)})
    data.pop("research_input_ref")
    with pytest.raises(ValidationError):
        VerificationPackage.model_validate(data)
    for claim in ("", " ", "x" * 601):
        with pytest.raises(ValidationError):
            ResearchFactSnapshot(research_fact_id="F1", claim_snapshot=claim)


def test_update_revalidates_mutated_nested_results():
    package = empty_package()
    partial = add_verification_result(package, result_for(package))
    # Nested result models keep their existing mutability; the boundary must revalidate.
    partial.results[0].verification_evidence.clear()
    with pytest.raises(ValidationError, match="VERIFIED requires"):
        add_verification_result(partial, result_for(package, 1))


def test_empty_membership_has_no_pending_work():
    data = research_data().model_dump(mode="json")
    data["facts"] = []
    package = create_verification_package(ResearchPackage.model_validate(data), research_input_ref=research_ref(4))
    assert package.completed_fact_ids == package.pending_fact_ids == []
    assert package.is_complete is True


def test_versioned_store_preserves_earlier_partial_knowledge(tmp_path):
    store = ArtifactStore(tmp_path / "test")
    package = empty_package()
    first = add_verification_result(package, result_for(package, 0))
    assert store.save("verification", first) == 1
    first_path = tmp_path / "test" / "verification" / "verification_v1.json"
    original_bytes = first_path.read_bytes()
    second = add_verification_result(first, result_for(package, 1, status="PARTIALLY_VERIFIED",
                                                       unresolved_issues=["Location remains uncertain"]))
    assert store.save("verification", second) == 2
    assert first_path.read_bytes() == original_bytes
    assert store.load("verification", 1, VerificationPackage) == first
    loaded = store.load_latest("verification", VerificationPackage)
    assert loaded == second and loaded.completed_fact_ids == ["RF-other", "RF-target"]
    assert loaded.pending_fact_ids == ["RF-third"] and loaded.is_complete is False
    assert first.pending_fact_ids == ["RF-target", "RF-third"]


def test_finalized_single_fact_result_is_accepted_without_extra_provider_or_tool_calls():
    from history_studio.models import build_verification_context
    from history_studio.verification import FactChecker
    from test_fact_checker_investigation import SequenceProvider
    from test_verification_evidence import TextTools, read, submission

    research = research_data()
    package = create_verification_package(research, research_input_ref=research_ref(4))
    context = build_verification_context(research, "RF-target", research_input_ref=research_ref(4))
    tools, provider = TextTools(), SequenceProvider([])
    checker = FactChecker(context, tools=tools, provider=provider)
    selection, _, _ = read(checker, tools)
    proposal = submission(checker, selection)
    calls_before = list(tools.calls)
    result = checker.finalize_submission(proposal)
    updated = add_verification_result(package, result)
    assert updated.results == (result,)
    assert updated.completed_fact_ids == ["RF-target"]
    assert updated.research_input_ref == result.research_input_ref == research_ref(4)
    assert provider.requests == [] and tools.calls == calls_before
