"""Runtime snapshot lineage, with immutable local artifacts and no external calls."""
import hashlib
import json

import pytest
from pydantic import ValidationError

from history_studio.models import ArtifactReference, VerificationContext, VerificationResult, build_verification_context
from history_studio.models.research_package import ResearchPackage
from history_studio.storage.artifact_store import ArtifactStore
from history_studio.verification import FactChecker
from history_studio.verification.submission import VerificationSubmission, VerificationSubmissionInput
from test_fact_checker_investigation import SequenceProvider
from test_verification_context import package_data, research_ref
from test_verification_evidence import TextTools, read, submission
from test_verification_models import result_data
from test_verification_submission import payload, terminal


def test_generic_reference_round_trip_and_immutability():
    reference = ArtifactReference(project_id="test", artifact_type="story", version=3)
    assert ArtifactReference.model_validate_json(reference.model_dump_json()) == reference
    assert reference.model_dump() == dict(project_id="test", artifact_type="story", version=3)
    with pytest.raises(ValidationError):
        reference.version = 4


@pytest.mark.parametrize("changes", [
    {"version": 0}, {"version": -1}, {"version": True}, {"version": "3"},
    {"version": 3.0}, {"project_id": "../test"}, {"artifact_type": "../research"},
    {"artifact_type": ""}, {"schema_version": 2},
])
def test_malformed_reference_rejected(changes):
    with pytest.raises(ValidationError):
        ArtifactReference(**(research_ref().model_dump() | changes))


def test_schema_version_cannot_replace_artifact_version():
    with pytest.raises(ValidationError):
        ArtifactReference(project_id="test", artifact_type="research", schema_version=2)


def test_context_and_builder_require_explicit_snapshot():
    package = package_data()
    with pytest.raises(TypeError, match="research_input_ref"):
        build_verification_context(package, "RF-target")
    context = build_verification_context(package, "RF-target", research_input_ref=research_ref(4))
    assert context.research_input_ref == research_ref(4)
    assert package.schema_version != context.research_input_ref.version
    data = context.model_dump()
    del data["research_input_ref"]
    with pytest.raises(ValidationError):
        VerificationContext.model_validate(data)
    assert VerificationContext.model_validate_json(context.model_dump_json()) == context


def test_builder_rejects_wrong_project_or_artifact_type():
    with pytest.raises(ValueError, match="project_id"):
        build_verification_context(package_data(), "RF-target", research_input_ref=
            ArtifactReference(project_id="other", artifact_type="research", version=3))
    wrong_type = ArtifactReference(project_id="test", artifact_type="story", version=3)
    with pytest.raises(ValidationError, match="research artifact"):
        build_verification_context(package_data(), "RF-target", research_input_ref=wrong_type)
    with pytest.raises(ValidationError, match="research artifact"):
        VerificationResult(**result_data(research_input_ref=wrong_type))


def test_result_requires_snapshot_and_round_trips():
    data = result_data()
    del data["research_input_ref"]
    with pytest.raises(ValidationError):
        VerificationResult(**data)
    result = VerificationResult(**result_data(research_input_ref=research_ref(4)))
    assert VerificationResult.model_validate_json(result.model_dump_json()) == result
    assert result.research_input_ref == research_ref(4)


def test_agent_cannot_override_snapshot_in_proposal_or_bound_submission():
    checker = FactChecker(build_verification_context(package_data(), "RF-target",
        research_input_ref=research_ref(4)))
    injected = payload() | {"research_input_ref": research_ref(3).model_dump()}
    with pytest.raises(ValidationError):
        checker.submit_verification(terminal(injected))
    assert "research_input_ref" not in VerificationSubmissionInput.model_fields
    accepted = checker.submit_verification(terminal(payload()))
    assert "research_input_ref" not in accepted.model_dump()
    with pytest.raises(ValidationError):
        VerificationSubmission(**(accepted.model_dump() | {"research_input_ref": research_ref(3)}))


def test_same_persisted_content_v3_v4_has_distinct_result_identity_without_finalization_io(tmp_path):
    store = ArtifactStore(tmp_path / "test")
    package = package_data()
    assert [store.save("research", package) for _ in range(4)] == [1, 2, 3, 4]
    results = []
    for version in (3, 4):
        loaded = store.load("research", version, ResearchPackage)
        assert loaded == package
        context = build_verification_context(loaded, "RF-target", research_input_ref=research_ref(version))
        tools, provider = TextTools(), SequenceProvider([])
        checker = FactChecker(context, tools=tools, provider=provider)
        # Mutating the caller's context cannot redirect the checker's detached provenance.
        context.research_input_ref = research_ref(99)
        selected, canonical, _ = read(checker, tools)
        proposal = submission(checker, selected)
        calls_before = list(tools.calls)
        result = checker.finalize_submission(proposal)
        assert checker.finalize_submission(proposal) == result
        assert tools.calls == calls_before and provider.requests == []
        assert result.research_input_ref == research_ref(version)
        canonical_payload = result.model_dump(mode="json", exclude={"verification_id"})
        encoded = json.dumps(["verification-v1", canonical_payload], ensure_ascii=False,
                             sort_keys=True, separators=(",", ":")).encode("utf-8")
        assert result.verification_id == "V-" + hashlib.sha256(encoded).hexdigest()
        assert result.research_fact_id == package.facts[1].fact_id
        assert result.claim_snapshot == package.facts[1].claim
        evidence = result.verification_evidence[0]
        start, end = canonical.spans[selected["span_id"]]
        assert evidence.excerpt == canonical.text[start:end]
        assert VerificationResult.model_validate_json(result.model_dump_json()) == result
        results.append(result)
    assert results[0].verification_id != results[1].verification_id
    assert results[0].research_input_ref != results[1].research_input_ref
    for field in ("research_fact_id", "claim_snapshot", "status", "verification_evidence",
                  "contradiction_evidence", "unresolved_issues", "independence_note", "rationale"):
        assert getattr(results[0], field) == getattr(results[1], field)
