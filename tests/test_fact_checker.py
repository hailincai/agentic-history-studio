import json

import pytest
from pydantic import ValidationError

from history_studio.models import VerificationContext, build_verification_context
from history_studio.models.verification import VerificationEvidence, VerificationResult
from history_studio.verification import FactChecker
from history_studio.verification.context import INSTRUCTIONS, build_context
from test_verification_context import package_data


def decode(prepared):
    assert prepared.startswith(INSTRUCTIONS + "\n")
    return json.loads(prepared[len(INSTRUCTIONS):])


def test_prepare_one_claim_preserves_complete_original_input():
    package = package_data()
    context = build_verification_context(package, "RF-target")
    prepared = FactChecker(context).prepare()
    state = decode(prepared)
    assert state == context.model_dump(mode="json")
    assert state["target_fact"]["fact_id"] == "RF-target"
    assert state["target_fact"]["claim"] == package.facts[1].claim
    assert state["target_fact"]["historical_time"] == package.facts[1].historical_time.model_dump(mode="json")
    assert state["target_fact"]["research_confidence"] == 0.6
    assert state["target_fact"]["evidence"] == [e.model_dump(mode="json") for e in package.facts[1].evidence]
    assert [s["source_id"] for s in state["sources"]] == ["SRC-B", "SRC-A"]
    assert "RF-other" not in prepared
    assert "SRC-unrelated" not in prepared and "SRC-discovered" not in prepared
    assert "plan" not in state and "facts" not in state
    assert state["investigation_boundary"] == "TARGET_CLAIM_ONLY"
    assert state["original_evidence_role"] == "RESEARCH_INPUT_NOT_INDEPENDENT_VERIFICATION"
    assert state["whole_topic_research_allowed"] is False


@pytest.mark.parametrize("invalid", [lambda: package_data(), lambda: [], lambda: {}])
def test_agent_boundary_rejects_package_and_other_inputs(invalid):
    with pytest.raises(TypeError, match="one VerificationContext"):
        FactChecker(invalid())
    with pytest.raises(TypeError, match="one VerificationContext"):
        build_context(invalid())


def test_preparation_requires_only_context_and_never_executes_provider(monkeypatch):
    from openai.resources.responses import Responses
    def forbidden(*args, **kwargs):
        pytest.fail("Preparation must not execute a provider/model request")
    monkeypatch.setattr(Responses, "create", forbidden)
    context = build_verification_context(package_data(), "RF-target")
    detached = VerificationContext.model_validate_json(context.model_dump_json())
    checker = FactChecker(detached)
    assert decode(checker.prepare())["target_fact"]["fact_id"] == "RF-target"
    assert checker.provider is None and checker.tools is None
    assert all(not hasattr(checker, method) for method in ("run", "verify", "complete"))


def test_preparation_is_deterministic_detached_and_creates_no_verification_data():
    context = build_verification_context(package_data(), "RF-target")
    before = context.model_dump_json()
    checker = FactChecker(context)
    prepared = checker.prepare()
    assert checker.prepare() == prepared
    assert context.model_dump_json() == before
    assert not isinstance(prepared, VerificationResult)
    assert all(not isinstance(e, VerificationEvidence) for e in context.target_fact.evidence)
    assert "verification_evidence" not in decode(prepared)
    assert "status" not in decode(prepared)
    context.target_fact.evidence[0].locator = "Caller changed input after preparation"
    assert checker.prepare() == prepared


def test_nested_broken_provenance_is_rejected_before_agent_preparation():
    context = build_verification_context(package_data(), "RF-target")
    context.sources.clear()
    with pytest.raises(ValidationError, match="Missing source metadata"):
        FactChecker(context)


@pytest.mark.parametrize("guidance", [
    "Verify only the target ResearchFact",
    "do not broaden the task into researching the whole historical topic",
    "original\nresearch evidence is NOT independent verification evidence by itself",
    "seek independent evidence relevant to this specific claim",
    "contradict, narrow,\nor qualify the claim",
    "do not investigate only for confirmation",
    "Insufficient evidence is not REJECTED",
    "Agent owns historical semantic judgment",
    "Runtime owns provenance integrity",
])
def test_instructions_explain_semantics_without_relying_only_on_literal_names(guidance):
    assert " ".join(guidance.split()) in " ".join(INSTRUCTIONS.split())
