"""Canonical verification provenance, independent of Agent semantic judgment and network I/O."""
import json

import pytest
from pydantic import ValidationError

from history_studio.models.verification import VerificationEvidence, VerificationResult
from history_studio.research.boundaries import ToolObservation
from history_studio.research.diagnostics import ArtifactValidationError
from history_studio.research.spans import make_spans
from history_studio.verification import FactChecker
from history_studio.verification.submission import VerificationSubmission
from test_fact_checker_decision import input_context
from test_fact_checker_dispatch import RecordingTools, native
from test_fact_checker_investigation import SequenceProvider, decision
from test_verification_submission import payload, terminal


class TextTools(RecordingTools):
    def __init__(self, text="A dated record supports the claim. 中文记载。", truncated=False):
        super().__init__()
        self.text = text
        self.truncated = truncated

    def read_source(self, source, max_chars):
        self.calls.append(("read", source.source_id, max_chars))
        self.read_result = ToolObservation(kind="source", source_id=source.source_id,
            sources=[source.model_copy(deep=True)], text=self.text, truncated=self.truncated)
        return self.read_result


def setup(text="A dated record supports the claim. 中文记载。", truncated=False):
    tools = TextTools(text, truncated)
    provider = SequenceProvider([])
    return FactChecker(input_context(), tools=tools, provider=provider), tools, provider


def read(checker, tools, source_id="SRC-A", max_chars=8000):
    observation = checker.execute_tool_call(native("read_source", json.dumps(dict(source_id=source_id))), max_chars=max_chars)
    canonical = make_spans(source_id, tools.text[:max_chars], tools.truncated or len(tools.text) > max_chars)
    return dict(source_id=source_id, span_id=next(iter(canonical.spans))), canonical, observation


def submission(checker, selection, status="VERIFIED", contradict=None):
    data = payload(status)
    if data["verification_evidence"]:
        data["verification_evidence"] = [selection]
    if data["contradiction_evidence"]:
        data["contradiction_evidence"] = [contradict or selection]
    return checker.submit_verification(terminal(data))


@pytest.mark.parametrize("status", ["VERIFIED", "PARTIALLY_VERIFIED", "DISPUTED", "REJECTED", "UNVERIFIED"])
def test_all_statuses_materialize_exact_canonical_roles_without_io_or_mutation(status):
    checker, tools, provider = setup()
    selected, canonical, observation = read(checker, tools)
    proposal = submission(checker, selected, status)
    before = proposal.model_dump_json()
    calls_before = list(tools.calls)
    result = checker.finalize_submission(proposal)
    assert isinstance(result, VerificationResult) and result.status.value == status
    assert result.research_fact_id == input_context().target_fact.fact_id
    assert result.claim_snapshot == input_context().target_fact.claim
    for evidence in result.verification_evidence + result.contradiction_evidence:
        assert isinstance(evidence, VerificationEvidence) and evidence is not observation
        assert evidence.source_version == canonical.source_version and evidence.span_id == selected["span_id"]
        start, end = canonical.spans[evidence.span_id]
        assert evidence.excerpt == canonical.text[start:end]
        assert evidence.locator is None
    assert proposal.model_dump_json() == before
    assert tools.calls == calls_before and provider.requests == []
    assert result.verification_id.startswith("V-") and len(result.verification_id) == 66
    assert checker.finalize_submission(proposal).model_dump_json() == result.model_dump_json()


@pytest.mark.parametrize("which", ["original", "discovered", "unknown"])
def test_known_metadata_and_search_snippets_never_authorize_evidence(which):
    checker, tools, provider = setup()
    if which == "discovered":
        tools.search_result.text = "A tempting search snippet"
        checker.execute_tool_call(native())
    selected = dict(source_id={"original": "SRC-A", "discovered": "SRC-new", "unknown": "SRC-unknown"}[which],
                    span_id="SPAN-invented")
    # The finalization API must independently reject even directly constructed proposals.
    data = payload("VERIFIED")
    data["verification_evidence"] = [selected]
    proposal = VerificationSubmission(**data, research_fact_id=input_context().target_fact.fact_id,
                                      claim_snapshot=input_context().target_fact.claim)
    with pytest.raises(ArtifactValidationError) as caught:
        checker.finalize_submission(proposal)
    assert caught.value.issue.type == ("unknown_source" if which == "unknown" else "source_not_read")
    assert proposal.status.value == "VERIFIED" and provider.requests == []


@pytest.mark.parametrize("failure", ["invented", "other_source", "stale"])
def test_span_identity_attacks_fail_deterministically(failure):
    checker, tools, provider = setup()
    selected, _, _ = read(checker, tools)
    if failure == "invented":
        selected["span_id"] = "SPAN-invented"
    elif failure == "other_source":
        read(checker, tools, "SRC-B")
        selected["source_id"] = "SRC-B"
    else:
        tools.text = "A changed fetched representation."
        read(checker, tools)
    proposal = submission(checker, selected)
    with pytest.raises(ArtifactValidationError) as caught:
        checker.finalize_submission(proposal)
    assert caught.value.issue.type == {"invented": "span_not_found", "other_source": "span_source_mismatch",
                                "stale": "stale_source_span"}[failure]
    assert provider.requests == []


def test_support_and_contradiction_roles_and_duplicate_list_entries_preserved():
    checker, tools, _ = setup("Supporting record.")
    support, _, _ = read(checker, tools)
    tools.text = "Contradicting record."
    conflict, _, _ = read(checker, tools, "SRC-B")
    data = payload("DISPUTED")
    data.update(verification_evidence=[support, support], contradiction_evidence=[conflict])
    result = checker.finalize_submission(checker.submit_verification(terminal(data)))
    assert [e.excerpt for e in result.verification_evidence] == ["Supporting record."] * 2
    assert [e.excerpt for e in result.contradiction_evidence] == ["Contradicting record."]


def test_observation_mutation_does_not_replace_canonical_material():
    checker, tools, _ = setup()
    selected, canonical, observation = read(checker, tools)
    observation.text = "fabricated observation override"
    result = checker.finalize_submission(submission(checker, selected))
    assert result.verification_evidence[0].excerpt == canonical.text


@pytest.mark.parametrize("field,value", [("research_fact_id", "RF-other"), ("claim_snapshot", "Replaced target")])
def test_finalization_rechecks_bound_target(field, value):
    checker, tools, _ = setup()
    selected, _, _ = read(checker, tools)
    proposal = submission(checker, selected).model_copy(update={field: value})
    with pytest.raises(ValueError, match="target fact"):
        checker.finalize_submission(proposal)


def test_mutated_submission_revalidated_and_canonical_registry_integrity_checked():
    checker, tools, _ = setup()
    selected, _, _ = read(checker, tools)
    proposal = submission(checker, selected)
    invalid = proposal.model_copy(update={"verification_evidence": []})
    with pytest.raises(ValidationError):
        checker.finalize_submission(invalid)
    checker._reads["SRC-A"].spans[selected["span_id"]] = (0, 1)
    with pytest.raises(ValueError, match="Canonical investigation material"):
        checker.finalize_submission(proposal)


def test_prior_investigation_does_not_authorize_next_invocation():
    checker, tools, provider = setup()
    selected, _, _ = read(checker, tools)
    proposal = submission(checker, selected)
    provider.replies = iter([decision(text="No action")])
    checker.investigate(max_steps=1)
    with pytest.raises(ArtifactValidationError) as caught:
        checker.finalize_submission(proposal)
    assert caught.value.issue.type == "source_not_read"


def test_unverified_with_no_evidence_needs_no_read_and_does_not_invent_evidence():
    checker, tools, provider = setup()
    result = checker.finalize_submission(checker.submit_verification(terminal(payload())))
    assert result.status.value == "UNVERIFIED"
    assert result.verification_evidence == result.contradiction_evidence == []
    assert tools.calls == [] and provider.requests == []


def test_read_metadata_mismatch_cannot_authorize_source():
    checker, tools, _ = setup()
    original = tools.read_source
    def mismatched(source, max_chars):
        result = original(source, max_chars)
        result.sources[0].url = "https://wrong.example/page"
        return result
    tools.read_source = mismatched
    with pytest.raises(ValueError, match="metadata must match"):
        checker.execute_tool_call(native("read_source", '{"source_id":"SRC-A"}'))
    assert checker._reads == {}


def test_changed_source_lookup_identity_cannot_relabel_accepted_read():
    checker, tools, _ = setup()
    selected, _, _ = read(checker, tools)
    proposal = submission(checker, selected)
    checker._sources["SRC-A"].url = "https://changed.example/page"
    with pytest.raises(ValueError, match="source identity changed"):
        checker.finalize_submission(proposal)


def test_long_truncated_source_is_exposed_as_whole_spans_within_existing_bound():
    checker, tools, provider = setup("中文文本。" * 1800)
    selected, canonical, observation = read(checker, tools)
    provider.replies = iter([decision(text="Reasoning only")])
    checker.decide_after_observation(observation)
    state = json.loads(provider.requests[0].input)
    exposed = state["current_observation"]
    assert exposed["text"] == "" and exposed["truncated"] is True
    assert exposed["spans"] == canonical.observation()["spans"]
    assert exposed["source_version"] == canonical.source_version
    assert "excerpts" not in exposed
    proposal = submission(checker, selected)
    assert checker.finalize_submission(proposal).verification_evidence[0].excerpt in canonical.text


def test_real_loop_selects_agent_visible_span_then_explicit_finalization_without_more_io():
    tools = TextTools("A historical record. 中文证据。")
    class SelectingProvider:
        def __init__(self):
            self.requests = []
        def decide(self, request):
            self.requests.append(request)
            if len(self.requests) == 1:
                return decision(native("read_source", '{"source_id":"SRC-A"}'))
            observed = json.loads(request.input)["current_observation"]
            data = payload("VERIFIED")
            data["verification_evidence"] = [dict(source_id=observed["source_id"], span_id=observed["spans"][0]["span_id"])]
            return decision(terminal(data))
    provider = SelectingProvider()
    checker = FactChecker(input_context(), tools=tools, provider=provider)
    outcome = checker.investigate()
    assert outcome.stop_reason.value == "SUBMITTED" and outcome.steps == 2
    assert isinstance(outcome.submission, VerificationSubmission)
    before = (len(provider.requests), len(tools.calls))
    result = checker.finalize_submission(outcome.submission)
    assert isinstance(result, VerificationResult) and result.verification_evidence[0].excerpt == tools.text
    assert (len(provider.requests), len(tools.calls)) == before == (2, 1)
