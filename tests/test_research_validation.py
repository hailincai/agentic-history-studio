import json
from pathlib import Path
from types import SimpleNamespace

import pytest
from pydantic import ValidationError

from history_studio.cli import main
from history_studio.models.research_package import ResearchPackage, ResearchRunStatus
from history_studio.research.actions import ResearchSelectionUpdate as ResearchUpdate
from history_studio.research.agent import ResearchAgent
from history_studio.research.boundaries import ToolObservation
from history_studio.research.diagnostics import ValidationDiagnostic, validation_diagnostic
from history_studio.research.web_tools import source_reference
from history_studio.storage import ArtifactStore
from history_studio.workflow import ProjectState as S
from test_research_agent import (
    FakeProvider, FakeTools, SOURCE, TEXT, action, calls, ledger, settings, setup_run, state, update,
)


def diagnostics(store: ArtifactStore) -> list[ValidationDiagnostic]:
    runtime_store = ArtifactStore(store.project_dir / ".runtime")
    return [runtime_store.load("diagnostics", version, ValidationDiagnostic)
            for version in runtime_store.list_versions("diagnostics")]


def malformed_fact(**changes):
    call = update()
    call.arguments["facts"][0].update(changes)
    return call


@pytest.mark.parametrize("changes, error_type, location, message", [
    ({"research_confidence": 1.5}, "less_than_equal", ["facts", 0, "research_confidence"], "upper bound"),
    ({"claim": "one; two"}, "value_error", ["facts", 0, "claim"], "one atomic claim"),
    ({"historical_time": {"display": "742–701", "start_year": 742, "end_year": 701}},
     "value_error", ["facts", 0, "historical_time"], "chronological"),
    ({"evidence": [{"source_id": SOURCE.source_id, "span_id": " "}]},
     "string_pattern_mismatch", ["facts", 0, "evidence", 0, "span_id"], "pattern"),
])
def test_malformed_fact_is_rejected_diagnosed_and_corrected(
    tmp_path: Path, changes: dict, error_type: str, location: list, message: str,
) -> None:
    project, store = setup_run(tmp_path)
    bad = malformed_fact(**changes)
    with pytest.raises(ValidationError):
        ResearchUpdate.model_validate(bad.arguments)
    provider = FakeProvider(calls()[:2] + [bad, update()])
    tools = FakeTools()
    events = []
    decide = provider.decide
    def inspect_retry(context, observation, max_output_tokens):
        if provider.calls == 3:
            # The failed proposal cannot mutate or persist partial accepted facts/plan.
            assert store.list_versions("research") == [1]
            assert store.load_latest("research", ResearchPackage).facts == []
            assert store.load_latest("research", ResearchPackage).plan.gaps == []
        return decide(context, observation, max_output_tokens)
    provider.decide = inspect_retry
    package = ResearchAgent(provider, tools, settings(), events.append).run(project, store)
    assert package.progress.status == ResearchRunStatus.COMPLETE
    assert package.progress.iterations == 1
    assert package.facts[0].research_confidence == 0.6
    assert tools.reads == [SOURCE.source_id]  # Current passages survive the rejection.
    diagnostic, = diagnostics(store)
    assert diagnostic.error_class == "ValidationError"
    assert diagnostic.iteration == 1 and diagnostic.turn == 3
    assert diagnostic.recoverable
    issue = diagnostic.errors[0]
    assert (issue.type, issue.location) == (error_type, location)
    assert message in issue.message
    retry_call, output = provider.observations[3]
    assert retry_call == bad
    assert json.loads(output)["status"] == "validation_error"
    assert json.loads(output)["errors"][0] == issue.model_dump()
    assert any(error_type in line and message in line for line in events)
    assert store.list_versions("research") == [1, 2]
    assert state(store).current_state == S.RESEARCH_COMPLETE


def test_package_reference_validation_can_self_correct(tmp_path: Path) -> None:
    project, store = setup_run(tmp_path)
    bad = update()
    bad.arguments["plan"]["gaps"][0]["fact_ids"] = ["unknown_fact"]
    # The proposal shape is valid; the merged package's referential validation rejects it.
    ResearchUpdate.model_validate(bad.arguments)
    package = ResearchAgent(FakeProvider(calls()[:2] + [bad, update()]), FakeTools(), settings()).run(project, store)
    assert package.progress.status == ResearchRunStatus.COMPLETE
    diagnostic, = diagnostics(store)
    assert diagnostic.errors[0].type == "value_error"
    assert diagnostic.errors[0].location == []
    assert diagnostic.errors[0].message == "Gap references an unknown fact"
    assert package.plan.gaps[0].fact_ids == ["RF-1"]


def test_fabricated_evidence_is_not_accepted_during_correction(tmp_path: Path) -> None:
    project, store = setup_run(tmp_path)
    provider = FakeProvider(calls()[:2] + [update(quote="Invented quotation"), update()])
    package = ResearchAgent(provider, FakeTools(), settings()).run(project, store)
    assert package.progress.status == ResearchRunStatus.COMPLETE
    diagnostic, = diagnostics(store)
    assert diagnostic.error_class == "ArtifactValidationError"
    assert diagnostic.errors[0].type == "span_not_found"
    assert diagnostic.errors[0].location == ["facts", 0, "evidence", 0, "span_id"]
    for version in store.list_versions("research"):
        assert "Invented quotation" not in store.load("research", version, ResearchPackage).model_dump_json()


def test_missing_checkpoint_field_is_diagnosable(tmp_path: Path) -> None:
    project, store = setup_run(tmp_path)
    provider = FakeProvider([action("checkpoint_research", recommendation="CONTINUE")])
    package = ResearchAgent(provider, FakeTools(), settings(max_turns_per_iteration=1)).run(project, store)
    assert package.progress.status == ResearchRunStatus.LIMIT_REACHED
    issue = diagnostics(store)[0].errors[0]
    assert issue.type == "missing" and issue.location == ["plan"]
    assert issue.message == "Field required"


def test_repeated_invalid_proposals_stop_at_turn_limit(tmp_path: Path) -> None:
    project, store = setup_run(tmp_path)
    provider = FakeProvider([malformed_fact(research_confidence=2)] * 4 + [update()])
    package = ResearchAgent(provider, FakeTools(), settings(max_turns_per_iteration=4)).run(project, store)
    assert package.progress.status == ResearchRunStatus.LIMIT_REACHED
    assert package.progress.stop_reason == "turn_limit"
    assert provider.calls == 4
    assert len(diagnostics(store)) == 4
    assert package.facts == []
    assert state(store).current_state == S.RESEARCHING
    assert ledger(store).model_calls == 4


def test_correction_cannot_bypass_hard_budget(tmp_path: Path) -> None:
    project, store = setup_run(tmp_path)
    provider = FakeProvider(calls()[:2] + [malformed_fact(research_confidence=2), update()])
    package = ResearchAgent(provider, FakeTools(), settings(soft_budget_usd=0.002, hard_budget_usd=0.004)).run(project, store)
    assert provider.calls == 3  # No paid correction request fits after the rejected proposal.
    assert package.progress.status == ResearchRunStatus.LIMIT_REACHED
    assert package.progress.stop_reason == "hard_budget"
    assert ledger(store).committed_budget_usd == pytest.approx(0.004)
    assert package.facts == []
    assert len(diagnostics(store)) == 1


def test_corrected_continue_cannot_reset_iteration_limit(tmp_path: Path) -> None:
    project, store = setup_run(tmp_path)
    provider = FakeProvider(calls()[:2] + [malformed_fact(research_confidence=2), update(complete=False), update()])
    package = ResearchAgent(provider, FakeTools(), settings(max_iterations=1)).run(project, store)
    assert package.progress.status == ResearchRunStatus.LIMIT_REACHED
    assert package.progress.stop_reason == "iteration_limit"
    assert provider.calls == 4
    assert package.progress.iterations == 1
    assert len(package.facts) == 1
    assert len(diagnostics(store)) == 1


def test_oversized_rejected_call_still_obeys_context_limit(tmp_path: Path) -> None:
    project, store = setup_run(tmp_path)
    provider = FakeProvider([malformed_fact(research_notes="x" * 12000), update()])
    package = ResearchAgent(provider, FakeTools(), settings()).run(project, store)
    assert package.progress.stop_reason == "observation_context_limit"
    assert provider.calls == 1
    assert diagnostics(store)[0].errors[0].type == "string_too_long"


def test_diagnostics_do_not_persist_input_unknown_fields_or_private_notes(tmp_path: Path, monkeypatch) -> None:
    project, store = setup_run(tmp_path)
    secret = "sk-test-secret-never-persist"
    private = "private-model-reasoning-never-persist"
    monkeypatch.setenv("OPENAI_API_KEY", secret)
    bad = malformed_fact(research_confidence=secret, research_notes=private)
    bad.arguments[secret] = private
    events = []
    package = ResearchAgent(FakeProvider([bad]), FakeTools(), settings(max_turns_per_iteration=1), events.append).run(project, store)
    assert package.facts == []
    diagnostic, = diagnostics(store)
    assert any(issue.type == "float_parsing" for issue in diagnostic.errors)
    assert any(issue.location == ["<unknown_field>"] for issue in diagnostic.errors)
    for path in store.project_dir.rglob("*.json"):
        text = path.read_text(encoding="utf-8")
        assert secret not in text and private not in text
    assert secret not in str(events) and private not in str(events)
    serialized = diagnostic.model_dump_json()
    assert '"input"' not in serialized and '"ctx"' not in serialized and '"url"' not in serialized


def test_unknown_validator_exception_text_is_not_logged() -> None:
    exc = ValidationError.from_exception_data("ResearchUpdate", [{
        "type": "value_error", "loc": ("facts", 0, "claim"), "input": "secret-input",
        "ctx": {"error": ValueError("secret-validator-private-reasoning")},
    }])
    diagnostic = validation_diagnostic(exc, run_id="run", iteration=1, turn=1, recoverable=True)
    assert diagnostic.errors[0].type == "value_error"
    assert diagnostic.errors[0].location == ["facts", 0, "claim"]
    assert "secret" not in diagnostic.model_dump_json()


def test_many_validation_errors_have_bounded_feedback(tmp_path: Path) -> None:
    project, store = setup_run(tmp_path)
    bad = action("checkpoint_research", **{f"unknown_{i}": "private-input" for i in range(40)})
    package = ResearchAgent(FakeProvider([bad]), FakeTools(), settings(max_turns_per_iteration=1)).run(project, store)
    diagnostic, = diagnostics(store)
    assert diagnostic.total_errors > 12
    assert len(diagnostic.errors) == 12
    assert "private-input" not in diagnostic.model_dump_json()
    assert package.progress.status == ResearchRunStatus.LIMIT_REACHED


def test_two_iteration_live_failure_shape_keeps_zero_facts_and_twelve_sources(tmp_path: Path) -> None:
    project, store = setup_run(tmp_path)
    batches = [[source_reference(f"https://example.org/{batch}/{i}") for i in range(6)] for batch in range(2)]
    tools = FakeTools()
    searches = iter(batches)
    tools.search_web = lambda query: ToolObservation(kind="search", sources=next(searches))
    tools.read_source = lambda source, max_chars: ToolObservation(kind="source", sources=[source],
        source_id=source.source_id, text=TEXT)
    plan_only = update(complete=False, gap_status="OPEN", new_fact=False)
    provider = FakeProvider([
        action("search_web", query="first query"), action("read_source", source_id=batches[0][0].source_id), plan_only,
        action("search_web", query="second query"), action("read_source", source_id=batches[1][0].source_id),
        malformed_fact(research_confidence=2),
    ])
    package = ResearchAgent(provider, tools, settings(max_turns_per_iteration=3)).run(project, store)
    assert store.list_versions("research") == [1, 2, 3]
    assert package.progress.iterations == 2
    assert ledger(store).search_calls == 2 and ledger(store).source_reads == 2
    assert len(package.sources) == 12 and package.facts == []
    diagnostic, = diagnostics(store)
    assert diagnostic.iteration == 2 and diagnostic.turn == 3
    assert diagnostic.errors[0].location == ["facts", 0, "research_confidence"]
    assert package.progress.status == ResearchRunStatus.LIMIT_REACHED
    assert state(store).last_successful_state == S.CREATED


def test_cli_shows_validation_details_live_and_in_later_status(tmp_path: Path, monkeypatch, capsys) -> None:
    project, store = setup_run(tmp_path)
    provider = FakeProvider(calls()[:2] + [malformed_fact(research_confidence=2)])
    tools = FakeTools()
    from history_studio.research.openai_provider import RunConfiguration
    config = RunConfiguration(research=settings(max_turns_per_iteration=3))
    config_path = tmp_path / "run.json"
    config_path.write_text(config.model_dump_json(), encoding="utf-8")
    monkeypatch.setattr("history_studio.research.openai_provider.create_client", lambda config: SimpleNamespace(close=lambda: None))
    monkeypatch.setattr("history_studio.research.openai_provider.OpenAIResearchProvider", lambda *args: provider)
    monkeypatch.setattr("history_studio.research.openai_provider.OpenAIWebTools", lambda *args: tools)
    assert main(["--projects-dir", str(tmp_path), "research", "test", "--config", str(config_path)]) == 2
    output = capsys.readouterr().out
    assert "facts.0.research_confidence [less_than_equal]" in output
    assert "upper bound" in output and "diagnostics_v1.json" in output
    assert main(["--projects-dir", str(tmp_path), "status", "test"]) == 0
    assert "facts.0.research_confidence [less_than_equal]" in capsys.readouterr().out


def test_unexpected_provider_failures_still_fail_without_raw_messages(tmp_path: Path) -> None:
    project, store = setup_run(tmp_path)
    events = []
    package = ResearchAgent(FakeProvider([RuntimeError("secret-provider-body")]), FakeTools(), settings(), events.append).run(project, store)
    assert package.progress.status == ResearchRunStatus.FAILED
    assert state(store).failed_state == S.RESEARCHING
    assert "secret-provider-body" not in str(events)
    from history_studio.research.diagnostics import RequestDiagnostic
    diagnostic = ArtifactStore(store.project_dir / ".runtime").load_latest("diagnostics", RequestDiagnostic)
    assert diagnostic.error_class == "RuntimeError"
    assert diagnostic.category == "other"
    assert "secret-provider-body" not in diagnostic.model_dump_json()


def test_persistence_validation_failure_is_diagnosed_but_not_retried(tmp_path: Path, monkeypatch) -> None:
    project, store = setup_run(tmp_path)
    provider = FakeProvider(calls())
    events = []
    agent = ResearchAgent(provider, FakeTools(), settings(), events.append)
    checkpoint = agent._checkpoint
    attempts = 0
    def fail_once(package, artifact_store):
        nonlocal attempts
        attempts += 1
        if attempts == 2:
            raise ValidationError.from_exception_data("ResearchPackage", [{
                "type": "missing", "loc": ("facts", 0, "claim"), "input": {"private": "secret"},
            }])
        checkpoint(package, artifact_store)
    monkeypatch.setattr(agent, "_checkpoint", fail_once)
    package = agent.run(project, store)
    assert provider.calls == 3
    assert package.progress.status == ResearchRunStatus.FAILED
    assert package.progress.stop_reason == "research_artifact_persistence_failed"
    assert package.facts == []
    assert state(store).failed_state == S.RESEARCHING
    diagnostic, = diagnostics(store)
    assert diagnostic.recoverable is False
    assert diagnostic.operation == "artifact_persistence"
    assert diagnostic.errors[0].location == ["facts", 0, "claim"]
    assert diagnostic.errors[0].message == "Field required"
    assert "secret" not in diagnostic.model_dump_json() + str(events)



def test_visible_read_text_supports_exact_correction_without_rereading(tmp_path, monkeypatch):
    from history_studio.research.web_tools import read_public_source, normalize_text
    from history_studio.research.context import INSTRUCTIONS
    project, store = setup_run(tmp_path)
    monkeypatch.setattr("history_studio.research.web_tools._fetch", lambda url:
        (url, "<html><p>The subject was born\n in 701.</p><script>hidden text</script>"
              "<p>A different source suggests 702.</p></html>"))
    class PageTools(FakeTools):
        def read_source(self, source, max_chars):
            self.reads.append(source.source_id)
            return read_public_source(source, max_chars)
    provider = FakeProvider(calls()[:2] + [update(quote="Birth occurred in 701."), update()])
    decide = provider.decide
    def decide_from_visible_text(context, observation, max_output_tokens):
        if provider.calls == 2:
            page = json.loads(observation[1])
            assert page["kind"] == "source"
            assert page["source_id"] == SOURCE.source_id
            assert page["spans"][0]["text"] == TEXT
            assert "hidden text" not in page["spans"][0]["text"]
            assert "usage" not in page
        if provider.calls == 3:
            feedback = json.loads(observation[1])
            assert feedback["errors"][0]["type"] == "span_not_found"
            retained = json.loads(context[len(INSTRUCTIONS):])["evidence_context"]
            assert retained["read_source_ids"] == [SOURCE.source_id]
            page = retained["latest_read_source"]
            assert page["spans"][0]["text"] == TEXT
            reply = decide(context, observation, max_output_tokens)
            # Correct solely from the text/source identity actually visible on this turn.
            reply.call.arguments["facts"][0]["evidence"] = [{
                "source_id": page["source_id"], "span_id": page["spans"][0]["span_id"]}]
            return reply
        return decide(context, observation, max_output_tokens)
    provider.decide = decide_from_visible_text
    tools = PageTools()
    package = ResearchAgent(provider, tools, settings()).run(project, store)
    assert package.progress.status == ResearchRunStatus.COMPLETE
    assert len(diagnostics(store)) == 1
    assert tools.reads == [SOURCE.source_id]
    assert package.facts[0].evidence[0].excerpt == TEXT
    assert provider.calls == 4


@pytest.mark.parametrize("quote, accepted", [
    ("The subject was born in 701.", True),
    ("The subject was\n born   in 701.", True),
    ("Birth occurred in 701.", False),
    ("The subject was born in 702.", False),
])
def test_canonical_evidence_final_invariant(tmp_path, quote, accepted):
    from history_studio.research.actions import ResearchUpdate as CanonicalUpdate
    from history_studio.research.diagnostics import ArtifactValidationError
    project, store = setup_run(tmp_path)
    data = update().arguments
    data["facts"][0]["evidence"] = [{"source_id": SOURCE.source_id, "excerpt": quote}]
    package = ResearchPackage(project_id=project.project_id, topic=project.topic, sources=[SOURCE],
                              progress={"run_id": "test"})
    agent = ResearchAgent(FakeProvider([]), FakeTools(), settings())
    if accepted:
        assert agent._apply_update(package, CanonicalUpdate.model_validate(data), {SOURCE.source_id: TEXT}).facts
    else:
        with pytest.raises(ArtifactValidationError, match="exact excerpt"):
            agent._apply_update(package, CanonicalUpdate.model_validate(data), {SOURCE.source_id: TEXT})
