import json
from pathlib import Path

import pytest

from history_studio.models import ProjectConfig
from history_studio.models.research_package import ResearchPackage, ResearchRunStatus
from history_studio.research.agent import ResearchAgent
from history_studio.research.boundaries import ModelReply, ToolCall, ToolObservation, Usage
from history_studio.research.config import ResearchSettings
from history_studio.research.context import ContextLimitError, build_context
from history_studio.research.usage import UsageLedger
from history_studio.research.web_tools import source_reference
from history_studio.research.spans import make_spans
from history_studio.storage import ArtifactStore
from history_studio.storage.artifact_store import write_json
from history_studio.workflow import RuntimeState, ProjectState as S

SOURCE = source_reference("https://example.org/history")
TEXT = "The subject was born in 701. A different source suggests 702."


def action(name: str, **arguments) -> ToolCall:
    return ToolCall(call_id="call1", name=name, arguments=arguments)


def update(*, complete: bool = True, quote: str = "The subject was born in 701.",
           gap_status: str = "COVERED", new_fact: bool = True) -> ToolCall:
    fact = dict(fact_id="RF-1", claim="The subject was born in 701.",
                historical_time={"display": "701", "start_year": 701, "precision": "YEAR"},
                evidence=[dict(source_id=SOURCE.source_id, span_id=next(iter(make_spans(SOURCE.source_id, TEXT).spans))
                               if quote in TEXT else "SPAN-invented")], research_confidence=0.6,
                research_notes="Candidate only; relevant to the requested early-life scope.")
    goal = dict(gap_id="G-1", question="When was the subject born?",
        critical=True, status=gap_status, fact_ids=["RF-1"] if new_fact or gap_status == "COVERED" else [],
        completion_criteria=["Identify evidence for the birth date"])
    if gap_status == "COVERED":
        goal["coverage_assessment"] = dict(
            criteria=[dict(criterion="Identify evidence for the birth date", addressed=True)],
            supporting_fact_ids=["RF-1"], unresolved_issues=[],
            rationale="The cited record supplies a candidate birth date.")
    return action("checkpoint_research", plan={"gaps": [goal],
        "coverage_summary": "Birth chronology investigated; candidate evidence collected."},
        facts=[fact] if new_fact else [], recommendation="COMPLETE" if complete else "CONTINUE")


class FakeProvider:
    def __init__(self, actions: list, cost: float = 0.001) -> None:
        self.actions = iter(actions)
        self.cost = cost
        self.contexts = []
        self.observations = []
        self.calls = 0

    def reserve_cost(self, context, observation, max_output_tokens):
        return self.cost

    def decide(self, context, observation, max_output_tokens):
        self.contexts.append(context)
        self.observations.append(observation)
        self.calls += 1
        choice = next(self.actions)
        if isinstance(choice, Exception):
            raise choice
        return ModelReply(call=choice, usage=Usage(model="fake", input_tokens=1, output_tokens=1,
                          estimated_model_cost_usd=self.cost, estimated_tool_cost_usd=0))


class FakeTools:
    def __init__(self) -> None:
        self.queries = []
        self.reads = []
        self.fail_read = False

    def search_reserve_cost(self, query):
        return 0.001

    def search_web(self, query):
        self.queries.append(query)
        return ToolObservation(kind="search", sources=[SOURCE])

    def read_source(self, source, max_chars):
        self.reads.append(source.source_id)
        if self.fail_read:
            raise RuntimeError("OPENAI_API_KEY=secret-do-not-log")
        return ToolObservation(kind="source", sources=[SOURCE], source_id=SOURCE.source_id, text=TEXT)


def setup_run(tmp_path: Path):
    project = ProjectConfig(project_id="test", topic="Historical subject", research_scope="early life")
    store = ArtifactStore(tmp_path / "test")
    write_json(store.project_dir / "project.json", project)
    write_json(store.project_dir / ".runtime/state.json", RuntimeState())
    return project, store


def settings(**changes) -> ResearchSettings:
    return ResearchSettings(**({"min_facts": 1, "min_sources": 1} | changes))


def calls(complete=True):
    return [action("search_web", query="model-chosen original chronology query"),
            action("read_source", source_id=SOURCE.source_id), update(complete=complete)]


def state(store):
    return RuntimeState.model_validate_json((store.project_dir / ".runtime/state.json").read_text())


def ledger(store):
    return UsageLedger.model_validate_json((store.project_dir / ".runtime/research_usage.json").read_text())


def test_autonomous_dispatch_completion_and_persistence(tmp_path: Path) -> None:
    project, store = setup_run(tmp_path)
    provider, tools = FakeProvider(calls()), FakeTools()
    package = ResearchAgent(provider, tools, settings()).run(project, store)
    assert tools.queries == ["model-chosen original chronology query"]
    assert tools.reads == [SOURCE.source_id]
    assert package.progress.status == ResearchRunStatus.COMPLETE
    assert state(store).current_state == S.RESEARCH_COMPLETE
    assert len(package.sources) == 1
    assert provider.observations[1][0].name == "search_web"
    assert provider.observations[2][0].name == "read_source"
    assert store.list_versions("research") == [1, 2]
    assert store.load_latest("research", ResearchPackage) == package
    assert ledger(store).search_calls == 1 and ledger(store).source_reads == 1
    assert "early life" in provider.contexts[0]


def test_each_iteration_checkpoints_and_completed_rerun_is_free(tmp_path: Path) -> None:
    project, store = setup_run(tmp_path)
    first = calls(complete=False)
    provider = FakeProvider(first + [update(new_fact=False)])
    package = ResearchAgent(provider, FakeTools(), settings()).run(project, store)
    assert package.progress.iterations == 2
    assert store.list_versions("research") == [1, 2, 3]
    assert store.load("research", 2, ResearchPackage).progress.status == ResearchRunStatus.RUNNING
    empty = FakeProvider([])
    assert ResearchAgent(empty, FakeTools(), settings()).run(project, store) == package
    assert empty.calls == 0


@pytest.mark.parametrize("limits, choices, reason", [
    ({"hard_budget_usd": 0, "soft_budget_usd": 0}, [], "hard_budget"),
    ({"max_searches": 0}, [action("search_web", query="query")], "search_limit"),
    ({"max_source_reads": 0}, calls(), "source_read_limit"),
    ({"max_iterations": 1}, calls(complete=False), "iteration_limit"),
    ({"max_turns_per_iteration": 1}, [action("search_web", query="query")], "turn_limit"),
])
def test_hard_limits_do_not_complete(tmp_path: Path, limits: dict, choices: list, reason: str) -> None:
    project, store = setup_run(tmp_path)
    tools = FakeTools()
    package = ResearchAgent(FakeProvider(choices), tools, settings(**limits)).run(project, store)
    assert package.progress.status == ResearchRunStatus.LIMIT_REACHED
    assert package.progress.stop_reason == reason
    assert state(store).current_state == S.RESEARCHING
    assert ledger(store).committed_budget_usd <= settings(**limits).hard_budget_usd
    if reason == "source_read_limit":
        assert tools.reads == []
    if reason == "search_limit":
        assert tools.queries == []


def test_soft_budget_signals_model_without_forcing_completion(tmp_path: Path) -> None:
    project, store = setup_run(tmp_path)
    provider = FakeProvider(calls())
    events = []
    package = ResearchAgent(provider, FakeTools(), settings(soft_budget_usd=0.001), events.append).run(project, store)
    assert package.progress.status == ResearchRunStatus.COMPLETE
    assert '"soft_budget_reached":true' in provider.contexts[-1]
    assert any("Soft budget" in event for event in events)


@pytest.mark.parametrize("changes", [{"min_facts": 2}, {"min_sources": 2}])
def test_complete_recommendation_cannot_bypass_minimums(tmp_path: Path, changes: dict) -> None:
    project, store = setup_run(tmp_path)
    package = ResearchAgent(FakeProvider(calls()), FakeTools(), settings(max_iterations=1, **changes)).run(project, store)
    assert package.progress.status == ResearchRunStatus.LIMIT_REACHED
    assert state(store).current_state == S.RESEARCHING


def test_critical_gap_blocks_completion(tmp_path: Path) -> None:
    project, store = setup_run(tmp_path)
    choices = calls()[:2] + [update(gap_status="INVESTIGATING")]
    package = ResearchAgent(FakeProvider(choices), FakeTools(), settings(max_iterations=1)).run(project, store)
    assert package.progress.status == ResearchRunStatus.LIMIT_REACHED
    assert package.plan.gaps[0].status == "INVESTIGATING"


@pytest.mark.parametrize("choice", [action("execute_shell", command="bad"), update(quote="Invented quotation")])
def test_unapproved_tools_and_fabricated_evidence_are_rejected(tmp_path: Path, choice: ToolCall) -> None:
    project, store = setup_run(tmp_path)
    package = ResearchAgent(FakeProvider(calls()[:2] + [choice]), FakeTools(),
                            settings(max_turns_per_iteration=3)).run(project, store)
    assert not package.facts
    if choice.name == "checkpoint_research":
        assert package.progress.status == ResearchRunStatus.LIMIT_REACHED
        assert package.progress.stop_reason == "turn_limit"
        assert state(store).current_state == S.RESEARCHING
    else:
        assert package.progress.status == ResearchRunStatus.FAILED
        assert state(store).failed_state == S.RESEARCHING
    assert state(store).last_successful_state == S.CREATED


def test_failure_and_resume_keep_prior_facts_budget_and_secrets_out(tmp_path: Path) -> None:
    project, store = setup_run(tmp_path)
    provider = FakeProvider(calls(complete=False) + [RuntimeError("secret-do-not-log")])
    events = []
    package = ResearchAgent(provider, FakeTools(), settings(), events.append).run(project, store)
    assert package.progress.status == ResearchRunStatus.FAILED
    assert len(package.facts) == 1
    original = (store.project_dir / "research/research_v2.json").read_bytes()
    spent = ledger(store).committed_budget_usd
    assert ledger(store).unknown_usage
    assert state(store).failed_state == S.RESEARCHING
    resumed_provider = FakeProvider([update(new_fact=False)])
    resumed = ResearchAgent(resumed_provider, FakeTools(), settings()).run(project, store)
    assert resumed.progress.status == ResearchRunStatus.COMPLETE
    assert resumed.progress.iterations == 3
    assert '"fact_id":"RF-1"' in resumed_provider.contexts[0]
    assert ledger(store).committed_budget_usd > spent
    assert (store.project_dir / "research/research_v2.json").read_bytes() == original
    for path in store.project_dir.rglob("*.json"):
        assert "secret-do-not-log" not in path.read_text(encoding="utf-8")
    assert "secret-do-not-log" not in str(events)


def test_resume_latest_valid_checkpoint(tmp_path: Path) -> None:
    project, store = setup_run(tmp_path)
    ResearchAgent(FakeProvider(calls(complete=False)), FakeTools(), settings(max_iterations=1)).run(project, store)
    (store.project_dir / "research/research_v4.json").write_text("{", encoding="utf-8")
    provider = FakeProvider([update(new_fact=False)])
    package = ResearchAgent(provider, FakeTools(), settings(max_iterations=3)).run(project, store)
    assert package.progress.status == ResearchRunStatus.COMPLETE
    assert store.list_versions("research")[-1] == 5


def test_lost_usage_ledger_refuses_budget_reset(tmp_path: Path) -> None:
    project, store = setup_run(tmp_path)
    ResearchAgent(FakeProvider(calls(complete=False)), FakeTools(), settings(max_iterations=1)).run(project, store)
    (store.project_dir / ".runtime/research_usage.json").unlink()
    with pytest.raises(ValueError, match="usage ledger"):
        ResearchAgent(FakeProvider([]), FakeTools(), settings()).run(project, store)


def test_context_is_bounded_and_pages_do_not_accumulate(tmp_path: Path) -> None:
    project, store = setup_run(tmp_path)
    provider = FakeProvider(calls(complete=False) + [update(new_fact=False)])
    package = ResearchAgent(provider, FakeTools(), settings()).run(project, store)
    assert all(len(context) <= settings().max_context_chars for context in provider.contexts)
    from history_studio.research.context import INSTRUCTIONS
    contexts = [json.loads(context[len(INSTRUCTIONS):]) for context in provider.contexts]
    assert contexts[2]["evidence_context"]["latest_read_source"]["spans"][0]["text"] == TEXT
    assert contexts[-1]["evidence_context"] == {"read_source_ids": [], "latest_read_source": None}
    assert all(len(context) <= settings().max_context_chars - settings().max_observation_chars
               for context in provider.contexts)
    with pytest.raises(ContextLimitError):
        build_context(project, package, settings(), False, {},
                      evidence_context={"latest_read_source": {"text": "x" * 24000}})
    assert provider.observations[-1] is None
    huge = project.model_copy(update={"topic": "x" * 30000})
    with pytest.raises(ContextLimitError):
        build_context(huge, package, settings(), False, {})


def test_tool_failure_is_resumable(tmp_path: Path) -> None:
    project, store = setup_run(tmp_path)
    tools = FakeTools(); tools.fail_read = True
    package = ResearchAgent(FakeProvider(calls()), tools, settings()).run(project, store)
    assert package.progress.status == ResearchRunStatus.FAILED
    assert state(store).failed_state == S.RESEARCHING
    assert ledger(store).source_reads == 1
    tools.fail_read = False
    resumed = ResearchAgent(FakeProvider(calls()[1:]), tools, settings()).run(project, store)
    assert resumed.progress.status == ResearchRunStatus.COMPLETE
    assert ledger(store).source_reads == 2


def test_project_budget_caps_research_budget(tmp_path: Path) -> None:
    project, store = setup_run(tmp_path)
    project.budget_usd = 0
    provider = FakeProvider([])
    package = ResearchAgent(provider, FakeTools(), settings()).run(project, store)
    assert package.progress.stop_reason == "hard_budget"
    assert provider.calls == 0


def test_agent_evolves_plan_and_keeps_competing_claims(tmp_path: Path) -> None:
    project, store = setup_run(tmp_path)
    initial_plan = action("checkpoint_research", plan={"gaps": [dict(gap_id="G-1",
        question="When was the subject born?", critical=True, status="OPEN", fact_ids=[])],
        "coverage_summary": "Chronology is an open question"}, facts=[], recommendation="CONTINUE")
    final = update()
    first_fact = final.arguments["facts"][0]
    first_fact["dispute_group_id"] = "DG-1"
    competing = dict(first_fact) | {"fact_id": "RF-2", "claim": "A different source suggests 702.",
        "historical_time": {"display": "702", "start_year": 702, "precision": "YEAR"},
        "evidence": [{"source_id": SOURCE.source_id, "span_id": next(iter(make_spans(SOURCE.source_id, TEXT).spans))}]}
    final.arguments["facts"].append(competing)
    final.arguments["plan"]["gaps"].append(dict(gap_id="G-2", question="Are there competing birth dates?",
        critical=True, status="COVERED", fact_ids=["RF-1", "RF-2"],
        completion_criteria=["Identify competing date claims"],
        coverage_assessment=dict(
            criteria=[dict(criterion="Identify competing date claims", addressed=True)],
            supporting_fact_ids=["RF-1", "RF-2"], unresolved_issues=[],
            rationale="Both competing date claims are preserved without adjudication.")))
    provider = FakeProvider([initial_plan] + calls()[:2] + [final])
    package = ResearchAgent(provider, FakeTools(), settings(min_facts=2)).run(project, store)
    assert package.progress.status == ResearchRunStatus.COMPLETE
    assert len(package.plan.gaps) == 2
    assert {f.dispute_group_id for f in package.facts} == {"DG-1"}
    assert len(package.sources) == 1
    assert store.load("research", 2, ResearchPackage).plan.gaps[0].status == "OPEN"


def test_repeated_source_search_is_deduplicated(tmp_path: Path) -> None:
    project, store = setup_run(tmp_path)
    choices = [action("search_web", query="first query"), action("search_web", query="second query")] + calls()[1:]
    package = ResearchAgent(FakeProvider(choices), FakeTools(), settings()).run(project, store)
    assert len(package.sources) == 1
    assert package.progress.status == ResearchRunStatus.COMPLETE
    assert ledger(store).search_calls == 2


def test_cannot_drop_a_critical_gap_to_complete(tmp_path: Path) -> None:
    project, store = setup_run(tmp_path)
    initial = update(complete=False, gap_status="OPEN", new_fact=False)
    final = update()
    final.arguments["plan"]["gaps"] = []
    package = ResearchAgent(FakeProvider([initial] + calls()[:2] + [final]), FakeTools(),
                            settings(max_turns_per_iteration=3)).run(project, store)
    assert package.progress.status == ResearchRunStatus.LIMIT_REACHED
    assert package.progress.stop_reason == "turn_limit"
    assert package.plan.gaps[0].status == "OPEN"


def test_crash_after_complete_package_before_workflow_is_reconciled(tmp_path: Path) -> None:
    project, store = setup_run(tmp_path)
    package = ResearchAgent(FakeProvider(calls()), FakeTools(), settings()).run(project, store)
    write_json(store.project_dir / ".runtime/state.json", RuntimeState(current_state=S.RESEARCHING,
        last_successful_state=S.CREATED), replace=True)
    provider = FakeProvider([])
    assert ResearchAgent(provider, FakeTools(), settings()).run(project, store) == package
    assert state(store).current_state == S.RESEARCH_COMPLETE
    assert provider.calls == 0


def test_active_lock_prevents_provider_calls(tmp_path: Path) -> None:
    project, store = setup_run(tmp_path)
    lock = store.project_dir / ".runtime/research.lock"
    lock.write_text("active")
    provider = FakeProvider([])
    with pytest.raises(ValueError, match="lock"):
        ResearchAgent(provider, FakeTools(), settings()).run(project, store)
    assert provider.calls == 0
    assert lock.exists()


def test_search_summary_cannot_be_promoted_to_fetched_evidence(tmp_path: Path) -> None:
    project, store = setup_run(tmp_path)
    tools = FakeTools()
    tools.search_web = lambda query: ToolObservation(kind="source", sources=[SOURCE],
        source_id=SOURCE.source_id, text=TEXT)
    package = ResearchAgent(FakeProvider(calls()), tools, settings()).run(project, store)
    assert package.progress.status == ResearchRunStatus.FAILED
    assert not package.facts


def test_failed_checkpoint_never_advances_workflow(tmp_path: Path, monkeypatch) -> None:
    project, store = setup_run(tmp_path)
    original_save = store.save
    count = 0
    def failing_save(artifact_type, artifact):
        nonlocal count
        count += 1
        if count > 1:
            raise OSError("disk full")
        return original_save(artifact_type, artifact)
    monkeypatch.setattr(store, "save", failing_save)
    with pytest.raises(OSError):
        ResearchAgent(FakeProvider(calls()), FakeTools(), settings()).run(project, store)
    assert store.list_versions("research") == [1]
    assert store.load("research", 1, ResearchPackage).facts == []
    assert state(store).current_state == S.FAILED
    assert state(store).last_successful_state == S.CREATED
    assert state(store).failed_state == S.RESEARCHING
    assert state(store).latest_error.startswith("research_artifact_persistence_failed")
