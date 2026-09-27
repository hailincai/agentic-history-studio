from copy import deepcopy

import pytest
import json

from history_studio.models.research_package import GapStatus
from history_studio.research.agent import ResearchAgent
from history_studio.research.context import INSTRUCTIONS
from history_studio.research.progress import completion_status, progress_marks, progress_signals
from history_studio.research.spans import make_spans
from history_studio.research.web_tools import source_reference
from test_research_agent import FakeProvider, FakeTools, SOURCE, TEXT, setup_run, settings, calls as legacy_calls, update as legacy_update, action, ledger


# This module's progress fixtures use the current coverage contract.
def update(**kwargs):
    call = legacy_update(**kwargs)
    goal = call.arguments["plan"]["gaps"][0]
    goal["completion_criteria"] = ["Compare dated records"]
    if goal["status"] in ("COVERED", "RESEARCHED_UNRESOLVED"):
        goal["coverage_assessment"] = dict(
            criteria=[dict(criterion="Compare dated records", addressed=True)],
            supporting_fact_ids=["RF-1"], unresolved_issues=["Records disagree"],
            rationale="Dated records examined.")
    return call


def calls(complete=True):
    return legacy_calls(complete)[:2] + [update(complete=complete)]


def initial_package(tmp_path):
    project, store = setup_run(tmp_path)
    package = ResearchAgent(FakeProvider(calls(complete=False)), FakeTools(),
                            settings(max_iterations=1)).run(project, store)
    return project, store, package


def test_unchanged_carry_forward_and_metadata_do_not_count_as_progress(tmp_path):
    _, _, package = initial_package(tmp_path)
    before = progress_marks(package)
    copied = package.model_copy(deep=True)
    copied.facts[0].research_notes = "Rewritten interpretation"
    copied.facts[0].research_confidence = 0.8
    copied.sources[0].title = "Different title"
    copied.sources[0].notes = "Read again"
    copied.plan.coverage_summary = "Reworded summary"
    assert progress_signals(progress_marks(copied), before) == {}
    assert progress_signals(progress_marks(package, {SOURCE.source_id: make_spans(SOURCE.source_id, TEXT)}), before) == {}


def test_new_facts_sources_read_evidence_and_coverage_count(tmp_path):
    _, _, package = initial_package(tmp_path)
    before = progress_marks(package)
    new = package.model_copy(deep=True)
    new.facts.append(new.facts[0].model_copy(update={"fact_id": "RF-2", "claim": "A competing claim", "dispute_group_id": "DG-1"}))
    assert progress_signals(progress_marks(new), before) == {"fact": 1}
    other = source_reference("https://example.org/other")
    new.sources.append(other)
    assert progress_signals(progress_marks(new), before)["source"] == 1
    read = make_spans(other.source_id, "New source evidence.")
    assert progress_signals(progress_marks(package, {other.source_id: read}), before) == {"read": 1}
    from history_studio.research.spans import resolve_selection
    new.facts[0].evidence.append(resolve_selection(other.source_id, next(iter(read.spans)),
        {other.source_id: read}, {s: (other.source_id, read.source_version) for s in read.spans}, []))
    assert progress_signals(progress_marks(new), before)["evidence"] == 1
    package.plan.gaps[0].status = GapStatus.OPEN
    baseline = progress_marks(package)
    package.plan.gaps[0].status = GapStatus.INVESTIGATING
    investigating = progress_marks(package)
    assert progress_signals(investigating, baseline) == {}
    package.plan.gaps[0].status = GapStatus.COVERED
    covered = progress_marks(package)
    assert progress_signals(covered, baseline | investigating) == {}
    package.plan.gaps[0].status = GapStatus.OPEN
    assert progress_signals(progress_marks(package), baseline | investigating | covered) == {}


def test_identical_checkpoints_stop_before_iteration_budget(tmp_path):
    project, store = setup_run(tmp_path)
    provider = FakeProvider(calls(complete=False) + [update(complete=False)] * 5)
    package = ResearchAgent(provider, FakeTools(), settings(max_iterations=5)).run(project, store)
    assert package.progress.stop_reason == "no_progress_limit"
    assert package.progress.iterations == 2
    assert provider.calls == 6
    assert ledger(store).consecutive_no_progress == 3
    # Initial, meaningful checkpoint, terminal condition; duplicates are not published.
    assert store.list_versions("research") == [1, 2, 3]
    assert json.loads(provider.observations[4][1])["status"] == "no_progress"
    assert json.loads(provider.observations[5][1])["consecutive_attempts"] == 2


def test_remaining_gaps_and_completion_shortfalls_are_visible(tmp_path):
    project, store = setup_run(tmp_path)
    pending = update(complete=False, gap_status="OPEN")
    provider = FakeProvider(calls()[:2] + [pending, pending, pending])
    package = ResearchAgent(provider, FakeTools(), settings(min_sources=2, max_no_progress_checkpoints=2)).run(project, store)
    context = json.loads(provider.contexts[3][len(INSTRUCTIONS):])
    assert context["coverage_status"]["unresolved_critical_gaps"] == ["G-1"]
    assert context["coverage_status"]["missing_cited_sources"] == 1
    assert context["previous_checkpoint_outcome"]["status"] == "checkpoint_accepted"
    feedback = json.loads(provider.observations[4][1])
    assert feedback["coverage"]["unresolved_gaps"] == ["G-1"]
    assert not feedback["coverage"]["can_complete"]
    assert package.progress.stop_reason == "no_progress_limit"


def test_agent_chooses_search_read_after_feedback_then_completes(tmp_path):
    project, store = setup_run(tmp_path)
    other = source_reference("https://example.org/second")
    class MoreSources(FakeTools):
        def search_web(self, query):
            result = super().search_web(query)
            if len(self.queries) > 1:
                result.sources = [other]
            return result
        def read_source(self, source, max_chars):
            result = super().read_source(source, max_chars)
            result.sources = [source]
            result.source_id = source.source_id
            return result
    final = update()
    final.arguments["facts"][0]["evidence"].append({"source_id": other.source_id,
        "span_id": next(iter(make_spans(other.source_id, TEXT).spans))})
    provider = FakeProvider(calls(complete=False) + [update(), action("search_web", query="independent evidence"),
        action("read_source", source_id=other.source_id), final])
    tools = MoreSources()
    package = ResearchAgent(provider, tools, settings(min_sources=2)).run(project, store)
    assert package.progress.status == "COMPLETE"
    feedback = json.loads(provider.observations[4][1])
    assert feedback["status"] == "no_progress"
    assert feedback["coverage"]["missing_cited_sources"] == 1
    assert len(tools.queries) == len(tools.reads) == 2
    assert ledger(store).consecutive_no_progress == 0
    assert completion_status(package, settings(min_sources=2))["can_complete"]


def test_repeated_complete_cannot_bypass_cited_source_requirement(tmp_path):
    project, store = setup_run(tmp_path)
    provider = FakeProvider(calls() + [update()] * 4)
    package = ResearchAgent(provider, FakeTools(), settings(min_sources=2)).run(project, store)
    assert package.progress.status == "LIMIT_REACHED"
    assert package.progress.stop_reason == "no_progress_limit"
    assert completion_status(package, settings(min_sources=2))["missing_cited_sources"] == 1


def test_resume_preserves_streak_and_does_not_credit_old_facts(tmp_path):
    project, store = setup_run(tmp_path)
    first = ResearchAgent(FakeProvider(calls(complete=False) + [update(complete=False), RuntimeError("offline")]),
                          FakeTools(), settings()).run(project, store)
    assert first.progress.status == "FAILED"
    before = ledger(store)
    assert before.consecutive_no_progress == 1
    provider = FakeProvider([update(complete=False)] * 3)
    package = ResearchAgent(provider, FakeTools(), settings()).run(project, store)
    assert package.progress.stop_reason == "no_progress_limit"
    assert provider.calls == 2
    assert ledger(store).progress_seen == before.progress_seen
    stopped = FakeProvider([])
    ResearchAgent(stopped, FakeTools(), settings()).run(project, store)
    assert stopped.calls == 0



def test_no_progress_feedback_preserves_current_read_spans(tmp_path):
    project, store = setup_run(tmp_path)
    provider = FakeProvider(calls(complete=False) + [action("read_source", source_id=SOURCE.source_id),
                                                   update(complete=False), update()])
    package = ResearchAgent(provider, FakeTools(), settings()).run(project, store)
    assert package.progress.status == "COMPLETE"
    feedback = json.loads(provider.observations[5][1])
    assert feedback["status"] == "no_progress"
    context = json.loads(provider.contexts[5][len(INSTRUCTIONS):])
    assert context["evidence_context"]["latest_read_source"]["spans"][0]["text"] == TEXT
    assert context["previous_checkpoint_outcome"]["status"] == "no_progress"


@pytest.mark.parametrize("status", [GapStatus.COVERED, GapStatus.RESEARCHED_UNRESOLVED])
def test_terminal_advancement_requires_new_linked_research(tmp_path, status):
    _, _, package = initial_package(tmp_path)
    package.plan.gaps[0].status = GapStatus.INVESTIGATING
    before = progress_marks(package)
    candidate = package.model_copy(deep=True)
    goal = candidate.plan.gaps[0]
    goal.status = status
    assert progress_signals(progress_marks(candidate), before) == {}
    candidate.facts.append(candidate.facts[0].model_copy(update={
        "fact_id": "RF-2", "claim": "A competing date is recorded"}))
    # Unrelated research still counts as research, but not this goal's coverage.
    assert progress_signals(progress_marks(candidate), before) == {"fact": 1}
    goal.coverage_assessment.supporting_fact_ids.append("RF-2")
    signals = progress_signals(progress_marks(candidate), before)
    assert signals == {"coverage": 1, "fact": 1}
    seen = before | progress_marks(candidate)
    for change in ("same", "rationale", "issues", "ordering", "terminal_switch"):
        copied = candidate.model_copy(deep=True)
        assessment = copied.plan.gaps[0].coverage_assessment
        if change == "rationale":
            assessment.rationale = "Reworded rationale"
        elif change == "issues":
            assessment.unresolved_issues = ["Reworded uncertainty"]
        elif change == "ordering":
            assessment.supporting_fact_ids.reverse()
            copied.facts.reverse()
        elif change == "terminal_switch":
            copied.plan.gaps[0].status = (GapStatus.COVERED if status == GapStatus.RESEARCHED_UNRESOLVED
                                         else GapStatus.RESEARCHED_UNRESOLVED)
        assert progress_signals(progress_marks(copied), seen) == {}


@pytest.mark.parametrize("status", ["COVERED", "RESEARCHED_UNRESOLVED"])
def test_unchanged_terminal_checkpoints_reach_no_progress_limit(tmp_path, status):
    project, store = setup_run(tmp_path)
    checkpoint = update(complete=False, gap_status=status)
    provider = FakeProvider(calls()[:2] + [deepcopy(checkpoint) for _ in range(5)])
    package = ResearchAgent(provider, FakeTools(), settings(max_iterations=5)).run(project, store)
    assert package.plan.gaps[0].status == status
    assert package.progress.stop_reason == "no_progress_limit"
    assert ledger(store).consecutive_no_progress == 3
    assert json.loads(provider.observations[4][1])["status"] == "no_progress"


def test_invalid_terminal_assessment_does_not_earn_coverage(tmp_path):
    _, _, package = initial_package(tmp_path)
    package.plan.gaps[0].status = GapStatus.OPEN
    before = progress_marks(package)
    package.facts.append(package.facts[0].model_copy(update={"fact_id": "RF-2", "claim": "New research"}))
    goal = package.plan.gaps[0]
    goal.status = GapStatus.RESEARCHED_UNRESOLVED
    goal.coverage_assessment.supporting_fact_ids = ["RF-2"]
    goal.coverage_assessment.unresolved_issues = []
    assert progress_signals(progress_marks(package), before) == {"fact": 1}


def test_investigating_and_terminal_advancement_with_new_evidence(tmp_path):
    _, _, package = initial_package(tmp_path)
    package.plan.gaps[0].status = GapStatus.OPEN
    before = progress_marks(package)
    package.plan.gaps[0].status = GapStatus.INVESTIGATING
    assert progress_signals(progress_marks(package), before) == {}
    package.facts[0].evidence.append(package.facts[0].evidence[0].model_copy(update={
        "excerpt": "A different source suggests 702.", "span_id": None, "source_version": None}))
    assert progress_signals(progress_marks(package), before) == {"coverage": 1, "evidence": 1}
    seen = before | progress_marks(package)
    package.plan.gaps[0].status = GapStatus.COVERED
    assert progress_signals(progress_marks(package), seen) == {}
    package.facts[0].evidence.append(package.facts[0].evidence[0].model_copy(update={
        "excerpt": TEXT, "span_id": None, "source_version": None}))
    assert progress_signals(progress_marks(package), seen) == {"coverage": 1, "evidence": 1}


def test_new_question_and_legacy_status_only_marks(tmp_path):
    _, _, package = initial_package(tmp_path)
    before = progress_marks(package)
    goal = package.plan.gaps[0].model_copy(update={"gap_id": "G-2", "question": "Which records survive?",
                                                 "status": GapStatus.OPEN})
    package.plan.gaps.append(goal)
    assert progress_signals(progress_marks(package), before) == {"question": 1}
    assert progress_signals({"coverage:legacy-status"}, set()) == {}
