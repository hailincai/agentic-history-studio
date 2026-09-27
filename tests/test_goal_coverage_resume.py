"""Offline checkpoint/resume verification for goal coverage and legacy packages."""
from copy import deepcopy
import json

import pytest

from history_studio.models.research_package import ResearchPackage
from history_studio.research.agent import ResearchAgent
from history_studio.research.context import INSTRUCTIONS
from history_studio.research.progress import completion_status, progress_marks, progress_signals
from test_research_agent import FakeProvider, FakeTools, setup_run, settings, ledger, update as legacy_update
from test_research_progress import calls, update


@pytest.mark.parametrize("status", ["COVERED", "RESEARCHED_UNRESOLVED"])
def test_terminal_checkpoint_resume_preserves_assessment_and_no_progress_history(tmp_path, status):
    project, store = setup_run(tmp_path)
    checkpoint = update(complete=False, gap_status=status)
    first = ResearchAgent(FakeProvider(calls()[:2] + [checkpoint, deepcopy(checkpoint), RuntimeError("offline interruption")]),
                          FakeTools(), settings()).run(project, store)
    assert first.progress.status == "FAILED"
    expected = first.plan.model_dump(mode="json")
    before = ledger(store)
    assert before.consecutive_no_progress == 1
    loaded = store.load_latest("research", ResearchPackage)
    assert loaded.plan.model_dump(mode="json") == expected
    assert progress_signals(progress_marks(loaded), set(before.progress_seen)) == {}

    provider = FakeProvider([deepcopy(checkpoint)] * 3)
    resumed = ResearchAgent(provider, FakeTools(), settings()).run(project, store)
    context = json.loads(provider.contexts[0][len(INSTRUCTIONS):])
    assert context["plan"] == expected
    assert resumed.plan.model_dump(mode="json") == expected
    assert resumed.facts == first.facts
    assert resumed.progress.stop_reason == "no_progress_limit"
    assert provider.calls == 2
    after = ledger(store)
    assert after.consecutive_no_progress == 3
    assert after.progress_seen == before.progress_seen
    assert after.run_id == before.run_id
    assert after.estimated_total_cost_usd >= before.estimated_total_cost_usd
    stopped = FakeProvider([])
    ResearchAgent(stopped, FakeTools(), settings()).run(project, store)
    assert stopped.calls == 0


@pytest.mark.parametrize("status", ["COVERED", "RESEARCHED_UNRESOLVED"])
def test_new_linked_research_after_resume_advances_coverage(tmp_path, status):
    project, store = setup_run(tmp_path)
    initial = update(complete=False)
    initial.arguments["plan"]["gaps"].append(dict(gap_id="G-2", question="Which competing date is recorded?",
        status="INVESTIGATING", completion_criteria=["Compare dated records"], fact_ids=[]))
    first = ResearchAgent(FakeProvider(calls()[:2] + [initial, RuntimeError("offline interruption")]),
                          FakeTools(), settings()).run(project, store)
    before = ledger(store)
    proposal = update(complete=False, gap_status=status)
    goal = proposal.arguments["plan"]["gaps"][0]
    goal.update(gap_id="G-2", question="Which competing date is recorded?", fact_ids=["RF-2"])
    goal["coverage_assessment"]["supporting_fact_ids"] = ["RF-2"]
    proposal.arguments["plan"]["gaps"].insert(0, first.plan.gaps[0].model_dump(mode="json"))
    fact = proposal.arguments["facts"][0]
    fact.update(fact_id="RF-2", claim="A competing date is recorded")
    # A new fact needs an authorized current read, even though the first goal is carried forward.
    provider = FakeProvider([calls()[1], proposal, RuntimeError("offline stop after checkpoint")])
    resumed = ResearchAgent(provider, FakeTools(), settings()).run(project, store)
    assert resumed.plan.gaps[0] == first.plan.gaps[0]
    assert resumed.plan.gaps[1].status == status
    assert len(resumed.facts) == 2
    after = ledger(store)
    signals = after.last_checkpoint_outcome["progress"]
    assert signals["fact"] == 1 and signals["coverage"] >= 1
    assert after.consecutive_no_progress == 0
    assert set(before.progress_seen) <= set(after.progress_seen)
    assert store.load_latest("research", ResearchPackage).plan == resumed.plan


@pytest.mark.parametrize("stored_status", ["FAILED", "COMPLETE"])
def test_legacy_load_and_resume_do_not_invent_coverage_or_grant_completion(tmp_path, stored_status):
    project, store = setup_run(tmp_path)
    first = ResearchAgent(FakeProvider(calls()[:2] + [legacy_update(complete=False), RuntimeError("offline interruption")]),
                          FakeTools(), settings()).run(project, store)
    path = store.project_dir / "research" / f"research_v{store.list_versions('research')[-1]}.json"
    data = json.loads(path.read_text(encoding="utf-8"))
    for goal in data["plan"]["gaps"]:
        goal.pop("completion_criteria", None)
        goal.pop("coverage_assessment", None)
    data["progress"]["status"] = stored_status
    path.write_text(json.dumps(data), encoding="utf-8")
    original_bytes = path.read_bytes()
    loaded = store.load_latest("research", ResearchPackage)
    goal = loaded.plan.gaps[0]
    assert goal.status == "COVERED"
    assert goal.completion_criteria == [] and goal.coverage_assessment is None
    assert not completion_status(loaded, settings())["can_complete"]
    provider = FakeProvider([RuntimeError("offline stop")])
    runner = ResearchAgent(provider, FakeTools(), settings())
    if stored_status == "COMPLETE":
        with pytest.raises(ValueError, match="Completion requires"):
            runner.run(project, store)
        assert provider.calls == 0
    else:
        resumed = runner.run(project, store)
        state = json.loads(provider.contexts[0][len(INSTRUCTIONS):])
        assert state["plan"]["gaps"][0]["completion_criteria"] == []
        assert state["plan"]["gaps"][0]["coverage_assessment"] is None
        assert resumed.plan == loaded.plan
    assert path.read_bytes() == original_bytes

