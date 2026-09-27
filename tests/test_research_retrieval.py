import json
from copy import deepcopy

import pytest

from history_studio.models.research_package import ResearchPackage
from history_studio.research.agent import ResearchAgent
from history_studio.research.context import INSTRUCTIONS, ContextLimitError, build_retrieval_context
from history_studio.research.retrieval import retrieve_relevant_spans
from history_studio.research.source_store import SourceStore
from history_studio.research.spans import make_spans, resolve_selection
from test_research_agent import FakeProvider, FakeTools, SOURCE, TEXT, calls, update, setup_run, settings


def size(value):
    return len(json.dumps(value, ensure_ascii=False, separators=(",", ":")))


@pytest.mark.parametrize("question,irrelevant,relevant", [
    ("birth childhood", "Minerals crystallize under pressure. ", "Birth and childhood records survive. "),
    ("出生地及家庭", "礦石晶體結構與地質變化。", "出生地與家庭背景記載。"),
])
def test_retrieval_ranks_late_relevant_spans_and_preserves_identity(question, irrelevant, relevant):
    read = make_spans("SRC-test", irrelevant * 150 + relevant * 20, True)
    result = retrieve_relevant_spans([question], read, 1600)
    assert 0 < len(result["spans"]) < len(read.spans)
    assert relevant.strip() in result["spans"][0]["text"]
    assert result == retrieve_relevant_spans([question], read, 1600)
    assert size(result) <= 1600
    assert result["source_id"] == read.source_id
    assert result["source_version"] == read.source_version
    assert result["truncated"] is True and result["retrieved_subset"] is True
    for span in result["spans"]:
        start, end = read.spans[span["span_id"]]
        assert span["text"] == read.text[start:end]
        evidence = resolve_selection(read.source_id, span["span_id"], {read.source_id: read},
            {key: (read.source_id, read.source_version) for key in read.spans}, [])
        assert evidence.excerpt == span["text"]


def test_ranking_ties_follow_offsets_and_skip_large_span_that_does_not_fit():
    # A punctuation-free first span is exactly 600 characters; the second is short.
    read = make_spans("SRC-test", "birth " + "x" * 594 + " childhood.")
    result = retrieve_relevant_spans(["birth childhood"], read, 450)
    assert len(result["spans"]) == 1
    assert result["spans"][0]["text"] == "childhood."
    full = retrieve_relevant_spans(["birth childhood"], read, 2000)
    assert [s["span_id"] for s in full["spans"]] == list(read.spans)
    assert not full["retrieved_subset"]


def test_unicode_ranking_does_not_normalize_canonical_text():
    read = make_spans("SRC-test", "ＣＨＩＬＤＨＯＯＤ Café records.")
    result = retrieve_relevant_spans(["childhood Cafe\u0301"], read, 1000)
    assert result["spans"][0]["text"] == read.text


def test_source_store_is_durable_versioned_and_not_a_second_representation(tmp_path):
    read = make_spans(SOURCE.source_id, TEXT * 50, True)
    store = SourceStore(tmp_path / "sources")
    store.put(read)
    store.put(read)
    changed = make_spans(SOURCE.source_id, TEXT + " Additional record.")
    store.put(changed)
    reopened = SourceStore(tmp_path / "sources")
    assert reopened.get(read.source_id, read.source_version) == read
    assert reopened.get(changed.source_id, changed.source_version) == changed
    assert len(list((tmp_path / "sources").rglob("*.json"))) == 2
    with pytest.raises(ValueError):
        reopened.get("../escape", read.source_version)
    path = tmp_path / "sources" / read.source_id / (read.source_version + ".json")
    data = json.loads(path.read_text(encoding="utf-8"))
    data["text"] += "tampered"
    path.write_text(json.dumps(data), encoding="utf-8")
    with pytest.raises(ValueError, match="mismatch"):
        reopened.get(read.source_id, read.source_version)


def context_fixture(tmp_path):
    project, _ = setup_run(tmp_path)
    proposal = update(complete=False, gap_status="OPEN", new_fact=False)
    proposal.arguments["plan"]["gaps"][0]["question"] = "What is documented?"
    proposal.arguments["plan"]["gaps"][0]["completion_criteria"] = ["Investigate childhood"]
    package = ResearchPackage(project_id=project.project_id, topic=project.topic,
        plan=proposal.arguments["plan"], sources=[SOURCE], progress={"run_id": "run"})
    return project, package


def test_actual_context_allowance_includes_metadata_and_criteria_relevance(tmp_path):
    project, package = context_fixture(tmp_path)
    read = make_spans(SOURCE.source_id, "Unrelated minerals crystallize. " * 100 + "Childhood records exist. " * 150)
    limits = settings(max_context_chars=10000, max_observation_chars=3000)
    context, evidence = build_retrieval_context(project, package, limits, False, {}, [SOURCE.source_id], read)
    assert len(context) <= 7000 - 64
    visible = evidence["latest_read_source"]
    assert 0 < len(visible["spans"]) < len(read.spans)
    assert "Childhood" in visible["spans"][0]["text"]
    assert json.loads(context[len(INSTRUCTIONS):])["evidence_context"] == evidence
    assert len(read.text) > 6000  # Complete canonical source remains intact.


@pytest.mark.parametrize("budget,text", [(150, "Childhood records."), (2000, "Minerals crystallize.")])
def test_no_fitting_useful_span_fails_deterministically(tmp_path, budget, text):
    project, package = context_fixture(tmp_path)
    read = make_spans(SOURCE.source_id, text)
    with pytest.raises(ContextLimitError, match="structured_context_limit"):
        build_retrieval_context(project, package, settings(), False, {}, [SOURCE.source_id], read,
                                observation_span_budget=budget)


def test_small_relevant_source_returns_all_original_spans():
    read = make_spans(SOURCE.source_id, TEXT)
    selected = retrieve_relevant_spans(["subject birth"], read, 2000)
    assert selected["spans"] == read.observation()["spans"]
    assert not selected["retrieved_subset"]


def test_large_read_correction_selection_persistence_and_iteration_clear(tmp_path):
    project, store = setup_run(tmp_path)
    text = "Minerals crystallize under pressure. " * 140 + TEXT
    canonical = make_spans(SOURCE.source_id, text)
    class LargeTools(FakeTools):
        def read_source(self, source, max_chars):
            result = super().read_source(source, max_chars)
            result.text = text
            return result
    initial = update(complete=False, gap_status="OPEN", new_fact=False)
    bad = update()
    bad.arguments["facts"][0]["claim"] = "one; two"
    provider = FakeProvider([initial, *calls()[:2], bad, update(complete=False), update(new_fact=False)])
    decide = provider.decide
    retained = []
    def inspect(context, observation, max_output_tokens):
        index = provider.calls
        state = json.loads(context[len(INSTRUCTIONS):])
        if index in (3, 4):
            page = state["evidence_context"]["latest_read_source"]
            retained.append(deepcopy(page))
            assert len(page["spans"]) < len(canonical.spans)
            assert page["source_version"] == canonical.source_version
            assert page["spans"][0]["span_id"] != next(iter(canonical.spans))
            if index == 3:
                assert json.loads(observation[1])["spans"] == page["spans"]
                assert SourceStore(store.project_dir / ".runtime" / "sources").get(
                    SOURCE.source_id, canonical.source_version) == canonical
            else:
                assert json.loads(observation[1])["status"] == "validation_error"
        if index == 5:
            assert state["evidence_context"] == {"read_source_ids": [], "latest_read_source": None}
            assert state["iteration"] == 3
        reply = decide(context, observation, max_output_tokens)
        if index in (3, 4):
            reply.call.arguments["facts"][0]["evidence"][0]["span_id"] = page["spans"][0]["span_id"]
        return reply
    provider.decide = inspect
    package = ResearchAgent(provider, LargeTools(), settings()).run(project, store)
    assert package.progress.status == "COMPLETE"
    assert retained[0] == retained[1]
    assert package.facts[0].evidence[0].excerpt == retained[0]["spans"][0]["text"]
    assert store.load_latest("research", ResearchPackage).facts[0].evidence == package.facts[0].evidence
    assert all(len(c) <= 15000 for c in provider.contexts)
    assert canonical.text not in package.model_dump_json()
    assert len(package.facts[0].evidence[0].excerpt) <= 600



def test_only_active_goal_text_is_used_and_retriever_is_replaceable(tmp_path):
    project, package = context_fixture(tmp_path)
    covered = update().arguments["plan"]["gaps"][0]
    # The goal object alone is sufficient here; package validity is tested elsewhere.
    from history_studio.models.research_package import ResearchGoal
    package.plan.gaps.append(ResearchGoal(**(covered | {"gap_id": "G2"})))
    captured = []
    def retriever(questions, source, budget):
        captured.extend(questions)
        return retrieve_relevant_spans(questions, source, budget)
    read = make_spans(SOURCE.source_id, "Childhood records exist.")
    build_retrieval_context(project, package, settings(), False, {}, [SOURCE.source_id], read,
                            retriever=retriever)
    assert captured == ["What is documented?", "Investigate childhood",
                        project.research_scope, project.topic]


def test_structured_budget_no_fit_even_when_observation_has_room(tmp_path):
    project, package = context_fixture(tmp_path)
    read = make_spans(SOURCE.source_id, "Childhood records exist. " * 20)
    baseline, _ = build_retrieval_context(project, package, settings(), False, {}, [SOURCE.source_id], None)
    limits = settings(max_context_chars=len(baseline) + 9000 + 100)
    with pytest.raises(ContextLimitError, match="structured_context_limit"):
        build_retrieval_context(project, package, limits, False, {}, [SOURCE.source_id], read,
                                observation_span_budget=9000)



def test_active_goal_with_no_literal_overlap_retains_scope_topic_anchors(tmp_path):
    project, package = context_fixture(tmp_path)
    goal = package.plan.gaps[0]
    goal.question = "Which competing date is recorded?"
    goal.completion_criteria = ["Compare dated records"]
    read = make_spans(SOURCE.source_id, TEXT)
    assert retrieve_relevant_spans([goal.question, *goal.completion_criteria], read, 9000)["spans"] == []
    context, evidence = build_retrieval_context(project, package, settings(), False, {},
                                               [SOURCE.source_id], read)
    assert evidence["latest_read_source"]["spans"] == read.observation()["spans"]
    assert evidence["latest_read_source"]["source_version"] == read.source_version
    assert len(context) <= 15000
