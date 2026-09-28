"""Offline bounded accepted-memory projection; canonical artifacts remain intact."""
import json
from copy import deepcopy

import pytest

from history_studio.models.project import ProjectConfig
from history_studio.models.research_package import ResearchPackage
from history_studio.research.context import INSTRUCTIONS, ContextLimitError, build_context, build_retrieval_context
from history_studio.research.knowledge import (
    KnowledgeQuery, LexicalKnowledgeRetriever, empty_knowledge, plan_overview, serialized_size,
)
from history_studio.research.spans import make_spans
from history_studio.storage import ArtifactStore
from test_research_agent import FakeProvider, FakeTools, ResearchAgent, calls, setup_run, settings


def package_with_facts(unrelated=0):
    source = dict(source_id="S1", title="Childhood records", url="https://example.org/childhood",
                  source_type="UNKNOWN", accessed_at="2026-01-01T00:00:00Z")
    canonical = make_spans("S1", "Childhood education records survive.")
    evidence = dict(source_id="S1", source_version=canonical.source_version,
                    span_id=next(iter(canonical.spans)), excerpt=canonical.text)
    facts = [dict(fact_id=f"F{i}", claim=f"Childhood education record {i} survives",
                  historical_time={"display": "Date uncertain"}, research_confidence=0.5,
                  evidence=[deepcopy(evidence)]) for i in range(2)]
    sources = [source]
    for i in range(unrelated):
        sid = f"M{i}"
        sources.append(source | dict(source_id=sid, title=f"Mineral specimen {i}", url=f"https://example.org/mineral/{i}"))
        facts.append(dict(fact_id=f"MF{i}", claim=f"Mineral specimen {i} contains quartz",
                          historical_time={"display": "Date uncertain"}, research_confidence=0.5,
                          evidence=[dict(source_id=sid, excerpt="Crystalline material. " * 30)]))
    return ResearchPackage(project_id="test", topic="Childhood education", research_scope="Childhood education",
        progress={"run_id": "run"}, facts=facts, sources=sources,
        plan={"gaps": [dict(gap_id="G1", question="Childhood education?", completion_criteria=["Identify schooling"],
                            status="INVESTIGATING")], "coverage_summary": "Research ongoing"})


def project_for(package):
    return ProjectConfig(project_id=package.project_id, topic=package.topic, research_scope=package.research_scope)


def decode(context):
    return json.loads(context[len(INSTRUCTIONS):])


def test_partial_view_counts_and_soft_consolidation_preserve_package():
    package = package_with_facts(6)
    for fact in package.facts:
        fact.evidence = deepcopy(package.facts[0].evidence)
    before = package.model_dump_json()
    class ThreeFacts:
        def retrieve(self, query, facts, sources, budget_chars):
            return {"facts": [{"fact_id": f.fact_id} for f in facts[:3]],
                    "evidence": {}, "sources": []}
    canonical = make_spans("S1", "Childhood education records survive.")
    for soft in (False, True):
        context = build_context(project_for(package), package, settings(min_sources=2), soft, {},
            {"read_source_ids": ["S1"], "latest_read_source": canonical.observation()},
            knowledge_retriever=ThreeFacts())
        state = decode(context)
        assert state["knowledge_view"] == {"is_partial": True, "selected_fact_count": 3}
        assert state["knowledge_totals"]["facts"] == 8
        assert len(context) <= settings().max_context_chars - settings().max_observation_chars
        if soft:
            summary = state["consolidation"]
            blockers = state[summary["blockers_from"]]
            assert blockers["missing_cited_sources"] == 1
            assert blockers["unresolved_critical_gaps"] == ["G1"]
            assert summary["current_read_source_id"] == "S1"
            assert summary["material_available"] is True
            assert "when appropriate" in summary["priority"]
            assert serialized_size(summary) < 350
        else:
            assert "consolidation" not in state
    assert package.model_dump_json() == before
    assert "Never infer global absence" in INSTRUCTIONS
    assert "No action is forced" in INSTRUCTIONS
    from history_studio.research.actions import action_contracts
    assert {"search_web", "read_source", "checkpoint_research"} <= action_contracts().keys()


def test_evidence_key_cannot_be_converted_into_source_identity():
    from history_studio.research.spans import resolve_selection
    from history_studio.research.diagnostics import ArtifactValidationError
    package = package_with_facts()
    package.sources[0].source_id = "SRC-542c8d22c6f8209b7b68"
    canonical = make_spans(package.sources[0].source_id, "Childhood education records survive.")
    for fact in package.facts:
        fact.evidence[0].source_id = canonical.source_id
        fact.evidence[0].source_version = canonical.source_version
        fact.evidence[0].span_id = next(iter(canonical.spans))
    state = decode(build_context(project_for(package), package, settings(), False, {}))
    key, evidence = next(iter(state["knowledge"]["evidence"].items()))
    assert key.startswith("E-")
    assert evidence["source_id"] == canonical.source_id
    assert evidence["span_id"] == next(iter(canonical.spans))
    invented = "SRC-" + key.removeprefix("E-")
    reads = {canonical.source_id: canonical}
    with pytest.raises(ArtifactValidationError) as caught:
        resolve_selection(invented, evidence["span_id"], reads, {}, ["span_id"],
                          known_source_ids={canonical.source_id})
    assert caught.value.issue.type == "unknown_source"
    assert caught.value.issue.span_selection["source_id"] == invented
    assert "evidence reference IDs" in caught.value.issue.message
    assert set(reads) == {canonical.source_id}
    with pytest.raises(ArtifactValidationError) as caught:
        resolve_selection(canonical.source_id, evidence["span_id"], {}, {}, ["span_id"],
                          known_source_ids={canonical.source_id})
    assert caught.value.issue.type == "source_not_read"
    assert resolve_selection(canonical.source_id, evidence["span_id"], reads, {}, [],
        known_source_ids={canonical.source_id}).excerpt == canonical.text
    assert "NEVER replace an E- prefix with SRC-" in INSTRUCTIONS


def test_facts_rank_before_linked_evidence_and_deduplicate_without_mutation():
    package = package_with_facts(8)
    # Lexically relevant evidence on an unrelated fact must not make that fact relevant.
    package.facts[-1].evidence[0].excerpt = "Childhood education records."
    before = package.model_dump_json()
    query = KnowledgeQuery(package.topic, package.research_scope, package.plan.gaps)
    result = LexicalKnowledgeRetriever().retrieve(query, package.facts, package.sources, 6000)
    assert [f["fact_id"] for f in result["facts"]] == ["F0", "F1"]
    assert len(result["evidence"]) == 1
    for fact in result["facts"]:
        original = next(f for f in package.facts if f.fact_id == fact["fact_id"])
        assert [result["evidence"][key] for key in fact["evidence_ids"]] == [
            e.model_dump(mode="json") for e in original.evidence]
        assert fact["claim"] == original.claim
    assert serialized_size(result) <= 6000
    assert package.model_dump_json() == before
    assert result == LexicalKnowledgeRetriever().retrieve(query, list(reversed(package.facts)),
                                                        list(reversed(package.sources)), 6000)


@pytest.mark.parametrize("changed", ["source_id", "source_version", "span_id", "excerpt", "locator"])
def test_dedup_preserves_all_canonical_distinctions(changed):
    package = package_with_facts()
    evidence = package.facts[1].evidence[0]
    if changed == "source_id":
        package.sources.append(package.sources[0].model_copy(update={"source_id": "S2", "url": "https://example.org/two"}))
        evidence.source_id = "S2"
    elif changed == "excerpt":
        evidence.excerpt += " Different text."
    elif changed == "locator":
        evidence.locator = "Different locator"
    else:
        setattr(evidence, changed, "different_identity")
    result = LexicalKnowledgeRetriever().retrieve(KnowledgeQuery(package.topic, package.research_scope,
        package.plan.gaps), package.facts, package.sources, 6000)
    assert len(result["facts"]) == len(result["evidence"]) == 2


def test_byte_budget_skips_oversized_whole_fact_without_orphan_evidence():
    package = package_with_facts()
    package.facts[0].research_notes = "Childhood " * 180
    query = KnowledgeQuery(package.topic, package.research_scope, package.plan.gaps)
    result = LexicalKnowledgeRetriever().retrieve(query, package.facts, package.sources, 1800)
    assert [f["fact_id"] for f in result["facts"]] == ["F1"]
    assert set(result["evidence"]) == set(result["facts"][0]["evidence_ids"])
    assert serialized_size(result) <= 1800


def test_large_package_scaling_and_persistence(tmp_path):
    lengths = []
    durable_sizes = []
    for count in (10, 100, 500):
        package = package_with_facts(count)
        before = package.model_dump_json()
        store = ArtifactStore(tmp_path / f"size-{count}")
        store.save("research", package)
        context = build_context(project_for(package), package, settings(), False, {})
        state = decode(context)
        lengths.append(len(context))
        durable_sizes.append(len(before))
        assert len(context) <= 15000
        assert {f["fact_id"] for f in state["knowledge"]["facts"]} == {"F0", "F1"}
        assert all(not f["fact_id"].startswith("MF") for f in state["knowledge"]["facts"])
        assert state["knowledge_totals"] == {"facts": count + 2, "sources": count + 1}
        assert package.model_dump_json() == before
        assert store.load_latest("research", ResearchPackage).model_dump_json() == before
    assert durable_sizes[-1] > 20 * 15000
    assert lengths[-1] - lengths[1] < 300
    assert durable_sizes[-1] > 4 * durable_sizes[1]
    print("scalability", list(zip((10, 100, 500), durable_sizes, lengths)))


def test_global_plan_keeps_all_goal_states_and_criteria_without_repeated_details():
    package = package_with_facts()
    goal = package.plan.gaps[0]
    from history_studio.models.research_package import ResearchGoal
    for index, status in enumerate(("OPEN", "COVERED", "RESEARCHED_UNRESOLVED")):
        package.plan.gaps.append(ResearchGoal(gap_id=f"G{index+2}", question=f"Question {index}", critical=False,
            status=status, fact_ids=["F0", "F1"], completion_criteria=["Criterion"],
            coverage_assessment=None if status == "OPEN" else dict(criteria=[dict(criterion="Criterion", addressed=True)],
                supporting_fact_ids=["F0"], unresolved_issues=["Conflicting dates"], rationale="Records examined")))
    overview = decode(build_context(project_for(package), package, settings(), False, {}))["plan"]
    assert len(overview["gaps"]) == 4
    for original, projected in zip(package.plan.gaps, overview["gaps"]):
        for key in ("gap_id", "question", "critical", "status", "completion_criteria"):
            assert projected[key] == getattr(original, key)
        assert "fact_ids" not in projected
        if original.coverage_assessment:
            assert projected["coverage_assessment"]["unresolved_issues"] == ["Conflicting dates"]
            assert projected["coverage_assessment"]["addressed"] == [True]
            assert "rationale" not in projected["coverage_assessment"]


def test_source_catalog_compact_bounded_and_linked_sources_available():
    package = package_with_facts(500)
    state = decode(build_context(project_for(package), package, settings(), False, {}))
    catalog = state["knowledge"]["sources"]
    assert 1 <= len(catalog) < len(package.sources)
    assert "S1" in {s["source_id"] for s in catalog}
    for source in catalog:
        assert set(source) == {"source_id", "title", "url"}


def test_feedback_deduplicates_coverage_but_preserves_validation_detail():
    package = package_with_facts()
    from history_studio.research.progress import completion_status
    feedback = dict(status="no_progress", consecutive_attempts=2,
                    coverage=completion_status(package, settings()), instruction="Find new evidence")
    before = deepcopy(feedback)
    state = decode(build_context(project_for(package), package, settings(), False, {}, previous_outcome=feedback))
    assert state["previous_checkpoint_outcome"] == {k:v for k,v in feedback.items() if k != "coverage"}
    assert feedback == before
    feedback = dict(status="validation_error", errors=[{"location": ["facts", 0], "message": "Correct evidence"}])
    state = decode(build_context(project_for(package), package, settings(), False, {}, previous_outcome=feedback))
    assert state["previous_checkpoint_outcome"] == feedback


def test_context_retriever_can_be_replaced_and_cannot_exceed_budget():
    package = package_with_facts()
    class AlternateRetriever:
        def retrieve(self, query, facts, sources, budget_chars):
            assert query.topic == package.topic
            assert query.goals == package.plan.gaps
            assert budget_chars > 0
            return empty_knowledge()
    state = decode(build_context(project_for(package), package, settings(), False, {}, knowledge_retriever=AlternateRetriever()))
    assert state["knowledge"] == empty_knowledge()
    class OversizedRetriever:
        def retrieve(self, query, facts, sources, budget_chars):
            return {"facts": ["x" * budget_chars], "evidence": {}, "sources": []}
    with pytest.raises(ContextLimitError, match="structured_context_limit"):
        build_context(project_for(package), package, settings(), False, {}, knowledge_retriever=OversizedRetriever())


def test_agent_accepts_alternate_retriever_without_workflow_changes(tmp_path):
    project, store = setup_run(tmp_path)
    class AlternateRetriever:
        def __init__(self): self.calls = 0
        def retrieve(self, query, facts, sources, budget_chars):
            self.calls += 1
            return empty_knowledge()
    engine = AlternateRetriever()
    result = ResearchAgent(FakeProvider(calls()), FakeTools(), settings(), knowledge_retriever=engine).run(project, store)
    assert result.progress.status == "COMPLETE" and engine.calls > 0


def test_knowledge_and_current_read_share_allowance_with_exact_source_spans():
    package = package_with_facts(100)
    read = make_spans("S1", "Childhood education records. " * 200)
    context, evidence = build_retrieval_context(project_for(package), package, settings(), False, {}, ["S1"], read)
    assert len(context) <= 15000
    assert decode(context)["knowledge"]["facts"]
    for span in evidence["latest_read_source"]["spans"]:
        start, end = read.spans[span["span_id"]]
        assert span["text"] == read.text[start:end]
    assert evidence["latest_read_source"]["source_version"] == read.source_version



def test_scope_topic_anchors_remain_when_active_goal_vocabulary_differs():
    package = package_with_facts()
    package.plan.gaps[0].question = "Which competing date is recorded?"
    package.plan.gaps[0].completion_criteria = ["Compare dated records"]
    query = KnowledgeQuery(package.topic, package.research_scope, package.plan.gaps)
    result = LexicalKnowledgeRetriever().retrieve(query, package.facts, package.sources, 6000)
    assert {f["fact_id"] for f in result["facts"]} == {"F0", "F1"}
