import json

import pytest

from history_studio.models.project import ProjectConfig
from history_studio.models.research_package import ResearchPackage
from history_studio.research.config import ResearchSettings
from history_studio.research.context import INSTRUCTIONS, ContextLimitError, build_context


@pytest.fixture
def context_inputs():
    project = ProjectConfig(project_id="test", topic="Historical trade",
                            research_scope="Trade routes and their changes over time")
    package = ResearchPackage(project_id="test", topic=project.topic, progress={"run_id": "run"},
        plan={"gaps": [{"gap_id": "G1", "question": "Which routes are documented?",
            "completion_criteria": ["Identify dated records"], "status": "INVESTIGATING",
            "coverage_assessment": {"criteria": [{"criterion": "Identify dated records", "addressed": False}],
                                    "rationale": "Records are still being collected."}}]})
    return project, package, ResearchSettings()


# Check small semantic clauses in the actual Agent-visible context, not a prompt snapshot.
@pytest.mark.parametrize("clauses", [
    ("Decompose broad research_scope", "independently assessable questions", "collectively cover",
     "do not merely restate the entire scope"),
    ("No fixed goal count", "genuinely narrow scope may have one goal"),
    ("concise, explicit completion_criteria", "what must be investigated"),
    ("critical=True means requested-scope work required for completion",
     "critical=False means optional enrichment", "Never mark scope-essential questions noncritical"),
    ("COVERED means the research criteria were adequately addressed", "goal's fact_ids"),
    ("RESEARCHED_UNRESOLVED means the criteria were genuinely investigated",
     "cannot support a definitive resolution", "nonempty supporting_fact_ids and unresolved_issues",
     "Never use it for an untouched goal"),
    ("Before marking any goal terminal, supply coverage_assessment", "exact declared criterion text",
     "addressed boolean", "supporting_fact_ids referencing persisted facts", "concise rationale",
     "every declared criterion exactly once"),
    ("OPEN and INVESTIGATING are nonterminal", "Preserve uncertainty",
     "rather than forcing disputed questions into COVERED"),
])
def test_goal_coverage_guidance_is_agent_visible(context_inputs, clauses):
    context = build_context(*context_inputs, False, {})
    instruction_text = " ".join(context[:len(INSTRUCTIONS)].split())
    for clause in clauses:
        assert clause in instruction_text
    assert "Unresolved critical gaps must remain OPEN/INVESTIGATING" not in instruction_text


def test_context_preserves_goal_contract_and_correction_state_without_duplicate_guidance(context_inputs):
    project, package, settings = context_inputs
    evidence = {"read_source_ids": ["S1"], "latest_read_source": {
        "source_id": "S1", "spans": [{"span_id": "SPAN-1", "text": "Retained source text."}]}}
    feedback = {"status": "validation_error", "message": "Correct the assessment"}
    for iteration in (1, 2):
        package.progress.iterations = iteration
        context = build_context(project, package, settings, False, {}, evidence, feedback)
        assert context.count(INSTRUCTIONS) == 1
        state = json.loads(context[len(INSTRUCTIONS):])
        assert state["plan"] == package.plan.model_dump(mode="json")
        assert state["project"]["research_scope"] == project.research_scope
        assert state["evidence_context"] == evidence
        assert state["previous_checkpoint_outcome"] == feedback
        assert state["iteration"] == iteration
        assert not state["coverage_status"]["can_complete"]
        assert len(context) <= settings.max_context_chars - settings.max_observation_chars


def test_context_limit_remains_authoritative(context_inputs):
    with pytest.raises(ContextLimitError, match="structured_context_limit"):
        build_context(*context_inputs, False, {}, evidence_context={"text": "x" * 24000})
