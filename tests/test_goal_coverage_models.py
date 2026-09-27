import json

import pytest
from pydantic import ValidationError

from history_studio.models.research_package import (
    CoverageAssessment, CriterionAssessment, GoalStatus, ResearchGoal,
    GapStatus, ResearchGap, ResearchPackage, ResearchPlan,
)


def assessment_data():
    return dict(criteria=[dict(criterion="Identify supporting records", addressed=True),
                          dict(criterion="Compare conflicting dates", addressed=True)],
                supporting_fact_ids=["RF-1", "RF-2"],
                unresolved_issues=["Available records disagree on the date"],
                rationale="Both records were examined; the date remains uncertain.")


def test_legacy_goal_and_package_defaults_roundtrip():
    legacy = {"schema_version": 2, "project_id": "test", "topic": "topic",
              "plan": {"gaps": [{"gap_id": "G1", "question": "Which records survive?",
                                  "critical": True, "status": "OPEN", "fact_ids": []}],
                       "coverage_summary": ""}, "facts": [], "sources": [],
              "progress": {"run_id": "run", "status": "RUNNING"}}
    package = ResearchPackage.model_validate_json(json.dumps(legacy))
    goal = package.plan.gaps[0]
    assert goal.completion_criteria == [] and goal.coverage_assessment is None
    assert ResearchGoal is ResearchGap and GoalStatus is GapStatus
    assert ResearchPackage.model_validate_json(package.model_dump_json()) == package
    assert "gap_id" in goal.model_dump() and "gaps" in package.plan.model_dump()
    other = ResearchGoal(gap_id="G2", question="Another question")
    goal.completion_criteria.append("Identify a record")
    assert other.completion_criteria == []


def test_criteria_assessment_and_unresolved_status_roundtrip():
    assessment = CoverageAssessment(**assessment_data())
    goal = ResearchGoal(gap_id="G1", question="What date is supported?",
        completion_criteria=[item.criterion for item in assessment.criteria],
        status=GoalStatus.RESEARCHED_UNRESOLVED, coverage_assessment=assessment)
    plan = ResearchPlan(gaps=[goal])
    data = json.loads(plan.model_dump_json())
    assert data["gaps"][0]["status"] == "RESEARCHED_UNRESOLVED"
    assert data["gaps"][0]["completion_criteria"] == [item.criterion for item in assessment.criteria]
    assert data["gaps"][0]["coverage_assessment"] == assessment_data()
    assert ResearchPlan.model_validate_json(plan.model_dump_json()) == plan
    # Package-level references are now enforced even though the standalone plan parses.
    with pytest.raises(ValidationError, match="unknown fact"):
        ResearchPackage(project_id="test", topic="topic", plan=plan, progress={"run_id": "run"})


def test_minimal_assessment_defaults_and_false_criterion():
    assessment = CoverageAssessment(criteria=[CriterionAssessment(criterion="Compare records", addressed=False)],
                                    rationale="Investigation is incomplete.")
    assert assessment.supporting_fact_ids == assessment.unresolved_issues == []
    assert CoverageAssessment.model_validate_json(assessment.model_dump_json()) == assessment


@pytest.mark.parametrize("changes", [
    {"criteria": []}, {"criteria": "not a list"},
    {"criteria": [{"criterion": " ", "addressed": True}]},
    {"criteria": [{"criterion": "Check records"}]},
    {"criteria": [{"criterion": "Check records", "addressed": "yes"}]},
    {"criteria": [{"criterion": "Check records", "addressed": 1}]},
    {"criteria": [{"criterion": "Check records", "addressed": True, "extra": "value"}]},
    {"supporting_fact_ids": ["../unsafe"]}, {"unresolved_issues": [" "]},
    {"rationale": " "}, {"rationale": "x" * 1501}, {"unknown": "field"},
])
def test_malformed_coverage_assessment_rejected(changes):
    with pytest.raises(ValidationError):
        CoverageAssessment.model_validate(assessment_data() | changes)


def test_invalid_goal_criteria_and_existing_invariants_remain():
    with pytest.raises(ValidationError):
        ResearchGoal(gap_id="G1", question="Question", completion_criteria=[" "])
    goal = ResearchGoal(gap_id="G1", question="Question")
    with pytest.raises(ValidationError, match="Gap IDs must be unique"):
        ResearchPlan(gaps=[goal, goal])
    with pytest.raises(ValidationError, match="Covered gaps require collected facts"):
        ResearchPackage(project_id="test", topic="topic", plan={"gaps": [
            {"gap_id": "G1", "question": "Question", "status": "COVERED"}]}, progress={"run_id": "run"})
    assessment = CoverageAssessment(**assessment_data())
    with pytest.raises(ValidationError):
        assessment.criteria[0].addressed = "yes"
