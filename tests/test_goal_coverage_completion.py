from copy import deepcopy
from types import SimpleNamespace

import pytest
from pydantic import ValidationError

from history_studio.models.research_package import ResearchPackage
from history_studio.research.progress import completion_status


SETTINGS = SimpleNamespace(min_facts=2, min_sources=2)


def package_data():
    return dict(project_id="test", topic="Historical records", progress={"run_id": "run"},
        sources=[dict(source_id=f"S{i}", title=f"Record {i}", url=f"https://example.org/{i}",
                      source_type="UNKNOWN", accessed_at="2026-01-01T00:00:00Z") for i in (1, 2)],
        facts=[dict(fact_id=f"F{i}", claim=f"Record {i} identifies a date", research_confidence=0.6,
                    historical_time={"display": "Date uncertain"},
                    evidence=[dict(source_id=f"S{i}", excerpt=f"Canonical quotation {i}")]) for i in (1, 2)],
        plan=dict(coverage_summary="Records examined.", gaps=[dict(
            gap_id="G1", question="Which date is supported?", status="COVERED", fact_ids=["F1", "F2"],
            completion_criteria=["Identify records", "Compare dates"], coverage_assessment=dict(
                criteria=[dict(criterion=c, addressed=True) for c in ("Identify records", "Compare dates")],
                supporting_fact_ids=["F1", "F2"], unresolved_issues=[], rationale="Both records assessed."))]))


@pytest.mark.parametrize("status", ["OPEN", "INVESTIGATING"])
def test_critical_nonterminal_blocks_despite_minimums(status):
    data = package_data()
    data["plan"]["gaps"][0]["status"] = status
    result = completion_status(ResearchPackage(**data), SETTINGS)
    assert result["missing_facts"] == result["missing_cited_sources"] == 0
    assert result["unresolved_critical_gaps"] == ["G1"]
    assert not result["can_complete"]


@pytest.mark.parametrize("run_status", ["RUNNING", "COMPLETE"])
@pytest.mark.parametrize("critical", [True, False])
def test_legacy_covered_readable_but_not_eligible(run_status, critical):
    data = package_data()
    data["progress"]["status"] = run_status
    goal = data["plan"]["gaps"][0]
    goal.pop("coverage_assessment")
    goal.pop("completion_criteria")
    goal["critical"] = critical
    package = ResearchPackage(**data)
    assert ResearchPackage.model_validate_json(package.model_dump_json()) == package
    assert not completion_status(package, SETTINGS)["can_complete"]


@pytest.mark.parametrize("status", ["COVERED", "RESEARCHED_UNRESOLVED"])
def test_valid_terminal_eligible_and_package_roundtrip(status):
    data = package_data()
    goal = data["plan"]["gaps"][0]
    goal["status"] = status
    if status == "RESEARCHED_UNRESOLVED":
        goal["coverage_assessment"]["unresolved_issues"] = ["Records conflict on the date"]
        # Python checks that criteria are assessed, not the Agent's semantic verdict.
        goal["coverage_assessment"]["criteria"][1]["addressed"] = False
    package = ResearchPackage(**data)
    assert completion_status(package, SETTINGS)["can_complete"]
    data["progress"]["status"] = "COMPLETE"
    complete = ResearchPackage(**data)
    assert ResearchPackage.model_validate_json(complete.model_dump_json()) == complete


@pytest.mark.parametrize("changes,match", [
    ({"supporting_fact_ids": []}, "persisted research support"),
    ({"unresolved_issues": []}, "require unresolved issues"),
    ({"unresolved_issues": [" "]}, "String should have at least"),
    ({"rationale": " "}, "String should have at least"),
])
def test_unresolved_cannot_bypass_research(changes, match):
    data = package_data()
    goal = data["plan"]["gaps"][0]
    goal["status"] = "RESEARCHED_UNRESOLVED"
    goal["coverage_assessment"].update(unresolved_issues=["Records disagree"])
    goal["coverage_assessment"].update(changes)
    with pytest.raises(ValidationError, match=match):
        ResearchPackage(**data)


def test_unresolved_requires_assessment():
    data = package_data()
    data["plan"]["gaps"][0].update(status="RESEARCHED_UNRESOLVED", coverage_assessment=None)
    with pytest.raises(ValidationError, match="require a coverage assessment"):
        ResearchPackage(**data)


@pytest.mark.parametrize("critical", [True, False])
@pytest.mark.parametrize("defect,match", [
    ("unknown_fact", "unknown fact"), ("missing_criterion", "match declared"),
    ("unknown_criterion", "match declared"), ("duplicate_criterion", "must be unique"),
])
def test_terminal_assessment_invariants_at_package_boundary(critical, defect, match):
    data = package_data()
    goal = data["plan"]["gaps"][0]
    goal["critical"] = critical
    assessment = goal["coverage_assessment"]
    if defect == "unknown_fact":
        assessment["supporting_fact_ids"] = ["F404"]
    elif defect == "missing_criterion":
        assessment["criteria"].pop()
    elif defect == "unknown_criterion":
        assessment["criteria"].append(dict(criterion="Unknown criterion", addressed=True))
    else:
        assessment["criteria"].append(deepcopy(assessment["criteria"][0]))
    with pytest.raises(ValidationError, match=match):
        ResearchPackage(**data)


@pytest.mark.parametrize("status", ["OPEN", "INVESTIGATING"])
def test_noncritical_nonterminal_does_not_block(status):
    data = package_data()
    data["plan"]["gaps"].append(dict(gap_id="G2", question="Extra context?", critical=False, status=status))
    assert completion_status(ResearchPackage(**data), SETTINGS)["can_complete"]


@pytest.mark.parametrize("requirement", ["facts", "cited_sources", "plan", "summary"])
def test_existing_completion_requirements_remain(requirement):
    data = package_data()
    settings = SETTINGS
    if requirement == "facts":
        settings = SimpleNamespace(min_facts=3, min_sources=2)
    elif requirement == "cited_sources":
        data["facts"][1]["evidence"][0]["source_id"] = "S1"
    elif requirement == "plan":
        data["plan"]["gaps"] = []
    else:
        data["plan"]["coverage_summary"] = " "
    assert not completion_status(ResearchPackage(**data), settings)["can_complete"]
