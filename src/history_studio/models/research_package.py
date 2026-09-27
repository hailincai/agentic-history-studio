from enum import StrEnum
from typing import Literal, Self

from pydantic import Field, model_validator

from .base import Contract, Identifier, Text, require_unique
from .research import ResearchFact
from .sources import SourceReference


class GapStatus(StrEnum):
    OPEN = "OPEN"
    INVESTIGATING = "INVESTIGATING"
    COVERED = "COVERED"
    # Terminal research outcome: investigated, but evidence cannot resolve the question.
    RESEARCHED_UNRESOLVED = "RESEARCHED_UNRESOLVED"


class CriterionAssessment(Contract):
    """Agent-supplied assessment of one completion criterion, not a semantic verdict by Python."""

    criterion: Text = Field(max_length=600)
    addressed: bool = Field(strict=True)


class CoverageAssessment(Contract):
    """Agent assessment whose references are checked by the containing package."""

    criteria: list[CriterionAssessment] = Field(min_length=1)
    supporting_fact_ids: list[Identifier] = Field(default_factory=list)
    unresolved_issues: list[Text] = Field(default_factory=list)
    rationale: Text = Field(max_length=1500)


class ResearchGap(Contract):
    gap_id: Identifier
    question: Text = Field(max_length=600)
    critical: bool = True
    status: GapStatus = GapStatus.OPEN
    fact_ids: list[Identifier] = Field(default_factory=list)
    completion_criteria: list[Text] = Field(default_factory=list,
        description="Explicit criteria for independently assessing this research goal.")
    coverage_assessment: CoverageAssessment | None = None


# Goal terminology without renaming existing imports, schema definitions or JSON keys.
GoalStatus = GapStatus
ResearchGoal = ResearchGap


TERMINAL_GOAL_STATUSES = frozenset({GapStatus.COVERED, GapStatus.RESEARCHED_UNRESOLVED})


def terminal_assessment_error(goal: ResearchGap, fact_ids: set[str]) -> str | None:
    """Check structure/references only; the Agent owns semantic research judgments."""
    assessment = goal.coverage_assessment
    if assessment is None:
        return "Terminal goals require a coverage assessment"
    if not set(assessment.supporting_fact_ids) <= fact_ids:
        return "Coverage assessment references an unknown fact"
    assessed = [item.criterion for item in assessment.criteria]
    if len(assessed) != len(set(assessed)):
        return "Criterion assessments must be unique"
    if goal.completion_criteria and set(assessed) != set(goal.completion_criteria):
        return "Criterion assessments must match declared completion criteria"
    if goal.status == GapStatus.RESEARCHED_UNRESOLVED:
        if not assessment.supporting_fact_ids:
            return "Researched-unresolved goals require persisted research support"
        if not assessment.unresolved_issues:
            return "Researched-unresolved goals require unresolved issues"
    return None


class ResearchPlan(Contract):
    gaps: list[ResearchGap] = Field(default_factory=list, max_length=30)
    coverage_summary: str = Field(default="", max_length=1500)

    @model_validator(mode="after")
    def unique_gaps(self) -> Self:
        require_unique([gap.gap_id for gap in self.gaps], "Gap IDs")
        return self


class ResearchRunStatus(StrEnum):
    RUNNING = "RUNNING"
    COMPLETE = "COMPLETE"
    LIMIT_REACHED = "LIMIT_REACHED"
    FAILED = "FAILED"


class ResearchProgress(Contract):
    run_id: Identifier
    iterations: int = Field(default=0, ge=0)
    status: ResearchRunStatus = ResearchRunStatus.RUNNING
    stop_reason: str | None = None


class ResearchPackage(Contract):
    schema_version: Literal[2] = 2
    project_id: Identifier
    topic: Text
    research_scope: Text | None = None
    plan: ResearchPlan = Field(default_factory=ResearchPlan)
    facts: list[ResearchFact] = Field(default_factory=list)
    sources: list[SourceReference] = Field(default_factory=list)
    progress: ResearchProgress

    @model_validator(mode="after")
    def linked_provenance(self) -> Self:
        require_unique([fact.fact_id for fact in self.facts], "Fact IDs")
        require_unique([" ".join(fact.claim.split()).casefold() for fact in self.facts], "Atomic claims")
        require_unique([source.source_id for source in self.sources], "Source IDs")
        require_unique([str(source.url) for source in self.sources], "Source URLs")
        source_ids = {source.source_id for source in self.sources}
        fact_ids = {fact.fact_id for fact in self.facts}
        for fact in self.facts:
            if fact.sources or not fact.evidence or fact.historical_time is None:
                raise ValueError("Package facts require structured time and evidence-only provenance")
            if any(e.source_id not in source_ids for e in fact.evidence):
                raise ValueError("Evidence references an unknown source")
        for gap in self.plan.gaps:
            if not set(gap.fact_ids) <= fact_ids:
                raise ValueError("Gap references an unknown fact")
            if gap.status == GapStatus.COVERED and not gap.fact_ids:
                raise ValueError("Covered gaps require collected facts")
            assessment = gap.coverage_assessment
            if assessment is not None and not set(assessment.supporting_fact_ids) <= fact_ids:
                raise ValueError("Coverage assessment references an unknown fact")
            if gap.status in TERMINAL_GOAL_STATUSES:
                # Old COVERED artifacts remain readable, but completion eligibility
                # separately requires an assessment even for those legacy records.
                if gap.status == GapStatus.COVERED and assessment is None:
                    continue
                error = terminal_assessment_error(gap, fact_ids)
                if error:
                    raise ValueError(f"Goal {gap.gap_id}: {error}")
        if self.progress.status == ResearchRunStatus.COMPLETE:
            if not self.facts or not self.sources or not self.plan.gaps:
                raise ValueError("Complete research requires facts, sources and a plan")
            if any(g.critical and g.status not in TERMINAL_GOAL_STATUSES for g in self.plan.gaps):
                raise ValueError("Complete research cannot have unresolved critical gaps")
        return self

    def facts_without_research_notes(self) -> list[dict[str, object]]:
        """Explicit projection for future consumers; no story behavior is implemented."""
        return [fact.model_dump(mode="json", exclude={"research_notes", "sources"}) for fact in self.facts]
