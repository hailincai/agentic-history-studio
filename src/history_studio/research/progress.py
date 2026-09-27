"""Deterministic coverage and monotonic progress signals, independent of prose edits."""
import hashlib
import json
from collections import Counter

from history_studio.models.research_package import (
    GapStatus, TERMINAL_GOAL_STATUSES, terminal_assessment_error,
)


def completion_status(package, settings) -> dict:
    cited = {e.source_id for f in package.facts for e in f.evidence}
    fact_ids = {fact.fact_id for fact in package.facts}
    invalid_terminal = [g.gap_id for g in package.plan.gaps
        if g.status in TERMINAL_GOAL_STATUSES and terminal_assessment_error(g, fact_ids)]
    unresolved = [g.gap_id for g in package.plan.gaps
        if g.status not in TERMINAL_GOAL_STATUSES or g.gap_id in invalid_terminal]
    critical = [g.gap_id for g in package.plan.gaps if g.critical and g.gap_id in unresolved]
    result = dict(fact_count=len(package.facts), cited_source_count=len(cited),
        missing_facts=max(0, settings.min_facts - len(package.facts)),
        missing_cited_sources=max(0, settings.min_sources - len(cited)),
        invalid_terminal_assessments=invalid_terminal,
        unresolved_gaps=unresolved, unresolved_critical_gaps=critical,
        missing_plan=not package.plan.gaps, missing_coverage_summary=not package.plan.coverage_summary.strip())
    result["can_complete"] = not any(result[k] for k in
        ("missing_facts", "missing_cited_sources", "unresolved_critical_gaps",
         "invalid_terminal_assessments", "missing_plan", "missing_coverage_summary"))
    return result


def progress_marks(package, reads=None) -> set[str]:
    marks = set()
    def add(kind, value):
        digest = hashlib.sha256(json.dumps(value, ensure_ascii=False, sort_keys=True).encode()).hexdigest()
        mark = kind + ":" + digest
        marks.add(mark)
        return mark
    for source in package.sources:
        add("source", str(source.url))
    support_marks = {}
    for fact in package.facts:
        support = {add("fact", " ".join(fact.claim.split()).casefold())}
        support_marks[fact.fact_id] = support
        for evidence in fact.evidence:
            support.add(add("evidence", [evidence.source_id, evidence.source_version, evidence.span_id,
                             " ".join(evidence.excerpt.split())]))
            if evidence.source_version:
                add("read", [evidence.source_id, evidence.source_version])
    for read in (reads or {}).values():
        add("read", [read.source_id, read.source_version])
    for gap in package.plan.gaps:
        question = " ".join(gap.question.split()).casefold()
        add("question", question)
        rank = 0
        references = gap.fact_ids
        if gap.status == GapStatus.INVESTIGATING:
            rank = 1
        elif gap.status in TERMINAL_GOAL_STATUSES and not terminal_assessment_error(gap, set(support_marks)):
            rank = 2
            references = gap.coverage_assessment.supporting_fact_ids
        for step in range(1, rank + 1):
            # Keep the existing coverage identity; attach research dependencies so
            # status edits cannot earn credit by themselves. Both terminal outcomes
            # share one rank, preventing toggling between them from earning credit.
            base = add("coverage", [question, step])
            marks.remove(base)
            for fact_id in references:
                for support in support_marks.get(fact_id, ()):
                    marks.add(base + ":" + support)
    return marks


def progress_signals(marks: set[str], seen: set[str]) -> dict[str, int]:
    advanced = set()
    for mark in marks - seen:
        if mark.startswith("coverage:"):
            parts = mark.split(":", 2)
            # Legacy status-only marks never earn fresh credit. Credit each step
            # once, and only alongside new research linked to this goal.
            if len(parts) != 3 or parts[2] in seen:
                continue
            base = ":".join(parts[:2])
            if base in seen or any(old.startswith(base + ":") for old in seen):
                continue
            advanced.add(base)
        else:
            advanced.add(mark)
    return dict(sorted(Counter(mark.split(":", 1)[0] for mark in advanced).items()))
