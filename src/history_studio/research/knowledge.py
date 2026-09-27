"""Context-only accepted knowledge retrieval; persistence and authorization stay canonical."""
import hashlib
import json
from dataclasses import dataclass
from typing import Protocol, Sequence, TypedDict

from history_studio.models.research import ResearchFact
from history_studio.models.research_package import ResearchGap, ResearchPlan
from history_studio.models.sources import SourceReference
from .retrieval import lexical_terms, query_texts


def serialized_size(value: object) -> int:
    return len(json.dumps(value, ensure_ascii=False, separators=(",", ":")))


@dataclass(frozen=True)
class KnowledgeQuery:
    topic: str
    research_scope: str | None
    goals: Sequence[ResearchGap]


class KnowledgeProjection(TypedDict):
    facts: list[dict]
    evidence: dict[str, dict]
    sources: list[dict]


def empty_knowledge() -> KnowledgeProjection:
    return {"facts": [], "evidence": {}, "sources": []}


class KnowledgeRetriever(Protocol):
    def retrieve(self, query: KnowledgeQuery, facts: Sequence[ResearchFact],
                 sources: Sequence[SourceReference], budget_chars: int) -> KnowledgeProjection:
        """Return whole accepted facts, linked evidence and source catalog within budget."""
        ...


def plan_overview(plan: ResearchPlan) -> dict:
    """Preserve every goal/criterion and uncertainty without repeating accepted details."""
    goals = []
    for goal in plan.gaps:
        assessment = goal.coverage_assessment
        addressed = {c.criterion: c.addressed for c in assessment.criteria} if assessment else {}
        goals.append(dict(gap_id=goal.gap_id, question=goal.question, critical=goal.critical,
            status=goal.status.value, completion_criteria=list(goal.completion_criteria),
            fact_count=len(goal.fact_ids), coverage_assessment=(None if assessment is None else dict(
                addressed=[addressed.get(c) for c in goal.completion_criteria],
                supporting_fact_count=len(assessment.supporting_fact_ids),
                unresolved_issues=list(assessment.unresolved_issues)))))
    return {"gaps": goals}


def compact_source(source: SourceReference) -> dict:
    result = dict(source_id=source.source_id, title=source.title, url=str(source.url))
    if source.source_type.value != "UNKNOWN":
        result["source_type"] = source.source_type.value
    return result


class LexicalKnowledgeRetriever:
    """Rank claims/notes by shared F1 terms; evidence follows facts without ranking."""

    def retrieve(self, query: KnowledgeQuery, facts: Sequence[ResearchFact],
                 sources: Sequence[SourceReference], budget_chars: int) -> KnowledgeProjection:
        result = empty_knowledge()
        if serialized_size(result) > budget_chars:
            raise ValueError("Knowledge budget cannot fit its empty projection")
        terms = lexical_terms(" ".join(query_texts(query.topic, query.research_scope, query.goals)))
        catalog = {source.source_id: compact_source(source) for source in sources}
        ranked = []
        for fact in facts:
            score = len(terms & lexical_terms(fact.claim + " " + (fact.research_notes or "")))
            if score:
                ranked.append((-score, fact.fact_id, fact))
        # Reserve a quarter for discovery choices; selected facts carry their own
        # source records atomically, even if that means skipping a large bundle.
        fact_budget = max(serialized_size(result), budget_chars * 3 // 4)
        source_ids = set()
        for _, _, fact in sorted(ranked, key=lambda item: item[:2]):
            evidence = dict(result["evidence"])
            keys = []
            for ref in fact.evidence:
                record = ref.model_dump(mode="json")
                identity = json.dumps(record, ensure_ascii=False, sort_keys=True, separators=(",", ":"))
                key = "E-" + hashlib.sha256(identity.encode("utf-8")).hexdigest()
                evidence[key] = record
                keys.append(key)
            projected = fact.model_dump(mode="json", exclude={"sources", "time_period", "evidence"})
            projected["evidence_ids"] = keys
            additions = sorted({ref.source_id for ref in fact.evidence} - source_ids)
            candidate = dict(facts=[*result["facts"], projected], evidence=evidence,
                             sources=[*result["sources"], *(catalog[key] for key in additions)])
            if serialized_size(candidate) <= fact_budget:
                result = candidate
                source_ids.update(additions)
        # Source choices are compact and bounded too; never dump all discoveries.
        ranked_sources = sorted(sources, key=lambda source: (
            -len(terms & lexical_terms((source.title or "") + " " + str(source.url))), source.source_id))
        for source in ranked_sources:
            if source.source_id in source_ids:
                continue
            candidate = {**result, "sources": [*result["sources"], catalog[source.source_id]]}
            if serialized_size(candidate) <= budget_chars:
                result = candidate
                source_ids.add(source.source_id)
        return result
