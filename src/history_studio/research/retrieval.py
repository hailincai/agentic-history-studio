"""Deterministic lexical retrieval over existing canonical spans; no provider dependency."""
import json
import re
import unicodedata
from collections.abc import Iterable

from .spans import SourceSpans
from history_studio.models.research_package import TERMINAL_GOAL_STATUSES


def lexical_terms(text: str) -> set[str]:
    text = unicodedata.normalize("NFKC", text).casefold()
    # Chinese characters/bigrams plus Unicode word tokens. Normalization affects
    # ranking only, never the canonical text or identifiers returned to the Agent.
    han = r"[\u3400-\u4dbf\u4e00-\u9fff\U00020000-\U0003134f]"
    terms = set()
    for run in re.findall(han + "+", text):
        terms.update(run)
        terms.update(run[i:i + 2] for i in range(len(run) - 1))
    terms.update(re.findall(r"[^\W_]+", re.sub(han, " ", text)))
    return terms


def retrieve_relevant_spans(questions: Iterable[str], source: SourceSpans,
                            context_budget: int) -> dict:
    """Return a whole-span selection whose compact JSON fits context_budget.

    Score = distinct query-term overlap; ties follow canonical source offsets.
    Zero-overlap spans are excluded. A non-fitting span does not prevent later
    smaller relevant spans from being selected. Empty selection means no fit/match.
    """
    terms = lexical_terms(" ".join(questions))
    ranked = []
    for span_id, (start, end) in source.spans.items():
        score = len(terms & lexical_terms(source.text[start:end]))
        if score:
            ranked.append((-score, start, span_id, source.text[start:end]))
    selected = dict(source_id=source.source_id, source_version=source.source_version,
                    truncated=source.truncated, total_span_count=len(source.spans),
                    retrieved_subset=True, spans=[])
    for _, _, span_id, text in sorted(ranked):
        item = dict(span_id=span_id, text=text)
        candidate = {**selected, "spans": [*selected["spans"], item],
                     "retrieved_subset": len(selected["spans"]) + 1 < len(source.spans)}
        if len(json.dumps(candidate, ensure_ascii=False, separators=(",", ":"))) <= context_budget:
            selected = candidate
    return selected



def query_texts(topic, research_scope, goals) -> list[str]:
    """Shared active-goal query with stable topic/scope anchors (including after resume)."""
    return [text for goal in goals if goal.status not in TERMINAL_GOAL_STATUSES
            for text in [goal.question, *goal.completion_criteria]] + [research_scope or "", topic]
