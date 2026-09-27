from enum import StrEnum

from pydantic import Field

from history_studio.models.base import Contract, Identifier, Text
from history_studio.models.research import ResearchFact
from history_studio.models.research_package import ResearchPlan


class SearchRequest(Contract):
    query: Text = Field(max_length=500)


class ReadRequest(Contract):
    source_id: Identifier


class Recommendation(StrEnum):
    CONTINUE = "CONTINUE"
    COMPLETE = "COMPLETE"


class ResearchUpdate(Contract):
    """Internal canonical update after deterministic evidence resolution."""
    plan: ResearchPlan
    facts: list[ResearchFact] = Field(default_factory=list, max_length=20)
    recommendation: Recommendation


class EvidenceSelection(Contract):
    """Select evidence returned by read_source; Python extracts canonical text.
    Never supply an excerpt, offsets, locator, or replacement text. IDs are scoped
    to the exact source version. For unchanged evidence on the same existing fact,
    repeat its accepted source_id/span_id; Python preserves the persisted record without
    rereading. New or changed selections require a current read. Do not invent span IDs.
    """
    source_id: Identifier = Field(description="The source_id supplying this span, either accepted evidence on this same fact or a current read.")
    span_id: Identifier = Field(description="Repeat the accepted span_id for unchanged evidence on this same fact, or select from current read_source spans for new evidence. Python copies canonical text; do not transcribe or override excerpts.")


class ResearchFactProposal(ResearchFact):
    """Candidate claim with selected source spans; Python supplies persisted excerpts."""
    evidence: list[EvidenceSelection] = Field(min_length=1)


class ResearchSelectionUpdate(Contract):
    plan: ResearchPlan
    facts: list[ResearchFactProposal] = Field(default_factory=list, max_length=20)
    recommendation: Recommendation


def action_contracts() -> dict[str, type[Contract]]:
    return {"search_web": SearchRequest, "read_source": ReadRequest, "checkpoint_research": ResearchSelectionUpdate}
