from typing import Self

from pydantic import AliasChoices, Field, field_validator, model_validator

from .base import Confidence, Contract, Identifier, Text
from .historical_time import HistoricalTime
from .sources import SourceReference


class EvidenceReference(Contract):
    """Persisted canonical source quotation, never model-authored evidence.

    Python resolves an Agent-selected source/span into excerpt text. Legacy artifacts
    without span metadata remain readable; excerpts still require exact provenance.
    Interpretation belongs in the fact's claim or research_notes, not in this text.
    """

    source_id: Identifier = Field(description="Identity of the source supplying this canonical evidence.")
    excerpt: Text = Field(max_length=1200, description=(
        "Canonical contiguous source text extracted by Python from the selected span. "
        "The Agent selects evidence IDs and cannot override this persisted excerpt."
    ))
    locator: Text | None = None
    source_version: Identifier | None = None
    span_id: Identifier | None = None

    @model_validator(mode="after")
    def paired_span_identity(self) -> Self:
        if (self.source_version is None) != (self.span_id is None):
            raise ValueError("Span ID and source version must be supplied together")
        return self


class ResearchFact(Contract):
    """One candidate claim, not a verified fact. Atomicity also needs semantic review.

    Phase 1 field spellings and embedded sources remain readable for compatibility.
    ResearchPackage requires the new evidence-only provenance form.
    """

    fact_id: Identifier = Field(validation_alias=AliasChoices("fact_id", "id"))
    claim: Text = Field(max_length=600, description="One atomic candidate claim; paraphrase/synthesis belongs here, while evidence.excerpt must remain verbatim source text.")
    historical_time: HistoricalTime | None = None
    time_period: Text | None = None
    location: Text | None = None
    people: list[Text] = Field(default_factory=list)
    category: Text = "historical"
    sources: list[SourceReference] = Field(default_factory=list)
    evidence: list[EvidenceReference] = Field(default_factory=list)
    research_confidence: Confidence = Field(validation_alias=AliasChoices("research_confidence", "researcher_confidence"))
    dispute_group_id: Identifier | None = None
    research_notes: str = Field(default="", max_length=2000, validation_alias=AliasChoices("research_notes", "notes"))

    @field_validator("claim")
    @classmethod
    def single_claim_shape(cls, value: str) -> str:
        if "\n" in value or ";" in value or "；" in value:
            raise ValueError("Use one atomic claim, not a list or semicolon-separated claims")
        return value

    @model_validator(mode="after")
    def provenance(self) -> Self:
        if not self.evidence and not self.sources:
            raise ValueError("A research fact requires evidence provenance")
        if self.historical_time is None and self.time_period is None:
            raise ValueError("A research fact requires a historical time expression")
        return self

    @property
    def id(self) -> str:
        return self.fact_id

    @property
    def researcher_confidence(self) -> float:
        return self.research_confidence

    @property
    def notes(self) -> str:
        return self.research_notes
