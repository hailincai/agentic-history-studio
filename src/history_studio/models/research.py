from pydantic import Field

from .base import Confidence, Contract, Identifier, Text
from .sources import SourceReference


class ResearchFact(Contract):
    id: Identifier
    claim: Text
    time_period: Text
    location: Text | None = None
    people: list[Text] = Field(default_factory=list)
    category: Text
    sources: list[SourceReference] = Field(min_length=1)
    researcher_confidence: Confidence
    notes: str = ""
