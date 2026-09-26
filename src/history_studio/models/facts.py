from enum import StrEnum

from pydantic import Field

from .base import Confidence, Contract, Identifier, Text
from .sources import SourceReference


class FactStatus(StrEnum):
    VERIFIED = "VERIFIED"
    PARTIALLY_VERIFIED = "PARTIALLY_VERIFIED"
    DISPUTED = "DISPUTED"
    REJECTED = "REJECTED"


class VerifiedFact(Contract):
    fact_id: Identifier
    claim: Text
    status: FactStatus
    confidence: Confidence
    supporting_sources: list[SourceReference] = Field(default_factory=list)
    conflicting_sources: list[SourceReference] = Field(default_factory=list)
    reasoning: Text
