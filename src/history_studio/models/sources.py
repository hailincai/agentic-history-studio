from datetime import date
from enum import StrEnum

from pydantic import AwareDatetime, HttpUrl

from .base import Contract, Identifier, Text


class SourceType(StrEnum):
    ACADEMIC = "ACADEMIC"
    MUSEUM = "MUSEUM"
    UNIVERSITY = "UNIVERSITY"
    BOOK = "BOOK"
    REFERENCE = "REFERENCE"
    PRIMARY_SOURCE = "PRIMARY_SOURCE"
    NEWS = "NEWS"
    WIKIPEDIA = "WIKIPEDIA"
    OTHER = "OTHER"
    INSTITUTIONAL = "INSTITUTIONAL"
    GENERAL_WEBSITE = "GENERAL_WEBSITE"
    USER_GENERATED = "USER_GENERATED"
    UNKNOWN = "UNKNOWN"


class SourceReference(Contract):
    source_id: Identifier
    title: Text
    url: HttpUrl
    source_type: SourceType
    domain: Text | None = None
    publisher: Text | None = None
    author: Text | None = None
    published_date: date | None = None
    accessed_at: AwareDatetime
    notes: str | None = None
