from enum import StrEnum
from typing import Self

from pydantic import Field, model_validator

from .base import Contract, Text


class TimePrecision(StrEnum):
    """YEAR: exact single year; RANGE: bounded year interval; CENTURY: bounded century.
    APPROXIMATE: approximate year, interval, or textual date; UNKNOWN: unspecified precision.
    APPROXIMATE and UNKNOWN allow absent bounds, a start alone, or ordered bounds.
    All precisions forbid an end without a start and require end_year >= start_year.
    """

    YEAR = "YEAR"
    APPROXIMATE = "APPROXIMATE"
    RANGE = "RANGE"
    CENTURY = "CENTURY"
    UNKNOWN = "UNKNOWN"


class HistoricalTime(Contract):
    """Historical date preserving its original expression; signed astronomical years (0 = 1 BCE).

    Supply all four fields in native tool calls; use null for unavailable bounds.
    YEAR requires start_year; end_year must be null or equal to start_year.
    RANGE and CENTURY require both bounds. Equal bounds are permitted; CENTURY does
    not enforce a particular interval length. APPROXIMATE and UNKNOWN allow both
    bounds null, start_year alone, or both bounds. For EVERY precision, a non-null
    end_year requires start_year and must be >= start_year. Never put a lone year
    in end_year. Preserve era/textual dates in display without inventing numeric bounds.

    Valid examples (each line is a complete tool value):
    {"display": "1200", "start_year": 1200, "end_year": null, "precision": "YEAR"}
    {"display": "1200", "start_year": 1200, "end_year": 1200, "precision": "YEAR"}
    {"display": "1200-1205", "start_year": 1200, "end_year": 1205, "precision": "RANGE"}
    {"display": "circa 1200", "start_year": 1200, "end_year": null, "precision": "APPROXIMATE"}
    {"display": "circa 1200-1205", "start_year": 1200, "end_year": 1205, "precision": "APPROXIMATE"}
    {"display": "13th century", "start_year": 1201, "end_year": 1300, "precision": "CENTURY"}
    {"display": "early in the era", "start_year": null, "end_year": null, "precision": "UNKNOWN"}
    {"display": "around the beginning of the era", "start_year": null, "end_year": null, "precision": "APPROXIMATE"}
    """

    display: Text = Field(description="Nonblank original historical date expression, including era names or textual uncertainty.")
    start_year: int | None = Field(default=None, strict=True, description=(
        "Signed astronomical integer year (0 = 1 BCE), or null. Put a single known year here, "
        "not only in end_year. Required non-null for YEAR, RANGE and CENTURY. "
        "If null, end_year must also be null."
    ))
    end_year: int | None = Field(default=None, strict=True, description=(
        "Signed astronomical integer year, or null. Non-null requires start_year and must be "
        ">= start_year. For YEAR use null or the same year as start_year; "
        "for RANGE and CENTURY both bounds must be non-null."
    ))
    precision: TimePrecision = Field(default=TimePrecision.UNKNOWN, description=(
        "YEAR: start required, end null or equal. RANGE/CENTURY: both bounds required. "
        "APPROXIMATE/UNKNOWN: both null, start alone, or ordered bounds are legal. "
        "Use APPROXIMATE for approximate dates and UNKNOWN for unspecified precision, "
        "including era/textual dates without numeric bounds."
    ))

    @model_validator(mode="after")
    def ordered_years(self) -> Self:
        if self.end_year is not None and self.start_year is None:
            raise ValueError("End year requires a start year")
        if self.start_year is not None and self.end_year is not None and self.end_year < self.start_year:
            raise ValueError("Historical range must be chronological")
        if self.precision == TimePrecision.YEAR:
            if self.start_year is None or self.end_year not in (None, self.start_year):
                raise ValueError("YEAR requires a single year")
        if self.precision in (TimePrecision.RANGE, TimePrecision.CENTURY):
            if self.start_year is None or self.end_year is None:
                raise ValueError("Range/century precision requires both bounds")
        return self
