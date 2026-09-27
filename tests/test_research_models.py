from datetime import datetime, timezone

import pytest
from pydantic import ValidationError

from history_studio.models.historical_time import HistoricalTime
from history_studio.models.research import EvidenceReference, ResearchFact
from history_studio.models.research_package import ResearchPackage, ResearchProgress
from history_studio.models.sources import SourceReference
from history_studio.research.web_tools import source_reference


def fact_data() -> dict:
    return dict(fact_id="RF-1", claim="One historical claim", historical_time={"display": "约701年"},
                research_confidence=0.6, evidence=[dict(source_id="s1", excerpt="original quotation")],
                research_notes="Unverified interpretation")


@pytest.mark.parametrize("data", [
    dict(display="701", start_year=701, precision="YEAR"),
    dict(display="742–744", start_year=742, end_year=744, precision="RANGE"),
    dict(display="8世纪", start_year=701, end_year=800, precision="CENTURY"),
    dict(display="天宝元年"), dict(display="1 BCE", start_year=0, precision="YEAR"),
])
def test_historical_expressions(data: dict) -> None:
    assert HistoricalTime.model_validate_json(HistoricalTime(**data).model_dump_json()).display == data["display"]


@pytest.mark.parametrize("data", [dict(display=" "), dict(display="x", start_year=True),
    dict(display="x", end_year=2), dict(display="x", start_year=2, end_year=1),
    dict(display="x", precision="YEAR"), dict(display="x", start_year=1, precision="RANGE")])
def test_invalid_historical_time(data: dict) -> None:
    with pytest.raises(ValidationError):
        HistoricalTime(**data)


@pytest.mark.parametrize("changes", [dict(claim="one; two"), dict(claim="one\ntwo"), dict(claim=["one", "two"]),
    dict(evidence=[]), dict(research_confidence=1.1), dict(research_confidence=-0.1)])
def test_atomic_provenance_contract(changes: dict) -> None:
    with pytest.raises(ValidationError):
        ResearchFact(**(fact_data() | changes))


@pytest.mark.parametrize("data", [dict(source_id="s1", excerpt=" "), dict(source_id="../s", excerpt="quote"),
    dict(source_id="s1", excerpt="x" * 1201)])
def test_invalid_evidence(data: dict) -> None:
    with pytest.raises(ValidationError):
        EvidenceReference(**data)


def test_package_provenance_disputes_and_notes_projection() -> None:
    source = SourceReference(source_id="s1", title="source", url="https://example.org/",
                             source_type="UNKNOWN", accessed_at=datetime.now(timezone.utc))
    f1 = ResearchFact(**(fact_data() | {"dispute_group_id": "DG-1"}))
    f2 = ResearchFact(**(fact_data() | {"fact_id": "RF-2", "claim": "Competing claim", "dispute_group_id": "DG-1"}))
    package = ResearchPackage(project_id="test", topic="topic", sources=[source], facts=[f1, f2],
                              progress=ResearchProgress(run_id="run1"))
    assert package.facts[0].dispute_group_id == package.facts[1].dispute_group_id
    assert "research_notes" in package.model_dump_json()
    assert all("research_notes" not in fact for fact in package.facts_without_research_notes())
    for changes in ({"sources": []}, {"sources": [source, source]}, {"facts": [f1, f1]},
                    {"progress": {"run_id": "run1", "status": "COMPLETE"}}):
        with pytest.raises(ValidationError):
            ResearchPackage.model_validate(package.model_dump() | changes)


def test_source_dedup_identity_is_stable() -> None:
    a = source_reference("https://EXAMPLE.org/page?utm_source=test&b=2&a=1#section")
    b = source_reference("https://example.org/page?a=1&b=2")
    assert a.source_id == b.source_id
    assert a.url == b.url
    with pytest.raises(ValueError):
        source_reference("https://user:password@example.org/")


@pytest.mark.parametrize("precision, start, end", [
    ("YEAR", -1, None), ("YEAR", 1200, 1200),
    ("RANGE", 1200, 1200), ("CENTURY", 1200, 1200),
    *[(p, s, e) for p in ("APPROXIMATE", "UNKNOWN")
      for s, e in ((None, None), (1200, None), (1200, 1200), (1200, 1205))],
])
def test_historical_time_legal_bounds(precision, start, end) -> None:
    value = HistoricalTime(display="original expression", precision=precision,
                           start_year=start, end_year=end)
    assert (value.start_year, value.end_year) == (start, end)


@pytest.mark.parametrize("precision", ["YEAR", "RANGE", "CENTURY", "APPROXIMATE", "UNKNOWN"])
@pytest.mark.parametrize("start, end, message", [
    (None, 1200, "End year requires a start year"),
    (1205, 1200, "Historical range must be chronological"),
])
def test_historical_time_global_bounds(precision, start, end, message) -> None:
    with pytest.raises(ValidationError, match=message):
        HistoricalTime(display="date", precision=precision, start_year=start, end_year=end)


@pytest.mark.parametrize("precision, start, end, message", [
    ("YEAR", None, None, "YEAR requires a single year"),
    ("YEAR", 1200, 1205, "YEAR requires a single year"),
    *[(p, s, None, "requires both bounds") for p in ("RANGE", "CENTURY") for s in (None, 1200)],
])
def test_historical_time_precision_bounds(precision, start, end, message) -> None:
    with pytest.raises(ValidationError, match=message):
        HistoricalTime(display="date", precision=precision, start_year=start, end_year=end)


@pytest.mark.parametrize("field", ["start_year", "end_year"])
@pytest.mark.parametrize("value", [True, 1200.0, "1200"])
def test_historical_years_remain_strict_integers(field, value) -> None:
    with pytest.raises(ValidationError) as error:
        HistoricalTime.model_validate(dict(display="1200", start_year=1200, end_year=1200,
                                           precision="YEAR") | {field: value})
    assert error.value.errors()[0]["type"] == "int_type"
