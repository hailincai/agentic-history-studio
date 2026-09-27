from pathlib import Path

import pytest

from history_studio.research.boundaries import Usage
from history_studio.research.usage import LimitReached, UsageLedger


def test_reservations_survive_unknown_usage_and_cannot_be_reset(tmp_path: Path) -> None:
    path = tmp_path / "usage.json"
    ledger = UsageLedger(run_id="run1")
    ledger.reserve(0.2, 0.35, "model", path)
    restored = UsageLedger.model_validate_json(path.read_text())
    assert restored.pending_request
    restored.record(Usage(model="unknown"), path)
    assert restored.unknown_usage
    assert restored.committed_budget_usd == 0.2
    with pytest.raises(LimitReached, match="hard_budget"):
        restored.reserve(0.2, 0.35, "search", path)
    assert restored.model_calls == 1
    assert restored.search_calls == 0


def test_bad_price_or_provider_overrun_stops(tmp_path: Path) -> None:
    path = tmp_path / "usage.json"
    ledger = UsageLedger(run_id="run1")
    with pytest.raises(LimitReached, match="unknown_cost_bound"):
        ledger.reserve(float("nan"), 1, "model", path)
    ledger.reserve(0.1, 1, "model", path)
    with pytest.raises(LimitReached, match="exceeded_reservation"):
        ledger.record(Usage(model="test", input_tokens=1, output_tokens=1,
                            estimated_model_cost_usd=0.2), path)
    assert UsageLedger.model_validate_json(path.read_text()).estimated_total_cost_usd == 0.2

    with pytest.raises(LimitReached, match="exceeded_reservation"):
        ledger.reserve(0.01, 1, "model", path)
