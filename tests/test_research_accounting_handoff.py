"""Local authenticated accounting recovery; never dispatch hosted search."""
from decimal import Decimal
from types import SimpleNamespace

import pytest

from history_studio.budget import BudgetError, BudgetExceeded, UnsupportedPrice, research_accounting
from history_studio.research.agent import ResearchAgent
from history_studio.research.usage import UsageLedger
from history_studio.storage import ArtifactStore
from test_project_budget import setup, ledger
from test_research_agent import FakeTools, settings, setup_run


@pytest.mark.parametrize("block", [0, 8000, 16000])
def test_observed_search_usage_cannot_authorize_dispatch(tmp_path, block):
    _, budget = setup(tmp_path)
    def forbidden():
        pytest.fail("No search billing convention has an established hard bound")
    with pytest.raises(UnsupportedPrice, match="billing semantics"):
        budget.responses(dict(model="gpt-4.1-mini", input="李白", tools=[{"type": "web_search"}],
                              max_tool_calls=1, max_output_tokens=1000), forbidden,
                         input_rate=.4, output_rate=1.6, search_content_tokens=block, search_call_usd=.01)
    assert len(ledger(budget).events) == 1


def test_observed_search_settlement_handoff_is_exact_and_idempotent(tmp_path):
    root, budget = setup(tmp_path)
    usage = UsageLedger(run_id="offline")
    path = root / ".runtime/research_usage.json"
    usage.reserve(.0165908, .25, "search", path)
    pricing = dict(input_usd_per_million=".4", output_usd_per_million="1.6",
                   search_content_tokens=8000, search_call_usd=".01")
    with research_accounting(usage.pending_accounting_id):
        budget.reserve(request_id="observed", operation="hosted_search", model="gpt-4.1-mini",
                       request_sha256="offline", maximum_usd=".0165908", basis=pricing)
    # Synthetic authenticated evidence reproduces the old calculation, without enabling it.
    with pytest.raises(BudgetExceeded) as caught:
        budget.close("observed", amount_usd=".0169232", basis=dict(response_id="offline",
                     input_tokens=8596, output_tokens=178, pricing=pricing))
    assert caught.value.settlement.amount_usd == Decimal(".0169232")
    assert usage.recover_settlement(root, path)
    restored = UsageLedger.model_validate_json(path.read_text())
    assert restored.input_tokens == 8596 and restored.output_tokens == 178
    assert restored.estimated_total_cost_usd == pytest.approx(.0169232)
    assert restored.estimated_tool_cost_usd == .01
    assert not restored.pending_request and not restored.unknown_usage
    before = path.read_bytes(), budget.path.read_bytes()
    assert not restored.recover_settlement(root, path)
    assert before == (path.read_bytes(), budget.path.read_bytes())


@pytest.mark.parametrize("failure", ["overrun", "after_settlement"])
def test_agent_restart_never_replays_authenticated_handoff(tmp_path, monkeypatch, failure):
    from history_studio.budget import ProjectBudget
    project, store = setup_run(tmp_path)
    budget = ProjectBudget(store.project_dir, stage="research")
    budget.initialize_new()
    calls = []
    class Provider:
        def reserve_cost(self, *args):
            return .03
        def decide(self, *args):
            calls.append("dispatch")
            raw = SimpleNamespace(model="gpt-4.1-mini", id="offline", output=[],
                usage=SimpleNamespace(input_tokens=50000 if failure == "overrun" else 50, output_tokens=178))
            budget.responses(dict(model="gpt-4.1-mini", input="local", tools=[], max_output_tokens=1000),
                             lambda: raw, input_rate=.4, output_rate=1.6)
            raise RuntimeError("safe injected failure after settlement")
    result = ResearchAgent(Provider(), FakeTools(), settings()).run(project, store)
    assert result.progress.status == ("LIMIT_REACHED" if failure == "overrun" else "FAILED")
    paid_before = budget.path.read_bytes()
    for _ in range(2):
        result = ResearchAgent(Provider(), FakeTools(), settings()).run(project, ArtifactStore(store.project_dir))
        assert result.progress.status == "LIMIT_REACHED"
    assert calls == ["dispatch"]
    assert budget.path.read_bytes() == paid_before
    usage = UsageLedger.model_validate_json((store.project_dir / ".runtime/research_usage.json").read_text())
    assert not usage.pending_request and usage.accounting_recovery_stop
    assert not usage.unknown_usage
    assert usage.input_tokens == (50000 if failure == "overrun" else 50)
    assert usage.estimated_total_cost_usd == pytest.approx(float(budget.snapshot()["committed_usd"]))
    assert result.progress.stop_reason == "authenticated_accounting_handoff_recovered"


def test_unknown_and_legacy_handoffs_are_not_inferred(tmp_path):
    root, budget = setup(tmp_path)
    usage = UsageLedger(run_id="legacy", pending_request=True)
    path = root / ".runtime/research_usage.json"
    usage.persist(path)
    before = path.read_bytes(), budget.path.read_bytes()
    assert not usage.recover_settlement(root, path)
    assert before == (path.read_bytes(), budget.path.read_bytes())
    usage.reserve(.01, .25, "model", path)
    assert not usage.recover_settlement(root, path)
    assert usage.pending_request


def test_recovery_preserves_earlier_unknown_usage(tmp_path):
    root, budget = setup(tmp_path)
    usage = UsageLedger(run_id="offline", unknown_usage=True)
    path = root / ".runtime/research_usage.json"
    usage.reserve(.01, .25, "model", path)
    pricing = dict(input_usd_per_million=".4", output_usd_per_million="1.6",
                   search_content_tokens=0, search_call_usd="0")
    with research_accounting(usage.pending_accounting_id):
        budget.reserve(request_id="known", operation="responses", model="gpt-4.1-mini",
                       request_sha256="offline", maximum_usd=".01", basis=pricing)
    budget.close("known", amount_usd=".000036", basis=dict(response_id="offline",
                 input_tokens=50, output_tokens=10, pricing=pricing))
    assert usage.recover_settlement(root, path)
    assert usage.unknown_usage


@pytest.mark.parametrize("settle", [False, True])
def test_recovery_never_trusts_unknown_or_inconsistent_settlement(tmp_path, settle):
    root, budget = setup(tmp_path)
    usage = UsageLedger(run_id="offline")
    path = root / ".runtime/research_usage.json"
    usage.reserve(.01, .25, "model", path)
    with research_accounting(usage.pending_accounting_id):
        budget.reserve(request_id="uncertain", operation="responses", model="gpt-4.1-mini",
                       request_sha256="offline", maximum_usd=".01", basis={})
    if settle:
        budget.close("uncertain", amount_usd=".0001", basis={"response_id": "offline"})
    before = path.read_bytes(), budget.path.read_bytes()
    if settle:
        with pytest.raises(BudgetError, match="authenticated usage"):
            usage.recover_settlement(root, path)
    else:
        assert not usage.recover_settlement(root, path)
        assert budget.snapshot()["outstanding_usd"] == Decimal(".01")
    assert before == (path.read_bytes(), budget.path.read_bytes())
    assert usage.pending_request
