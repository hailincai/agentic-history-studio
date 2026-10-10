from pathlib import Path
from typing import Any, Literal
from uuid import uuid4

from pydantic import Field

from history_studio.models.base import Contract, Nonnegative
from history_studio.storage.artifact_store import write_json
from .boundaries import Usage


class LimitReached(RuntimeError):
    """Contains an internal safe condition code, never provider error text."""


class UsageLedger(Contract):
    run_id: str
    iterations_started: int = Field(default=0, ge=0)
    model_calls: int = Field(default=0, ge=0)
    tool_calls: int = Field(default=0, ge=0)
    search_calls: int = Field(default=0, ge=0)
    source_reads: int = Field(default=0, ge=0)
    input_tokens: int = Field(default=0, ge=0)
    output_tokens: int = Field(default=0, ge=0)
    models: list[str] = Field(default_factory=list)
    estimated_model_cost_usd: Nonnegative = 0
    estimated_tool_cost_usd: Nonnegative = 0
    committed_budget_usd: Nonnegative = 0
    unknown_usage: bool = False
    pending_request: bool = False
    pending_accounting_id: str | None = None
    pending_prior_unknown_usage: bool | None = None
    accounting_recovery_stop: bool = False
    consecutive_no_progress: int = Field(default=0, ge=0)
    progress_seen: list[str] = Field(default_factory=list)
    last_checkpoint_outcome: dict[str, Any] | None = None

    @property
    def estimated_total_cost_usd(self) -> float:
        return self.estimated_model_cost_usd + self.estimated_tool_cost_usd

    def persist(self, path: Path) -> None:
        write_json(path, self, replace=True)

    def reserve(self, amount: float, hard_budget: float, kind: Literal["model", "search", "read"],
                path: Path) -> None:
        import math
        if self.estimated_total_cost_usd > self.committed_budget_usd + 1e-9:
            raise LimitReached("provider_cost_exceeded_reservation")
        if not math.isfinite(amount) or amount < 0:
            raise LimitReached("unknown_cost_bound")
        if self.committed_budget_usd + amount > hard_budget + 1e-12:
            raise LimitReached("hard_budget")
        # Never release reservations, even on missing usage, timeout or crash.
        self.committed_budget_usd += amount
        self.pending_request = True
        self.pending_accounting_id = self.run_id + ":" + uuid4().hex
        self.pending_prior_unknown_usage = self.unknown_usage
        if kind in ("model", "search"):
            self.model_calls += 1
        if kind == "search":
            self.search_calls += 1
        elif kind == "read":
            self.source_reads += 1
        self.persist(path)

    def record(self, usage: Usage, path: Path) -> None:
        if usage.model:
            if usage.model not in self.models:
                self.models.append(usage.model)
            self.input_tokens += usage.input_tokens or 0
            self.output_tokens += usage.output_tokens or 0
            self.unknown_usage |= (usage.input_tokens is None or usage.output_tokens is None
                                   or usage.estimated_model_cost_usd is None)
        self.estimated_model_cost_usd += usage.estimated_model_cost_usd or 0
        self.estimated_tool_cost_usd += usage.estimated_tool_cost_usd or 0
        self.pending_request = False
        self.pending_accounting_id = None
        self.pending_prior_unknown_usage = None
        self.persist(path)
        if self.estimated_total_cost_usd > self.committed_budget_usd + 1e-9:
            raise LimitReached("provider_cost_exceeded_reservation")

    def recover_settlement(self, project_dir: Path, path: Path) -> bool:
        """Apply exact authenticated usage once, then stop without replaying lost output."""
        if not self.pending_request or self.pending_accounting_id is None:
            return False
        from history_studio.budget import ProjectBudget, BudgetError, usd
        if not (project_dir / ".runtime/project_budget.json").exists():
            return False  # Fake/legacy runs carry no project settlement authority.
        event = ProjectBudget(project_dir, stage="research").research_settlement(self.pending_accounting_id)
        if event is None:
            return False
        basis = event.basis
        counts = basis.get("input_tokens"), basis.get("output_tokens")
        pricing = basis.get("pricing", {})
        if (event.operation not in ("responses", "hosted_search")
                or not isinstance(basis.get("response_id"), str) or not basis["response_id"]
                or any(type(count) is not int or count < 0 for count in counts)):
            raise BudgetError("Research settlement lacks authenticated usage evidence")
        tool = usd(pricing.get("search_call_usd", 0))
        amount = ((counts[0] + pricing.get("search_content_tokens", 0))
                  * usd(pricing.get("input_usd_per_million"))
                  + counts[1] * usd(pricing.get("output_usd_per_million"))) / 1000000 + tool
        if amount != event.amount_usd:
            raise BudgetError("Research settlement accounting evidence differs")
        if self.pending_prior_unknown_usage is not None:
            self.unknown_usage = self.pending_prior_unknown_usage
        self.accounting_recovery_stop = True
        try:
            self.record(Usage(model=event.model, input_tokens=counts[0], output_tokens=counts[1],
                              estimated_model_cost_usd=float(amount - tool),
                              estimated_tool_cost_usd=float(tool)), path)
        except LimitReached:
            pass  # The same durable snapshot already records the overrun and stop flag.
        return True
