from pathlib import Path
from typing import Any, Literal

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
        self.persist(path)
        if self.estimated_total_cost_usd > self.committed_budget_usd + 1e-9:
            raise LimitReached("provider_cost_exceeded_reservation")
