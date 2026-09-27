from typing import Self

from pydantic import Field, model_validator

from history_studio.models.base import Contract, Nonnegative


class ResearchSettings(Contract):
    soft_budget_usd: Nonnegative = 0.50
    hard_budget_usd: Nonnegative = 1.00
    max_iterations: int = Field(default=12, ge=1, le=100)
    max_searches: int = Field(default=20, ge=0)
    max_source_reads: int = Field(default=30, ge=0)
    max_turns_per_iteration: int = Field(default=8, ge=1, le=30)
    max_no_progress_checkpoints: int = Field(default=3, ge=1, le=20)
    min_facts: int = Field(default=3, ge=1)
    min_sources: int = Field(default=2, ge=1)
    max_context_chars: int = Field(default=24000, ge=4000)
    max_observation_chars: int = Field(default=9000, ge=1000)
    max_page_chars: int = Field(default=8000, ge=1000, le=30000)
    max_output_tokens: int = Field(default=3000, ge=500, le=10000)

    @model_validator(mode="after")
    def ordered_limits(self) -> Self:
        if self.soft_budget_usd > self.hard_budget_usd:
            raise ValueError("Soft budget cannot exceed hard budget")
        if self.max_observation_chars >= self.max_context_chars:
            raise ValueError("Observation must fit inside the context budget")
        return self
