from pydantic import Field

from .base import Contract, Identifier, Nonnegative, PositiveSeconds, Text


class ProjectConfig(Contract):
    project_id: Identifier
    topic: Text
    language: Text = "zh-CN"
    target_duration_minutes: PositiveSeconds = 5
    output_width: int = Field(default=1280, gt=0, strict=True)
    output_height: int = Field(default=720, gt=0, strict=True)
    budget_usd: Nonnegative = 20
