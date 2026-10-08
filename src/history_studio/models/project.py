from pydantic import Field, field_validator

from .base import Contract, Identifier, Nonnegative, PositiveSeconds, Text
from .storyboard import GenerationMethod


class ProjectConfig(Contract):
    project_id: Identifier
    research_scope: Text | None = None
    topic: Text
    language: Text = "zh-CN"
    target_duration_minutes: PositiveSeconds = 5
    output_width: int = Field(default=1280, gt=0, strict=True)
    output_height: int = Field(default=720, gt=0, strict=True)
    budget_usd: Nonnegative = 20
    allowed_generation_methods: tuple[GenerationMethod, ...] = Field(default=tuple(GenerationMethod), min_length=1)

    @field_validator("allowed_generation_methods")
    @classmethod
    def unique_methods(cls, value):
        if len(set(value)) != len(value):
            raise ValueError("Allowed generation methods must be unique")
        return value
