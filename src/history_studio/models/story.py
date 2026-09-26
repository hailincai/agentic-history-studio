from typing import Self

from pydantic import Field, model_validator

from .base import Contract, Identifier, PositiveSeconds, Text, require_unique


class StoryBeat(Contract):
    sequence: int = Field(gt=0, strict=True)
    title: Text
    time_period: Text
    purpose: Text
    fact_ids: list[Identifier] = Field(min_length=1)
    target_seconds: PositiveSeconds


class StoryPlan(Contract):
    """Beats are ordered explicitly; historical date interpretation is deferred."""

    title: Text
    thesis: Text
    central_question: Text
    hook: Text
    beats: list[StoryBeat] = Field(min_length=1)
    ending: Text
    target_duration_seconds: PositiveSeconds

    @model_validator(mode="after")
    def ordered_beats(self) -> Self:
        sequences = [beat.sequence for beat in self.beats]
        require_unique(sequences, "Beat sequences")
        if sequences != sorted(sequences):
            raise ValueError("Beats must be in sequence order")
        return self

    @property
    def total_estimated_duration_seconds(self) -> float:
        return sum(beat.target_seconds for beat in self.beats)
