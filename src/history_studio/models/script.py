from typing import Self

from pydantic import Field, model_validator

from .base import Contract, Identifier, PositiveSeconds, Text, require_unique


class ScriptScene(Contract):
    scene_id: Identifier
    sequence: int = Field(gt=0, strict=True)
    narration: Text
    fact_ids: list[Identifier] = Field(min_length=1)
    duration_seconds: PositiveSeconds


class Script(Contract):
    title: Text
    scenes: list[ScriptScene] = Field(min_length=1)
    target_duration_seconds: PositiveSeconds

    @model_validator(mode="after")
    def unique_scenes(self) -> Self:
        require_unique([scene.scene_id for scene in self.scenes], "Scene IDs")
        sequences = [scene.sequence for scene in self.scenes]
        require_unique(sequences, "Scene sequences")
        if sequences != sorted(sequences):
            raise ValueError("Scenes must be in sequence order")
        return self

    @property
    def total_estimated_duration_seconds(self) -> float:
        return sum(scene.duration_seconds for scene in self.scenes)
