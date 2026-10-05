"""Bounded, detached Script Writer input; not a durable output artifact."""
from typing import Self

from pydantic import Field, model_validator

from .artifact_reference import ArtifactReference
from .story_package import StoryNarrativeBeat, StorySection, StoryStructure


class ScriptContextBeat(StoryNarrativeBeat):
    """Approved beat blueprint with only its authorized facts and exact chronology.

    Reuses Story's grounding shape and status/use policy without reinterpreting it.
    Nested values are detached by the Runtime builder, not upstream artifact handles.
    """


class ScriptContextSection(StorySection):
    """Preserves approved section identity, purpose and presentation order."""

    beats: tuple[ScriptContextBeat, ...] = Field(min_length=1)


class ScriptContext(StoryStructure):
    """Runtime-owned working projection of the exact approved Story blueprint.

    Reuses the existing blueprint shape/identity rules, with separate context types
    and no durable package envelope or upstream provenance. Approval is a caller
    precondition, not proved by construction. This is a semantic knowledge boundary,
    not a token budget; it adds no facts and does not authenticate generated Script.
    """

    story_input_ref: ArtifactReference
    sections: tuple[ScriptContextSection, ...] = Field(min_length=1)

    @model_validator(mode="after")
    def story_provenance(self) -> Self:
        if self.story_input_ref.artifact_type != "story":
            raise ValueError("Script input must reference a story artifact")
        return self
