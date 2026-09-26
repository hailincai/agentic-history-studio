from enum import StrEnum
from typing import Self

from pydantic import model_validator

from history_studio.models.base import Contract, Text


class ProjectState(StrEnum):
    CREATED = "CREATED"
    RESEARCHING = "RESEARCHING"
    RESEARCH_COMPLETE = "RESEARCH_COMPLETE"
    FACT_CHECKING = "FACT_CHECKING"
    WAITING_FACT_APPROVAL = "WAITING_FACT_APPROVAL"
    FACTS_APPROVED = "FACTS_APPROVED"
    STORY_GENERATING = "STORY_GENERATING"
    WAITING_STORY_APPROVAL = "WAITING_STORY_APPROVAL"
    STORY_APPROVED = "STORY_APPROVED"
    SCRIPT_GENERATING = "SCRIPT_GENERATING"
    WAITING_SCRIPT_APPROVAL = "WAITING_SCRIPT_APPROVAL"
    SCRIPT_APPROVED = "SCRIPT_APPROVED"
    STORYBOARD_GENERATING = "STORYBOARD_GENERATING"
    WAITING_STORYBOARD_APPROVAL = "WAITING_STORYBOARD_APPROVAL"
    STORYBOARD_APPROVED = "STORYBOARD_APPROVED"
    GENERATING_MEDIA = "GENERATING_MEDIA"
    ASSEMBLING = "ASSEMBLING"
    COMPLETE = "COMPLETE"
    FAILED = "FAILED"


# Enter these states only after the corresponding result/decision is persisted.
# Running stages (including media generation and assembly) are not checkpoints.
DURABLE_CHECKPOINTS = frozenset({
    ProjectState.CREATED,
    ProjectState.RESEARCH_COMPLETE,
    ProjectState.WAITING_FACT_APPROVAL,
    ProjectState.FACTS_APPROVED,
    ProjectState.WAITING_STORY_APPROVAL,
    ProjectState.STORY_APPROVED,
    ProjectState.WAITING_SCRIPT_APPROVAL,
    ProjectState.SCRIPT_APPROVED,
    ProjectState.WAITING_STORYBOARD_APPROVAL,
    ProjectState.STORYBOARD_APPROVED,
    ProjectState.COMPLETE,
})


class RuntimeState(Contract):
    model_config = {"frozen": True}
    current_state: ProjectState = ProjectState.CREATED
    last_successful_state: ProjectState = ProjectState.CREATED
    failed_state: ProjectState | None = None
    latest_error: Text | None = None

    @model_validator(mode="after")
    def consistent_snapshot(self) -> Self:
        if self.last_successful_state not in DURABLE_CHECKPOINTS:
            raise ValueError("Last successful state must be a durable checkpoint")
        if self.current_state == ProjectState.FAILED:
            if self.last_successful_state == ProjectState.COMPLETE:
                raise ValueError("FAILED requires a resumable checkpoint")
            if self.failed_state in (None, ProjectState.FAILED, ProjectState.COMPLETE):
                raise ValueError("FAILED requires an interrupted workflow state")
            if self.latest_error is None:
                raise ValueError("FAILED requires an error")
            effective_state = self.failed_state
        else:
            if self.failed_state is not None or self.latest_error is not None:
                raise ValueError("Active state must have no failure metadata")
            effective_state = self.current_state
        if effective_state in DURABLE_CHECKPOINTS:
            if self.last_successful_state != effective_state:
                raise ValueError("Checkpoint state must match last successful state")
        elif self.last_successful_state == ProjectState.COMPLETE:
            raise ValueError("A complete checkpoint cannot have an active stage")
        return self
