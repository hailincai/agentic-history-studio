from enum import StrEnum
from typing import Literal, Self

from pydantic import AwareDatetime, Field, model_validator

from history_studio.models.base import Contract, Identifier, Text


class ApprovalDecision(StrEnum):
    APPROVED = "APPROVED"
    REJECTED = "REJECTED"
    REVISION_REQUESTED = "REVISION_REQUESTED"


class ApprovalStage(StrEnum):
    FACTS = "facts"
    STORY = "story"
    SCRIPT = "script"
    STORYBOARD = "storyboard"


class ApprovalRecord(Contract):
    """External human attestation, never an agent-generated approval.

    The caller authenticates the human; this local contract is not an identity service.
    """

    model_config = {"frozen": True}
    project_id: Identifier
    stage: ApprovalStage
    artifact_type: ApprovalStage
    artifact_version: int = Field(gt=0, strict=True)
    decision: ApprovalDecision
    feedback: str = ""
    decided_at: AwareDatetime
    decided_by: Text
    decision_source: Literal["human"]

    @model_validator(mode="after")
    def matching_stage(self) -> Self:
        if self.stage != self.artifact_type:
            raise ValueError("Approval stage must match artifact type")
        return self
