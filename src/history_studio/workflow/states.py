from enum import StrEnum
from typing import Self

from pydantic import Field, model_validator

from history_studio.models.base import Contract, Text
from history_studio.models.artifact_reference import ArtifactReference
from .artifacts import WorkflowArtifactBindings


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
    artifacts: WorkflowArtifactBindings = Field(default_factory=WorkflowArtifactBindings)

    @model_validator(mode="before")
    @classmethod
    def migrate_research_binding(cls, value):
        if isinstance(value, dict) and "research_input_ref" in value:
            value = dict(value)
            legacy = value.pop("research_input_ref")
            nested = value.get("artifacts", {})
            if isinstance(nested, WorkflowArtifactBindings):
                nested = nested.model_dump(mode="json")
            nested = dict(nested)
            if "research" in nested:
                def parse(reference):
                    return None if reference is None else ArtifactReference.model_validate(reference)
                if parse(nested["research"]) != parse(legacy):
                    raise ValueError("Conflicting legacy and typed research bindings")
            nested["research"] = legacy
            value["artifacts"] = nested
        return value

    @property
    def research_input_ref(self) -> ArtifactReference | None:
        """Read-only Python compatibility alias; serialization writes only artifacts."""
        return self.artifacts.research

    def require_research_input_ref(self, project_id: str) -> ArtifactReference:
        """Fail closed at project-scoped provenance boundaries; never infer a legacy version."""
        if self.research_input_ref is None:
            raise ValueError("Exact completed research snapshot binding is missing; explicit reconciliation required")
        self.artifacts.validate_project(project_id)
        reference = ArtifactReference.model_validate(self.artifacts.research.model_dump(mode="json"))
        if reference.artifact_type != "research":
            raise ValueError("Workflow research reference must identify a research artifact")
        if reference.project_id != project_id:
            raise ValueError("Research snapshot reference belongs to a different project")
        return reference

    def require_verification_ref(self, project_id: str) -> ArtifactReference:
        self.artifacts.validate_project(project_id)
        if self.artifacts.verification is None:
            raise ValueError("Exact completed verification snapshot binding is missing; explicit reconciliation required")
        return ArtifactReference.model_validate(self.artifacts.verification.model_dump(mode="json"))

    def require_approved_verification_ref(self, project_id: str) -> ArtifactReference:
        self.artifacts.validate_project(project_id)
        if self.artifacts.approved_verification is None:
            raise ValueError("Exact approved verification binding is missing")
        return ArtifactReference.model_validate(self.artifacts.approved_verification.model_dump(mode="json"))

    def require_story_ref(self, project_id: str) -> ArtifactReference:
        self.artifacts.validate_project(project_id)
        if self.artifacts.story is None:
            raise ValueError("Exact Story review binding is missing")
        return ArtifactReference.model_validate(self.artifacts.story.model_dump(mode="json"))

    def require_approved_story_ref(self, project_id: str) -> ArtifactReference:
        self.artifacts.validate_project(project_id)
        if self.artifacts.approved_story is None:
            raise ValueError("Exact approved Story binding is missing")
        return ArtifactReference.model_validate(self.artifacts.approved_story.model_dump(mode="json"))

    def require_script_ref(self, project_id: str) -> ArtifactReference:
        self.artifacts.validate_project(project_id)
        if self.artifacts.script is None:
            raise ValueError("Exact Script review binding is missing")
        return ArtifactReference.model_validate(self.artifacts.script.model_dump(mode="json"))

    def require_approved_script_ref(self, project_id: str) -> ArtifactReference:
        self.artifacts.validate_project(project_id)
        if self.artifacts.approved_script is None:
            raise ValueError("Exact approved Script binding is missing")
        return ArtifactReference.model_validate(self.artifacts.approved_script.model_dump(mode="json"))

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
