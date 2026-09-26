from typing import Annotated

from pydantic import Field, RootModel

from history_studio.models import Script, Storyboard, StoryPlan, VerifiedFact
from history_studio.storage.artifact_store import ArtifactStore

from .approvals import ApprovalDecision, ApprovalRecord, ApprovalStage
from .states import DURABLE_CHECKPOINTS, ProjectState as S, RuntimeState


class InvalidTransitionError(ValueError):
    """A transition would violate workflow order or a human approval gate."""


class ProjectStateMachine:
    """Explicit workflow order. Failure recovery returns to the interrupted state.

    Normal transitions are orchestration primitives; future runners must persist
    their stage artifacts before calling them. Approval gates check persisted JSON.
    """

    def __init__(self, state: RuntimeState | None = None) -> None:
        self._state = state or RuntimeState()

    @property
    def state(self) -> RuntimeState:
        return self._state

    @property
    def resume_state(self) -> S:
        return self.state.failed_state or self.state.current_state

    def transition(self, target: S) -> RuntimeState:
        allowed = {
            S.CREATED: S.RESEARCHING,
            S.RESEARCHING: S.RESEARCH_COMPLETE,
            S.RESEARCH_COMPLETE: S.FACT_CHECKING,
            S.FACT_CHECKING: S.WAITING_FACT_APPROVAL,
            S.FACTS_APPROVED: S.STORY_GENERATING,
            S.STORY_GENERATING: S.WAITING_STORY_APPROVAL,
            S.STORY_APPROVED: S.SCRIPT_GENERATING,
            S.SCRIPT_GENERATING: S.WAITING_SCRIPT_APPROVAL,
            S.SCRIPT_APPROVED: S.STORYBOARD_GENERATING,
            S.STORYBOARD_GENERATING: S.WAITING_STORYBOARD_APPROVAL,
            S.STORYBOARD_APPROVED: S.GENERATING_MEDIA,
            S.GENERATING_MEDIA: S.ASSEMBLING,
            S.ASSEMBLING: S.COMPLETE,
        }
        if allowed.get(self.state.current_state) != target:
            raise InvalidTransitionError(f"Cannot transition {self.state.current_state} -> {target}")
        return self._set_state(target)

    def fail(self, error: str) -> RuntimeState:
        if self.state.current_state in (S.COMPLETE, S.FAILED):
            raise InvalidTransitionError("Cannot fail a complete or already failed workflow")
        self._state = RuntimeState(current_state=S.FAILED,
                                   last_successful_state=self.state.last_successful_state,
                                   failed_state=self.state.current_state,
                                   latest_error=error)
        return self.state

    def recover(self) -> RuntimeState:
        if self.state.current_state != S.FAILED:
            raise InvalidTransitionError("Only a failed workflow can recover")
        return self._set_state(self.resume_state)

    def apply_human_decision(self, record: ApprovalRecord, store: ArtifactStore) -> RuntimeState:
        gates = {
            ApprovalStage.FACTS: (S.WAITING_FACT_APPROVAL, S.FACTS_APPROVED, S.FACT_CHECKING),
            ApprovalStage.STORY: (S.WAITING_STORY_APPROVAL, S.STORY_APPROVED, S.STORY_GENERATING),
            ApprovalStage.SCRIPT: (S.WAITING_SCRIPT_APPROVAL, S.SCRIPT_APPROVED, S.SCRIPT_GENERATING),
            ApprovalStage.STORYBOARD: (S.WAITING_STORYBOARD_APPROVAL, S.STORYBOARD_APPROVED,
                                       S.STORYBOARD_GENERATING),
        }
        waiting, approved, revision = gates[record.stage]
        versions = store.list_versions(record.artifact_type)
        if (self.state.current_state != waiting or record.project_id != store.project_dir.name
                or not versions or record.artifact_version != versions[-1]):
            raise InvalidTransitionError("Human decision must match this project's current gate and latest artifact")
        contracts = {
            ApprovalStage.FACTS: RootModel[VerifiedFact | Annotated[list[VerifiedFact], Field(min_length=1)]],
            ApprovalStage.STORY: StoryPlan,
            ApprovalStage.SCRIPT: Script,
            ApprovalStage.STORYBOARD: Storyboard,
        }
        store.load(record.artifact_type, record.artifact_version, contracts[record.stage])
        for version in store.list_versions("approvals"):
            previous = store.load("approvals", version, ApprovalRecord)
            if (previous.project_id == record.project_id and previous.stage == record.stage
                    and previous.artifact_version == record.artifact_version):
                raise InvalidTransitionError("This artifact version already has a decision; persist a new version")
        store.save("approvals", record)
        return self._set_state(approved if record.decision == ApprovalDecision.APPROVED else revision)

    def _set_state(self, state: S) -> RuntimeState:
        checkpoint = state if state in DURABLE_CHECKPOINTS else self.state.last_successful_state
        self._state = RuntimeState(current_state=state, last_successful_state=checkpoint)
        return self.state
