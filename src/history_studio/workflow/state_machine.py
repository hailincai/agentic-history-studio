from typing import Annotated

from pydantic import Field, RootModel

from history_studio.models import ArtifactReference, Script, Storyboard, StoryPlan, VerifiedFact
from history_studio.storage.artifact_store import ArtifactStore

from .approvals import ApprovalDecision, ApprovalRecord, ApprovalStage
from .states import DURABLE_CHECKPOINTS, ProjectState as S, RuntimeState
from .artifacts import WorkflowArtifactBindings


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
                                   artifacts=self.state.artifacts,
                                   latest_error=error)
        return self.state

    def complete_research(self, reference: ArtifactReference, *, project_id: str) -> RuntimeState:
        """Bind Runtime's successfully published artifact only at research completion."""
        if self.state.current_state != S.RESEARCHING:
            raise InvalidTransitionError("Research binding requires the active RESEARCHING stage")
        reference = ArtifactReference.model_validate(reference.model_dump(mode="json"))
        if reference.project_id != project_id or reference.artifact_type != "research":
            raise ValueError("Research reference must identify this project's research artifact")
        self.state.artifacts.validate_project(project_id)
        return self._set_state(S.RESEARCH_COMPLETE, artifacts=self.state.artifacts.with_research(reference))

    def complete_verification(self, reference: ArtifactReference, *, project_id: str) -> RuntimeState:
        """Caller reloads the exact completed package before publishing this gate snapshot."""
        if self.state.current_state != S.FACT_CHECKING:
            raise InvalidTransitionError("Verification binding requires the active FACT_CHECKING stage")
        reference = ArtifactReference.model_validate(reference.model_dump(mode="json"))
        if reference.project_id != project_id or reference.artifact_type != "verification":
            raise ValueError("Verification reference must identify this project's verification artifact")
        self.state.require_research_input_ref(project_id)
        return self._set_state(S.WAITING_FACT_APPROVAL,
                               artifacts=self.state.artifacts.with_verification(reference))

    def recover(self) -> RuntimeState:
        if self.state.current_state != S.FAILED:
            raise InvalidTransitionError("Only a failed workflow can recover")
        return self._set_state(self.resume_state)

    def complete_story(self, reference: ArtifactReference, *, project_id: str) -> RuntimeState:
        """Caller publishes/reloads exact Story output before this atomic gate snapshot."""
        if self.state.current_state != S.STORY_GENERATING:
            raise InvalidTransitionError("Story binding requires active STORY_GENERATING")
        reference = ArtifactReference.model_validate(reference.model_dump(mode="json"))
        if reference.project_id != project_id or reference.artifact_type != "story":
            raise ValueError("Story reference must identify this project's story artifact")
        self.state.require_approved_verification_ref(project_id)
        return self._set_state(S.WAITING_STORY_APPROVAL, artifacts=self.state.artifacts.with_story(reference))

    def complete_script(self, reference: ArtifactReference, *, project_id: str) -> RuntimeState:
        """Caller validates and reloads exact Script output before atomic gate publication."""
        if self.state.current_state != S.SCRIPT_GENERATING:
            raise InvalidTransitionError("Script binding requires active SCRIPT_GENERATING")
        reference = ArtifactReference.model_validate(reference.model_dump(mode="json"))
        if reference.project_id != project_id or reference.artifact_type != "script":
            raise ValueError("Script reference must identify this project's script artifact")
        self.state.require_approved_story_ref(project_id)
        return self._set_state(S.WAITING_SCRIPT_APPROVAL, artifacts=self.state.artifacts.with_script(reference))

    def apply_human_decision(self, record: ApprovalRecord, store: ArtifactStore) -> RuntimeState:
        if record.stage == ApprovalStage.FACTS:
            return self.apply_fact_review_decision(record, store)
        if record.stage == ApprovalStage.STORY:
            return self.apply_story_review_decision(record, store)
        if record.stage == ApprovalStage.SCRIPT:
            return self.apply_script_review_decision(record, store)
        if record.stage == ApprovalStage.STORYBOARD:
            return self.apply_storyboard_review_decision(record, store)
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

    def complete_storyboard(self, reference: ArtifactReference, *, project_id: str) -> RuntimeState:
        """Caller validates and reloads exact Storyboard before atomic gate publication."""
        if self.state.current_state != S.STORYBOARD_GENERATING:
            raise InvalidTransitionError("Storyboard binding requires active STORYBOARD_GENERATING")
        reference = ArtifactReference.model_validate(reference.model_dump(mode="json"))
        if reference.project_id != project_id or reference.artifact_type != "storyboard":
            raise ValueError("Storyboard reference must identify this project's storyboard artifact")
        self.state.require_approved_script_ref(project_id)
        return self._set_state(S.WAITING_STORYBOARD_APPROVAL,
                               artifacts=self.state.artifacts.with_storyboard(reference))

    def apply_fact_review_decision(self, record: ApprovalRecord, store: ArtifactStore) -> RuntimeState:
        """Only the human-attested, exact bound VerificationPackage can be approved."""
        from .fact_review import load_fact_review
        record = ApprovalRecord.model_validate(record.model_dump(mode="json"))
        load_fact_review(self.state, store)
        reference = self.state.require_verification_ref(store.project_dir.name)
        if (record.stage != ApprovalStage.FACTS or record.artifact_type != "verification"
                or record.project_id != reference.project_id or record.artifact_version != reference.version):
            raise InvalidTransitionError("Human decision must identify the exact bound verification artifact")
        for version in store.list_versions("approvals"):
            previous = store.load("approvals", version, ApprovalRecord)
            if (previous.project_id == record.project_id and previous.stage == record.stage
                    and previous.artifact_type == record.artifact_type
                    and previous.artifact_version == record.artifact_version):
                raise InvalidTransitionError("This artifact version already has a decision; persist a new version")
        data = self.state.artifacts.model_dump(mode="json")
        data["approved_verification"] = (reference.model_dump(mode="json")
            if record.decision == ApprovalDecision.APPROVED else None)
        bindings = WorkflowArtifactBindings.model_validate(data)
        store.save("approvals", record)
        return self._set_state(S.FACTS_APPROVED if record.decision == ApprovalDecision.APPROVED
                               else S.FACT_CHECKING, artifacts=bindings)

    def _set_state(self, state: S, *, artifacts: WorkflowArtifactBindings | None = None) -> RuntimeState:
        checkpoint = state if state in DURABLE_CHECKPOINTS else self.state.last_successful_state
        self._state = RuntimeState(current_state=state, last_successful_state=checkpoint,
            artifacts=artifacts if artifacts is not None else self.state.artifacts)
        return self.state

    def apply_story_review_decision(self, record: ApprovalRecord, store: ArtifactStore) -> RuntimeState:
        """Human acceptance of exactly the bound StoryPackage, never a latest artifact."""
        from .story_review import load_story_review
        record = ApprovalRecord.model_validate(record.model_dump(mode="json"))
        load_story_review(self.state, store)
        reference = self.state.require_story_ref(store.project_dir.name)
        if (record.stage != ApprovalStage.STORY or record.artifact_type != "story"
                or record.project_id != reference.project_id or record.artifact_version != reference.version):
            raise InvalidTransitionError("Human decision must identify the exact bound Story artifact")
        for version in store.list_versions("approvals"):
            previous = store.load("approvals", version, ApprovalRecord)
            if (previous.project_id == record.project_id and previous.stage == record.stage
                    and previous.artifact_type == record.artifact_type
                    and previous.artifact_version == record.artifact_version):
                raise InvalidTransitionError("This Story artifact already has a human decision")
        bindings = self.state.artifacts.without_story_approval()
        if record.decision == ApprovalDecision.APPROVED:
            bindings = WorkflowArtifactBindings.model_validate(bindings.model_dump(mode="json") | {
                "approved_story": reference.model_dump(mode="json")})
        store.save("approvals", record)
        return self._set_state(S.STORY_APPROVED if record.decision == ApprovalDecision.APPROVED
                               else S.STORY_GENERATING, artifacts=bindings)

    def apply_script_review_decision(self, record: ApprovalRecord, store: ArtifactStore) -> RuntimeState:
        """Human acceptance of exactly the bound ScriptPackage, never a latest artifact."""
        from .script_review import load_script_review
        record = ApprovalRecord.model_validate(record.model_dump(mode="json"))
        load_script_review(self.state, store)
        reference = self.state.require_script_ref(store.project_dir.name)
        if (record.stage != ApprovalStage.SCRIPT or record.artifact_type != "script"
                or record.project_id != reference.project_id or record.artifact_version != reference.version):
            raise InvalidTransitionError("Human decision must identify the exact bound Script artifact")
        for version in store.list_versions("approvals"):
            previous = store.load("approvals", version, ApprovalRecord)
            if (previous.project_id == record.project_id and previous.stage == record.stage
                    and previous.artifact_type == record.artifact_type
                    and previous.artifact_version == record.artifact_version):
                raise InvalidTransitionError("This Script artifact already has a human decision")
        bindings = self.state.artifacts.without_script_approval()
        if record.decision == ApprovalDecision.APPROVED:
            bindings = WorkflowArtifactBindings.model_validate(bindings.model_dump(mode="json") | {
                "approved_script": reference.model_dump(mode="json")})
        store.save("approvals", record)
        return self._set_state(S.SCRIPT_APPROVED if record.decision == ApprovalDecision.APPROVED
                               else S.SCRIPT_GENERATING, artifacts=bindings)

    def apply_storyboard_review_decision(self, record: ApprovalRecord, store: ArtifactStore) -> RuntimeState:
        """Human acceptance of exactly the bound StoryboardPackage, never latest."""
        from .storyboard_review import load_storyboard_review
        record = ApprovalRecord.model_validate(record.model_dump(mode="json"))
        load_storyboard_review(self.state, store)
        reference = self.state.require_storyboard_ref(store.project_dir.name)
        if (record.stage != ApprovalStage.STORYBOARD or record.artifact_type != "storyboard"
                or record.project_id != reference.project_id or record.artifact_version != reference.version):
            raise InvalidTransitionError("Human decision must identify the exact bound Storyboard artifact")
        for version in store.list_versions("approvals"):
            previous = store.load("approvals", version, ApprovalRecord)
            if (previous.project_id == record.project_id and previous.stage == record.stage
                    and previous.artifact_type == record.artifact_type
                    and previous.artifact_version == record.artifact_version):
                raise InvalidTransitionError("This Storyboard artifact already has a human decision")
        bindings = self.state.artifacts.without_storyboard_approval()
        if record.decision == ApprovalDecision.APPROVED:
            bindings = WorkflowArtifactBindings.model_validate(bindings.model_dump(mode="json") | {
                "approved_storyboard": reference.model_dump(mode="json")})
        store.save("approvals", record)
        return self._set_state(S.STORYBOARD_APPROVED if record.decision == ApprovalDecision.APPROVED
                               else S.STORYBOARD_GENERATING, artifacts=bindings)
