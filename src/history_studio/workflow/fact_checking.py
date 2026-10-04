"""Workflow position around the existing completed-fact runner; no verdict policy."""
from pydantic import Field

from history_studio.models import ProjectConfig, VerificationPackage, create_verification_package
from history_studio.models.base import Contract, Text
from history_studio.models.research_package import ResearchPackage, ResearchRunStatus
from history_studio.storage.artifact_store import ArtifactStore, write_json
from history_studio.verification import FactCheckingRunner, FactCheckingOutcome, FactCheckingStopReason
from .states import RuntimeState, ProjectState as S
from .state_machine import ProjectStateMachine, InvalidTransitionError


class FactCheckingWorkflowOutcome(Contract):
    state: RuntimeState
    stage: FactCheckingOutcome | None = None
    error_type: Text | None = Field(default=None, max_length=120)


class FactCheckingWorkflow:
    """One local writer; accepted artifact checkpoints survive all workflow failures.

    Input/state validation fails before entry. Execution errors use existing FAILED
    recovery. State-write errors propagate if even recording failure is impossible.
    The read-only CLI resume command remains informational; verify invokes re-entry.
    """

    def __init__(self, runner: FactCheckingRunner) -> None:
        self.runner = runner

    def run(self, project: ProjectConfig, store: ArtifactStore, *,
            max_steps: int = 4, max_output_tokens: int = 3000) -> FactCheckingWorkflowOutcome:
        if project.project_id != store.project_dir.name:
            raise ValueError("Project ID must match the artifact directory")
        path = store.project_dir / ".runtime" / "state.json"
        state = RuntimeState.model_validate_json(path.read_text(encoding="utf-8"))
        machine = ProjectStateMachine(state)
        if state.current_state == S.FAILED:
            if state.failed_state != S.FACT_CHECKING:
                raise InvalidTransitionError("Only interrupted FACT_CHECKING can resume verification")
        elif state.current_state not in (S.RESEARCH_COMPLETE, S.FACT_CHECKING, S.WAITING_FACT_APPROVAL):
            raise InvalidTransitionError("Fact Checking requires RESEARCH_COMPLETE or FACT_CHECKING")
        reference = state.require_research_input_ref(project.project_id)
        if state.current_state == S.WAITING_FACT_APPROVAL:
            return FactCheckingWorkflowOutcome(state=state)
        research = store.load("research", reference.version, ResearchPackage)
        if (research.project_id != project.project_id or research.topic != project.topic
                or research.research_scope != project.research_scope
                or research.progress.status != ResearchRunStatus.COMPLETE):
            raise ValueError("Bound research must match this project's completed research snapshot")
        membership = create_verification_package(research, research_input_ref=reference).research_facts
        if state.current_state == S.FAILED:
            machine.recover()
        elif state.current_state == S.RESEARCH_COMPLETE:
            machine.transition(S.FACT_CHECKING)
        # No external execution until active workflow position is durable.
        write_json(path, machine.state, replace=True)
        stage = None
        try:
            stage = self.runner.run(store, research_input_ref=reference,
                                    max_steps=max_steps, max_output_tokens=max_output_tokens)
            if stage.stop_reason != FactCheckingStopReason.COMPLETE or not stage.package.is_complete:
                machine.fail(f"fact_checking_{stage.stop_reason.value.lower()}")
                write_json(path, machine.state, replace=True)
                return FactCheckingWorkflowOutcome(state=machine.state, stage=stage)
            if stage.package.research_input_ref != reference or stage.package.research_facts != membership:
                raise ValueError("Completed verification must match the bound research membership")
            if stage.verification_version is None:
                raise ValueError("Complete verification requires a persisted checkpoint")
            durable = store.load("verification", stage.verification_version, VerificationPackage)
            if durable != stage.package or not durable.is_complete:
                raise ValueError("Human Gate requires the complete durable verification checkpoint")
            ready = ProjectStateMachine(machine.state)
            ready.transition(S.WAITING_FACT_APPROVAL)
            write_json(path, ready.state, replace=True)
            return FactCheckingWorkflowOutcome(state=ready.state, stage=stage)
        except Exception as exc:
            # Keep the interrupted stage FACT_CHECKING even if the final state write failed.
            if machine.state.current_state != S.FAILED:
                machine.fail(f"fact_checking_execution_failed:{type(exc).__name__[:120]}")
            write_json(path, machine.state, replace=True)
            return FactCheckingWorkflowOutcome(state=machine.state, stage=stage,
                                               error_type=type(exc).__name__[:120])
