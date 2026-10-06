"""One-writer Storyboard workflow with exact bindings and regeneration after orphan publication."""
from collections.abc import Callable

from history_studio.model_io import ModelProvider
from history_studio.models import ArtifactReference, ProjectConfig, VisualDirectorContext, StoryboardPackage
from history_studio.models.base import Contract, Text
from history_studio.storage.artifact_store import ArtifactStore, write_json
from history_studio.visual_director import (
    VisualDirector, VisualDirectorGenerationOutcome, VisualDirectorGenerationStopReason, build_visual_director_context,
    validate_storyboard_integrity, StoryboardIntegrityReport,
)
from .states import RuntimeState, ProjectState as S
from .state_machine import ProjectStateMachine, InvalidTransitionError


class StoryboardWorkflowOutcome(Contract):
    state: RuntimeState
    stage: VisualDirectorGenerationOutcome | None = None
    error_type: Text | None = None
    validation_report: StoryboardIntegrityReport | None = None


class StoryboardWorkflow:
    """Persisted artifact != workflow-bound artifact. Existence proves no lineage.

    Only durable exact RuntimeState references participate in workflow lineage.
    After publication but before binding, output is orphaned; recovery regenerates
    using durable input, never discovers/adopts output. Repeated model cost is possible.
    Binding and gate transition are published together in one atomic state write,
    following complete_verification. Artifact/state writes are not a transaction.
    """

    def __init__(self, *, provider_factory: Callable[[VisualDirectorContext], ModelProvider]) -> None:
        self.provider_factory = provider_factory

    def run(self, project: ProjectConfig, store: ArtifactStore, *,
            max_steps: int = 4, max_output_tokens: int = 6000) -> StoryboardWorkflowOutcome:
        if project.project_id != store.project_dir.name:
            raise ValueError("Project ID must match artifact directory")
        path = store.project_dir / ".runtime/state.json"
        state = RuntimeState.model_validate_json(path.read_text(encoding="utf-8"))
        if state.current_state == S.FAILED:
            if state.failed_state != S.STORYBOARD_GENERATING:
                raise InvalidTransitionError("Only interrupted STORYBOARD_GENERATING can resume Storyboard")
        elif state.current_state not in (S.SCRIPT_APPROVED, S.STORYBOARD_GENERATING, S.WAITING_STORYBOARD_APPROVAL):
            raise InvalidTransitionError("Storyboard requires SCRIPT_APPROVED or STORYBOARD_GENERATING")
        reference = state.require_approved_script_ref(project.project_id)
        context = build_visual_director_context(store, script_input_ref=reference)
        if state.current_state == S.WAITING_STORYBOARD_APPROVAL:
            if state.artifacts.storyboard is None:
                raise ValueError("Waiting Storyboard requires exact output binding")
            package = store.load("storyboard", state.artifacts.storyboard.version, StoryboardPackage)
            if package.script_input_ref != reference:
                raise ValueError("Bound Storyboard must match approved Script")
            if not validate_storyboard_integrity(context, package).is_valid:
                raise ValueError("Bound Storyboard integrity is invalid")
            return StoryboardWorkflowOutcome(state=state)
        machine = ProjectStateMachine(state)
        if state.current_state == S.FAILED:
            machine.recover()
        elif state.current_state == S.SCRIPT_APPROVED:
            machine.transition(S.STORYBOARD_GENERATING)
        write_json(path, machine.state, replace=True)
        stage = None
        report = None
        try:
            # Dependencies cannot mutate the authoritative input used by the Agent.
            provider = self.provider_factory(VisualDirectorContext.model_validate(context.model_dump(mode="json")))
            stage = VisualDirector(context, provider=provider).generate(
                max_steps=max_steps, max_output_tokens=max_output_tokens)
            stage = VisualDirectorGenerationOutcome.model_validate(stage.model_dump(mode="json"))
            if stage.stop_reason != VisualDirectorGenerationStopReason.SUBMITTED:
                machine.fail("storyboard_limit_reached")
                write_json(path, machine.state, replace=True)
                return StoryboardWorkflowOutcome(state=machine.state, stage=stage)
            package = StoryboardPackage.model_validate(stage.package.model_dump(mode="json"))
            report = validate_storyboard_integrity(context, package)
            if not report.is_valid:
                raise ValueError("Storyboard integrity validation failed: " + ",".join(
                    issue.code.value for issue in report.issues))
            version = store.save("storyboard", package)
            durable = store.load("storyboard", version, StoryboardPackage)
            if durable != package:
                raise ValueError("Published Storyboard must equal accepted output")
            if not validate_storyboard_integrity(context, durable).is_valid:
                raise ValueError("Published Storyboard integrity is invalid")
            ready = ProjectStateMachine(machine.state)
            ready.complete_storyboard(ArtifactReference(project_id=project.project_id,
                artifact_type="storyboard", version=version), project_id=project.project_id)
            write_json(path, ready.state, replace=True)
            return StoryboardWorkflowOutcome(state=ready.state, stage=stage)
        except Exception as exc:
            # Never bind a publication whose final state publication failed.
            if machine.state.current_state != S.FAILED:
                machine.fail(f"storyboard_execution_failed:{type(exc).__name__[:120]}")
            write_json(path, machine.state, replace=True)
            return StoryboardWorkflowOutcome(state=machine.state, stage=stage,
                error_type=type(exc).__name__[:120], validation_report=report)
