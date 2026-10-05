"""One-writer Script workflow with exact bindings and regeneration after orphan publication."""
from collections.abc import Callable

from history_studio.model_io import ModelProvider
from history_studio.models import ArtifactReference, ProjectConfig, ScriptContext, ScriptPackage
from history_studio.models.base import Contract, Text
from history_studio.storage.artifact_store import ArtifactStore, write_json
from history_studio.script import (
    ScriptWriter, ScriptGenerationOutcome, ScriptGenerationStopReason, build_script_context,
    validate_script_grounding, ScriptGroundingValidationReport,
)
from .states import RuntimeState, ProjectState as S
from .state_machine import ProjectStateMachine, InvalidTransitionError


class ScriptWorkflowOutcome(Contract):
    state: RuntimeState
    stage: ScriptGenerationOutcome | None = None
    error_type: Text | None = None
    validation_report: ScriptGroundingValidationReport | None = None


class ScriptWorkflow:
    """Persisted artifact != workflow-bound artifact. Existence proves no lineage.

    Only durable exact RuntimeState references participate in workflow lineage.
    After publication but before binding, output is orphaned; recovery regenerates
    using durable input, never discovers/adopts output. Repeated model cost is possible.
    Binding and gate transition are published together in one atomic state write,
    following complete_verification. Artifact/state writes are not a transaction.
    """

    def __init__(self, *, provider_factory: Callable[[ScriptContext], ModelProvider]) -> None:
        self.provider_factory = provider_factory

    def run(self, project: ProjectConfig, store: ArtifactStore, *,
            max_steps: int = 4, max_output_tokens: int = 6000) -> ScriptWorkflowOutcome:
        if project.project_id != store.project_dir.name:
            raise ValueError("Project ID must match artifact directory")
        path = store.project_dir / ".runtime/state.json"
        state = RuntimeState.model_validate_json(path.read_text(encoding="utf-8"))
        if state.current_state == S.FAILED:
            if state.failed_state != S.SCRIPT_GENERATING:
                raise InvalidTransitionError("Only interrupted SCRIPT_GENERATING can resume Script")
        elif state.current_state not in (S.STORY_APPROVED, S.SCRIPT_GENERATING, S.WAITING_SCRIPT_APPROVAL):
            raise InvalidTransitionError("Script requires STORY_APPROVED or SCRIPT_GENERATING")
        reference = state.require_approved_story_ref(project.project_id)
        context = build_script_context(store, story_input_ref=reference)
        if state.current_state == S.WAITING_SCRIPT_APPROVAL:
            if state.artifacts.script is None:
                raise ValueError("Waiting Script requires exact output binding")
            package = store.load("script", state.artifacts.script.version, ScriptPackage)
            if package.story_input_ref != reference:
                raise ValueError("Bound Script must match approved Story")
            if not validate_script_grounding(context, package).is_valid:
                raise ValueError("Bound Script grounding is invalid")
            return ScriptWorkflowOutcome(state=state)
        machine = ProjectStateMachine(state)
        if state.current_state == S.FAILED:
            machine.recover()
        elif state.current_state == S.STORY_APPROVED:
            machine.transition(S.SCRIPT_GENERATING)
        write_json(path, machine.state, replace=True)
        stage = None
        report = None
        try:
            # Dependencies cannot mutate the authoritative input used by the Agent.
            provider = self.provider_factory(ScriptContext.model_validate(context.model_dump(mode="json")))
            stage = ScriptWriter(context, provider=provider).generate(
                max_steps=max_steps, max_output_tokens=max_output_tokens)
            stage = ScriptGenerationOutcome.model_validate(stage.model_dump(mode="json"))
            if stage.stop_reason != ScriptGenerationStopReason.SUBMITTED:
                machine.fail("script_limit_reached")
                write_json(path, machine.state, replace=True)
                return ScriptWorkflowOutcome(state=machine.state, stage=stage)
            package = ScriptPackage.model_validate(stage.package.model_dump(mode="json"))
            report = validate_script_grounding(context, package)
            if not report.is_valid:
                raise ValueError("Script grounding validation failed: " + ",".join(
                    issue.code.value for issue in report.issues))
            version = store.save("script", package)
            durable = store.load("script", version, ScriptPackage)
            if durable != package:
                raise ValueError("Published Script must equal accepted output")
            if not validate_script_grounding(context, durable).is_valid:
                raise ValueError("Published Script grounding is invalid")
            ready = ProjectStateMachine(machine.state)
            ready.complete_script(ArtifactReference(project_id=project.project_id,
                artifact_type="script", version=version), project_id=project.project_id)
            write_json(path, ready.state, replace=True)
            return ScriptWorkflowOutcome(state=ready.state, stage=stage)
        except Exception as exc:
            # Never bind a publication whose final state publication failed.
            if machine.state.current_state != S.FAILED:
                machine.fail(f"script_execution_failed:{type(exc).__name__[:120]}")
            write_json(path, machine.state, replace=True)
            return ScriptWorkflowOutcome(state=machine.state, stage=stage,
                error_type=type(exc).__name__[:120], validation_report=report)
