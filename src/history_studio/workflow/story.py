"""One-writer Story workflow with exact bindings and regeneration after orphan publication."""
from collections.abc import Callable

from history_studio.model_io import ModelProvider
from history_studio.models import ArtifactReference, ProjectConfig, StoryContext, StoryPackage
from history_studio.models.base import Contract, Text
from history_studio.models.production_brief import load_production_brief
from history_studio.storage.artifact_store import ArtifactStore, write_json
from history_studio.story import StoryArchitect, StoryGenerationOutcome, StoryGenerationStopReason, build_story_context
from .states import RuntimeState, ProjectState as S
from .state_machine import ProjectStateMachine, InvalidTransitionError


class StoryWorkflowOutcome(Contract):
    state: RuntimeState
    stage: StoryGenerationOutcome | None = None
    error_type: Text | None = None


class StoryWorkflow:
    """Persisted artifact != workflow-bound artifact. Existence proves no lineage.

    Only durable exact RuntimeState references participate in workflow lineage.
    After publication but before binding, output is orphaned; recovery regenerates
    using durable input, never discovers/adopts output. Repeated model cost is possible.
    Binding and gate transition are published together in one atomic state write,
    following complete_verification. Artifact/state writes are not a transaction.
    """

    def __init__(self, *, provider_factory: Callable[[StoryContext], ModelProvider]) -> None:
        self.provider_factory = provider_factory

    def run(self, project: ProjectConfig, store: ArtifactStore, *,
            max_steps: int = 4, max_output_tokens: int = 3000) -> StoryWorkflowOutcome:
        if project.project_id != store.project_dir.name:
            raise ValueError("Project ID must match artifact directory")
        path = store.project_dir / ".runtime/state.json"
        state = RuntimeState.model_validate_json(path.read_text(encoding="utf-8"))
        if state.current_state == S.FAILED:
            if state.failed_state != S.STORY_GENERATING:
                raise InvalidTransitionError("Only interrupted STORY_GENERATING can resume Story")
        elif state.current_state not in (S.FACTS_APPROVED, S.STORY_GENERATING, S.WAITING_STORY_APPROVAL):
            raise InvalidTransitionError("Story requires FACTS_APPROVED or STORY_GENERATING")
        reference = state.require_approved_verification_ref(project.project_id)
        context = build_story_context(store, verification_input_ref=reference)
        if state.current_state == S.WAITING_STORY_APPROVAL:
            if state.artifacts.story is None:
                raise ValueError("Waiting Story requires exact output binding")
            package = store.load("story", state.artifacts.story.version, StoryPackage)
            if package.verification_input_ref != reference:
                raise ValueError("Bound Story must match approved verification")
            return StoryWorkflowOutcome(state=state)
        machine = ProjectStateMachine(state)
        context.production_brief = load_production_brief(store.project_dir, project_id=project.project_id)
        if state.current_state == S.FAILED:
            machine.recover()
        elif state.current_state == S.FACTS_APPROVED:
            machine.transition(S.STORY_GENERATING)
        write_json(path, machine.state, replace=True)
        stage = None
        try:
            # Dependencies cannot mutate the authoritative input used by the Agent.
            provider = self.provider_factory(StoryContext.model_validate(context.model_dump(mode="json")))
            stage = StoryArchitect(context, provider=provider).generate(
                max_steps=max_steps, max_output_tokens=max_output_tokens)
            stage = StoryGenerationOutcome.model_validate(stage.model_dump(mode="json"))
            if stage.stop_reason != StoryGenerationStopReason.SUBMITTED:
                machine.fail("story_limit_reached")
                write_json(path, machine.state, replace=True)
                return StoryWorkflowOutcome(state=machine.state, stage=stage)
            package = StoryPackage.model_validate(stage.package.model_dump(mode="json"))
            if package.verification_input_ref != reference:
                raise ValueError("Story output must match exact approved verification")
            version = store.save("story", package)
            durable = store.load("story", version, StoryPackage)
            if durable != package:
                raise ValueError("Published Story must equal accepted output")
            ready = ProjectStateMachine(machine.state)
            ready.complete_story(ArtifactReference(project_id=project.project_id,
                artifact_type="story", version=version), project_id=project.project_id)
            write_json(path, ready.state, replace=True)
            return StoryWorkflowOutcome(state=ready.state, stage=stage)
        except Exception as exc:
            # Never bind a publication whose final state publication failed.
            if machine.state.current_state != S.FAILED:
                machine.fail(f"story_execution_failed:{type(exc).__name__[:120]}")
            write_json(path, machine.state, replace=True)
            return StoryWorkflowOutcome(state=machine.state, stage=stage, error_type=type(exc).__name__[:120])
