"""Exact Human Storyboard review; Runtime checks structure, Human accepts semantics."""
import json

from history_studio.models import StoryboardPackage, VisualDirectorContext
from history_studio.visual_director import build_visual_director_context, validate_storyboard_integrity
from history_studio.storage.artifact_store import ArtifactStore, write_json
from .approvals import ApprovalRecord
from .states import RuntimeState, ProjectState as S
from .state_machine import ProjectStateMachine, InvalidTransitionError


def load_storyboard_review(state: RuntimeState, store: ArtifactStore) -> StoryboardPackage:
    """Reload only the bound candidate and its exact workflow-approved Script authority."""
    if state.current_state != S.WAITING_STORYBOARD_APPROVAL:
        raise InvalidTransitionError("Storyboard Review requires WAITING_STORYBOARD_APPROVAL")
    reference = state.require_storyboard_ref(store.project_dir.name)
    approved = state.require_approved_script_ref(store.project_dir.name)
    package = store.load("storyboard", reference.version, StoryboardPackage)
    context = build_visual_director_context(store, script_input_ref=approved)
    report = validate_storyboard_integrity(context, package)
    if not report.is_valid:
        raise ValueError("Storyboard review integrity validation failed: " + ",".join(
            issue.code.value for issue in report.issues))
    return package


def storyboard_review_lines(state: RuntimeState, package: StoryboardPackage,
                            context: VisualDirectorContext) -> list[str]:
    """Escaped bounded display including source narration; no semantic evaluation.

    Caller supplies exact loaded context. Human judges unsupported events/people,
    relationships, misleading certainty, prompt fit, treatment and decomposition.
    Full persisted artifacts remain available for untruncated inspection.
    """
    if state.current_state != S.WAITING_STORYBOARD_APPROVAL:
        raise InvalidTransitionError("Storyboard Review requires WAITING_STORYBOARD_APPROVAL")
    ref = state.require_storyboard_ref(context.script_input_ref.project_id)
    approved = state.require_approved_script_ref(ref.project_id)
    if context.script_input_ref != approved or not validate_storyboard_integrity(context, package).is_valid:
        raise ValueError("Review display requires exact approved Script authority and valid Storyboard integrity")

    def bounded(value):
        text = str(value)
        return json.dumps(text if len(text) <= 1000 else text[:1000] + "… [truncated]", ensure_ascii=False)

    segments = {segment.segment_id: segment for section in context.sections for segment in section.segments}
    lines = [f"Review {ref.project_id}/storyboard:v{ref.version}",
        f"Script provenance: {approved.project_id}/script:v{approved.version}",
        f"Title: {bounded(package.title)}",
        "Human review owns historical implications, uncertainty, prompt fit, visual treatment and shot decomposition."]
    for section in package.sections:
        lines.append(f"Section {section.section_id}; title: {bounded(section.title)}")
        for shot in section.shots:
            source = segments[shot.source_segment_id]
            lines.extend([f"Source segment {source.segment_id}; kind: {source.kind}",
                          f"Narration: {bounded(source.narration)}"])
            if source.grounding is not None:
                lines.append(f"Story beat: {source.grounding.story_beat_id}")
                lines.extend(f"Fact: {fact_id}" for fact_id in source.grounding.research_fact_ids)
            lines.extend([f"Shot {shot.shot_id}; kind: {shot.kind}",
                f"Visual description: {bounded(shot.visual_description)}",
                f"Generation prompt: {bounded(shot.generation_prompt)}",
                f"Generation method: {shot.generation_method}; framing: {shot.framing}; camera motion: {shot.camera_motion}",
                f"Estimated duration seconds (planning only): {shot.estimated_duration_seconds}"])
    return lines


def apply_storyboard_review(path, record: ApprovalRecord) -> RuntimeState:
    """Single-writer audit precedes atomic approval binding/state publication.

    A crash between writes requires explicit reconciliation, as in existing gates.
    No automatic audit replay, semantic judgment or media generation occurs.
    """
    store = ArtifactStore(path)
    state_path = store.project_dir / ".runtime/state.json"
    state = RuntimeState.model_validate_json(state_path.read_text(encoding="utf-8"))
    machine = ProjectStateMachine(state)
    machine.apply_storyboard_review_decision(record, store)
    write_json(state_path, machine.state, replace=True)
    return machine.state
