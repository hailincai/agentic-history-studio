"""Exact human Script review, with existing separate audit/state publication."""
import json

from history_studio.models import ScriptPackage
from history_studio.script import build_script_context, validate_script_grounding
from history_studio.storage.artifact_store import ArtifactStore, write_json
from .approvals import ApprovalRecord
from .states import RuntimeState, ProjectState as S
from .state_machine import ProjectStateMachine, InvalidTransitionError


def load_script_review(state: RuntimeState, store: ArtifactStore) -> ScriptPackage:
    """Reload bound Script and mechanically authenticate exact approved Story lineage."""
    if state.current_state != S.WAITING_SCRIPT_APPROVAL:
        raise InvalidTransitionError("Script Review requires WAITING_SCRIPT_APPROVAL")
    reference = state.require_script_ref(store.project_dir.name)
    approved = state.require_approved_story_ref(store.project_dir.name)
    package = store.load("script", reference.version, ScriptPackage)
    context = build_script_context(store, story_input_ref=approved)
    report = validate_script_grounding(context, package)
    if not report.is_valid:
        raise ValueError("Script review grounding validation failed: " + ",".join(
            issue.code.value for issue in report.issues))
    return package


def script_review_lines(state: RuntimeState, package: ScriptPackage) -> list[str]:
    """Escaped bounded display; exact persisted artifact remains available for full review.

    Human owns narration truth, qualification adequacy, chronology as narrated, and
    unsupported interpretation. These lines perform no semantic evaluation.
    """
    def bounded(value):
        text = str(value)
        return json.dumps(text if len(text) <= 1000 else text[:1000] + "… [truncated]", ensure_ascii=False)
    ref = state.artifacts.script
    provenance = package.story_input_ref
    lines = [f"Review {ref.project_id}/script:v{ref.version}",
        f"Story provenance: {provenance.project_id}/story:v{provenance.version}",
        f"Title: {bounded(package.title)}"]
    for section in package.sections:
        lines.append(f"Section {section.section_id}; title: {bounded(section.title)}")
        for segment in section.segments:
            lines.extend([f"Segment {segment.segment_id}; kind: {segment.kind}",
                          f"Narration: {bounded(segment.narration)}"])
            if segment.grounding is not None:
                lines.append(f"Story beat: {segment.grounding.story_beat_id}")
                lines.extend(f"Fact: {fact_id}" for fact_id in segment.grounding.research_fact_ids)
    return lines


def apply_script_review(path, record: ApprovalRecord) -> RuntimeState:
    """Audit precedes atomic binding/state publication under the single-writer convention.

    A crash between writes requires explicit reconciliation. No automatic audit replay,
    regeneration, semantic evaluation or approval inference occurs.
    """
    store = ArtifactStore(path)
    state_path = store.project_dir / ".runtime/state.json"
    state = RuntimeState.model_validate_json(state_path.read_text(encoding="utf-8"))
    machine = ProjectStateMachine(state)
    machine.apply_script_review_decision(record, store)
    write_json(state_path, machine.state, replace=True)
    return machine.state
