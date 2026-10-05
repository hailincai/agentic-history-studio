"""Exact human Story review, using the existing audit/state publication convention."""
import json

from history_studio.models import StoryPackage
from history_studio.story import build_story_context
from history_studio.storage.artifact_store import ArtifactStore, write_json
from .approvals import ApprovalRecord
from .states import RuntimeState, ProjectState as S
from .state_machine import ProjectStateMachine, InvalidTransitionError


def load_story_review(state: RuntimeState, store: ArtifactStore) -> StoryPackage:
    """Load only bound Story and authenticate its grounding/chronology against lineage."""
    if state.current_state != S.WAITING_STORY_APPROVAL:
        raise InvalidTransitionError("Story Review requires WAITING_STORY_APPROVAL")
    reference = state.require_story_ref(store.project_dir.name)
    approved = state.require_approved_verification_ref(store.project_dir.name)
    package = store.load("story", reference.version, StoryPackage)
    if package.verification_input_ref != approved:
        raise ValueError("Story provenance must exactly match approved verification")
    context = build_story_context(store, verification_input_ref=approved)
    for section in package.plan.sections:
        for beat in section.beats:
            for fact in beat.fact_refs:
                context.validate_fact_reference(fact)
            for entry in beat.fact_chronology:
                if entry.historical_time != context.fact(entry.research_fact_id).historical_time:
                    raise ValueError("Story chronology must exactly match authoritative context")
    return package


def story_review_lines(state: RuntimeState, package: StoryPackage) -> list[str]:
    """Escaped, bounded values; every section/beat/reference is represented.

    Full exact artifact remains available for inspection. Human review owns prose
    correctness, narrative quality, qualification adequacy and chronology as presented.
    """
    def bounded(value):
        text = str(value)
        return json.dumps(text if len(text) <= 1000 else text[:1000] + "… [truncated]", ensure_ascii=False)
    ref = state.artifacts.story
    provenance = package.verification_input_ref
    lines = [f"Review {ref.project_id}/story:v{ref.version}",
        f"Verification provenance: {provenance.project_id}/verification:v{provenance.version}",
        f"Title: {bounded(package.plan.title)}", f"Thesis: {bounded(package.plan.narrative_thesis)}"]
    for section in package.plan.sections:
        lines.append(f"Section {section.section_id}; purpose: {bounded(section.purpose)}")
        for beat in section.beats:
            lines.extend([f"Beat {beat.beat_id}; kind: {beat.kind}; role: {bounded(beat.narrative_role)}",
                          f"Summary: {bounded(beat.summary)}"])
            for fact in beat.fact_refs:
                lines.append(f"Fact {fact.research_fact_id}; status: {fact.status}; use: {fact.use}; "
                             f"qualification: {bounded(fact.qualification)}")
            for entry in beat.fact_chronology:
                time = entry.historical_time
                lines.append(f"Chronology {entry.research_fact_id}: {bounded(time.display)}; "
                             f"precision: {time.precision}; start: {time.start_year}; end: {time.end_year}")
            lines.extend(f"Uncertainty: {bounded(note)}" for note in beat.uncertainty_notes)
    return lines


def apply_story_review(path, record: ApprovalRecord) -> RuntimeState:
    """Single local writer. Audit precedes atomic approved-binding/state publication.

    A crash between audit and state writes requires explicit reconciliation, matching
    Fact Review. No automatic replay/adoption, transaction, LLM or regeneration occurs.
    """
    store = ArtifactStore(path)
    state_path = store.project_dir / ".runtime/state.json"
    state = RuntimeState.model_validate_json(state_path.read_text(encoding="utf-8"))
    machine = ProjectStateMachine(state)
    machine.apply_story_review_decision(record, store)
    write_json(state_path, machine.state, replace=True)
    return machine.state
