"""Human review of exact accepted knowledge; no model or execution-history access."""
import json

from history_studio.models import VerificationPackage, create_verification_package
from history_studio.models.research_package import ResearchPackage, ResearchRunStatus
from history_studio.storage.artifact_store import ArtifactStore, write_json
from .approvals import ApprovalRecord
from .states import RuntimeState, ProjectState as S
from .state_machine import InvalidTransitionError, ProjectStateMachine


def load_fact_review(state: RuntimeState, store: ArtifactStore) -> VerificationPackage:
    """Validate the bound target and its complete captured membership, without discovery."""
    if state.current_state != S.WAITING_FACT_APPROVAL:
        raise InvalidTransitionError("Fact Review requires WAITING_FACT_APPROVAL")
    project_id = store.project_dir.name
    reference = state.require_verification_ref(project_id)
    research_ref = state.require_research_input_ref(project_id)
    package = store.load("verification", reference.version, VerificationPackage)
    if not package.is_complete:
        raise ValueError("Fact Review requires a complete VerificationPackage")
    if package.research_input_ref != research_ref:
        raise ValueError("Verification research snapshot must match the workflow research binding")
    research = store.load("research", research_ref.version, ResearchPackage)
    if research.project_id != project_id or research.progress.status != ResearchRunStatus.COMPLETE:
        raise ValueError("Fact Review requires this project's completed research snapshot")
    expected = create_verification_package(research, research_input_ref=research_ref)
    if package.research_facts != expected.research_facts:
        raise ValueError("Verification membership must match the exact bound research snapshot")
    return package


def fact_review_lines(state: RuntimeState, package: VerificationPackage) -> list[str]:
    """Bound each displayed value; include every fact in authoritative research order.

    JSON escaping prevents embedded newlines/control characters impersonating labels.
    Only accepted domain fields are selected, never execution context or traffic.
    """
    def bounded(value, limit=1000):
        value = str(value)
        if len(value) > limit:
            value = value[:limit] + "… [truncated]"
        return json.dumps(value, ensure_ascii=False)

    reference = state.artifacts.verification
    lines = [f"Review {reference.project_id}/verification:v{reference.version}"]
    results = {result.research_fact_id: result for result in package.results}
    for fact in package.research_facts:
        result = results[fact.research_fact_id]
        lines.extend([f"Fact ID: {fact.research_fact_id}",
            f"Claim snapshot: {bounded(fact.claim_snapshot, 600)}",
            f"Verification status: {result.status}"])
        for name in ("verification_evidence", "contradiction_evidence"):
            evidence = getattr(result, name)
            lines.append(f"{name} ({len(evidence)}):")
            for item in evidence[:20]:
                lines.append(f"  source_id={item.source_id}; source_version={item.source_version}; "
                    f"span_id={item.span_id}; excerpt={bounded(item.excerpt)}; locator={bounded(item.locator)}")
            if len(evidence) > 20:
                lines.append("  [additional evidence omitted; inspect the exact artifact]")
        lines.append(f"unresolved_issues ({len(result.unresolved_issues)}):")
        lines.extend(f"  {bounded(issue)}" for issue in result.unresolved_issues[:20])
        if len(result.unresolved_issues) > 20:
            lines.append("  [additional issues omitted; inspect the exact artifact]")
        lines.extend([f"rationale: {bounded(result.rationale)}",
                      f"independence_note: {bounded(result.independence_note)}"])
    return lines


def apply_fact_review(path, record: ApprovalRecord) -> RuntimeState:
    """One local writer. State and approved identity share one atomic state-file write.

    Approval audit publication precedes state publication, as in the existing gate.
    A crash between files fails closed and requires explicit reconciliation; it must
    never infer approval from latest artifacts or automatically replay a decision.
    """
    store = ArtifactStore(path)
    state_path = store.project_dir / ".runtime" / "state.json"
    state = RuntimeState.model_validate_json(state_path.read_text(encoding="utf-8"))
    machine = ProjectStateMachine(state)
    machine.apply_fact_review_decision(record, store)
    write_json(state_path, machine.state, replace=True)
    return machine.state
