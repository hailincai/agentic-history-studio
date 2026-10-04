"""Exact stored snapshot projection, without narrative generation or workflow binding."""
from history_studio.models.artifact_reference import ArtifactReference
from history_studio.models.research_package import ResearchPackage
from history_studio.models.story_context import StoryContext
from history_studio.models.verification_package import VerificationPackage
from history_studio.models.verification import VerificationStatus
from history_studio.storage.artifact_store import ArtifactStore


def build_story_context(store: ArtifactStore, *,
                        verification_input_ref: ArtifactReference) -> StoryContext:
    """Load exactly the Runtime-selected approved verification and its research version.

    Approval is a Runtime precondition, not inferred here. The project's store directory
    must match the reference project. No latest discovery or caller-supplied chronology
    package is accepted. Existing store load contracts revalidate all nested inputs.
    Research contributes only historical_time for captured verification membership.
    """
    reference = ArtifactReference.model_validate(verification_input_ref.model_dump(mode="json"))
    if reference.artifact_type != "verification":
        raise ValueError("Story input must reference a verification artifact")
    if store.project_dir.name != reference.project_id:
        raise ValueError("ArtifactStore project must match verification reference")
    verification = store.load("verification", reference.version, VerificationPackage)
    research_ref = verification.research_input_ref
    if research_ref.project_id != reference.project_id:
        raise ValueError("Research and verification snapshot projects must match")
    research = store.load("research", research_ref.version, ResearchPackage)
    if research.project_id != research_ref.project_id:
        raise ValueError("Research package project must match exact research reference")
    research_facts = {fact.fact_id: fact for fact in research.facts}
    results = {result.research_fact_id: result for result in verification.results}
    eligible, excluded, pending = [], [], []
    for snapshot in verification.research_facts:
        fact = research_facts.get(snapshot.research_fact_id)
        if fact is None:
            raise ValueError("Verification membership missing from exact research snapshot")
        if fact.claim != snapshot.claim_snapshot:
            raise ValueError("Research claim must exactly match verification claim snapshot")
        result = results.get(snapshot.research_fact_id)
        if result is None:
            pending.append(snapshot.model_dump(mode="json"))
            continue
        projected = result.model_dump(mode="json", include={
            "research_fact_id", "claim_snapshot", "status", "verification_evidence",
            "contradiction_evidence", "unresolved_issues", "rationale"})
        projected["historical_time"] = fact.historical_time.model_dump(mode="json")
        target = excluded if result.status in (VerificationStatus.REJECTED, VerificationStatus.UNVERIFIED) else eligible
        target.append(projected)
    return StoryContext.model_validate(dict(
        verification_input_ref=reference.model_dump(mode="json"),
        research_input_ref=research_ref.model_dump(mode="json"),
        eligible_facts=eligible, excluded_facts=excluded, pending_claims=pending))
