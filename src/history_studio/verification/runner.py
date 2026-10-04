"""Completed-fact checkpoints only; investigation state is never persisted or resumed."""
from collections.abc import Callable
from enum import StrEnum

from pydantic import Field

from history_studio.model_io import ModelProvider
from history_studio.models import (
    ArtifactReference, VerificationContext, VerificationPackage, add_verification_result,
    build_verification_context, create_verification_package,
)
from history_studio.models.base import Contract, Identifier, Text
from history_studio.models.research_package import ResearchPackage
from history_studio.research.boundaries import ResearchTools
from history_studio.storage.artifact_store import ArtifactStore
from .agent import FactChecker
from .investigation import InvestigationStopReason


class FactCheckingStopReason(StrEnum):
    COMPLETE = "COMPLETE"
    LIMIT_REACHED = "LIMIT_REACHED"
    MODEL_TEXT = "MODEL_TEXT"
    NO_TOOL_CALL = "NO_TOOL_CALL"
    FAILED = "FAILED"


class FactCheckingPhase(StrEnum):
    DEPENDENCIES = "DEPENDENCIES"
    INVESTIGATION = "INVESTIGATION"
    FINALIZATION = "FINALIZATION"
    ACCEPTANCE = "ACCEPTANCE"
    PERSISTENCE = "PERSISTENCE"


class FactCheckingOutcome(Contract):
    """Runtime summary, containing only accepted knowledge and bounded stop metadata."""

    package: VerificationPackage
    verification_version: int | None = Field(default=None, gt=0)
    completed_count: int = Field(default=0, ge=0)
    stop_reason: FactCheckingStopReason
    stopped_fact_id: Identifier | None = None
    failure_phase: FactCheckingPhase | None = None
    error_type: Text | None = Field(default=None, max_length=120)

    @property
    def is_complete(self) -> bool:
        return self.package.is_complete


class FactCheckingRunner:
    """One local writer; injected per-fact dependencies, no verdict policy or workflow transitions.

    Exceptions during input/resume validation propagate before investigation. Per-fact
    failures return FAILED with phase/type only, never exception bodies or model traffic.
    Concurrent runners and ambiguous filesystem publication failures require operator
    coordination; there is no automatic retry or exactly-once external execution.
    """

    def __init__(self, *, provider_factory: Callable[[VerificationContext], ModelProvider],
                 tools_factory: Callable[[VerificationContext], ResearchTools]) -> None:
        self.provider_factory = provider_factory
        self.tools_factory = tools_factory

    def run(self, store: ArtifactStore, *, research_input_ref: ArtifactReference,
            max_steps: int = 4, max_output_tokens: int = 3000) -> FactCheckingOutcome:
        """Load the exact input and checkpoint each successfully accepted fact in research order."""
        for name, value in (("max_steps", max_steps), ("max_output_tokens", max_output_tokens)):
            if isinstance(value, bool) or not isinstance(value, int) or value < 1:
                raise ValueError(f"{name} must be a positive integer")
        reference = ArtifactReference.model_validate(research_input_ref.model_dump(mode="json"))
        if reference.artifact_type != "research" or reference.project_id != store.project_dir.name:
            raise ValueError("Research reference must identify this project's research artifact")
        research = store.load("research", reference.version, ResearchPackage)
        expected = create_verification_package(research, research_input_ref=reference)
        package, version = expected, None
        # Global latest may belong to another input. Revalidate every inspected artifact;
        # malformed artifacts fail closed rather than silently discarding accepted work.
        for candidate_version in reversed(store.list_versions("verification")):
            candidate = store.load("verification", candidate_version, VerificationPackage)
            if candidate.research_input_ref != reference:
                continue
            if candidate.research_facts != expected.research_facts:
                raise ValueError("Verification membership does not match the exact research snapshot")
            package, version = candidate, candidate_version
            break
        completed_count = 0
        while package.pending_fact_ids:
            fact_id = package.pending_fact_ids[0]
            phase = FactCheckingPhase.DEPENDENCIES
            try:
                context = build_verification_context(research, fact_id, research_input_ref=reference)
                # Factories cannot alter the trusted context shared with the checker.
                provider = self.provider_factory(context.model_copy(deep=True))
                tools = self.tools_factory(context.model_copy(deep=True))
                checker = FactChecker(context, tools=tools, provider=provider)
                phase = FactCheckingPhase.INVESTIGATION
                outcome = checker.investigate(max_steps=max_steps, max_output_tokens=max_output_tokens)
                if outcome.stop_reason != InvestigationStopReason.SUBMITTED:
                    return FactCheckingOutcome(package=package, verification_version=version,
                        completed_count=completed_count, stopped_fact_id=fact_id,
                        stop_reason=FactCheckingStopReason(outcome.stop_reason.value))
                if outcome.submission is None:
                    raise ValueError("SUBMITTED outcome requires a validated submission")
                phase = FactCheckingPhase.FINALIZATION
                result = checker.finalize_submission(outcome.submission)
                phase = FactCheckingPhase.ACCEPTANCE
                updated = add_verification_result(package, result)
                phase = FactCheckingPhase.PERSISTENCE
                new_version = store.save("verification", updated)
                # Only successful publication advances the durable checkpoint/count.
                package, version = updated, new_version
                completed_count += 1
            except Exception as exc:
                return FactCheckingOutcome(package=package, verification_version=version,
                    completed_count=completed_count, stopped_fact_id=fact_id,
                    stop_reason=FactCheckingStopReason.FAILED, failure_phase=phase,
                    error_type=type(exc).__name__[:120])
        return FactCheckingOutcome(package=package, verification_version=version,
            completed_count=completed_count, stop_reason=FactCheckingStopReason.COMPLETE)
