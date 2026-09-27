import json
from collections.abc import Callable
from pathlib import Path
from uuid import uuid4

from pydantic import BaseModel, ValidationError

from history_studio.models.project import ProjectConfig
from history_studio.models.research_package import (
    GapStatus, ResearchPackage, ResearchProgress, ResearchRunStatus,
)
from history_studio.storage.artifact_store import ArtifactStore, write_json
from history_studio.workflow import ProjectState, ProjectStateMachine, RuntimeState
from .actions import Recommendation, ReadRequest, ResearchUpdate, ResearchSelectionUpdate, SearchRequest, action_contracts
from .spans import make_spans, resolve_selection, SourceSpans
from .progress import completion_status, progress_marks, progress_signals
from .boundaries import ResearchProvider, ResearchTools, ToolCall, ToolObservation
from .config import ResearchSettings
from .context import ContextLimitError, build_retrieval_context
from .source_store import SourceStore
from .knowledge import KnowledgeRetriever
from .diagnostics import (
    ArtifactValidationError, ValidationDiagnostic, RequestDiagnostic, diagnostic_lines,
    validation_diagnostic, request_diagnostic,
)
from .usage import LimitReached, UsageLedger
from .evidence_diagnostics import evidence_mismatch
from .web_tools import canonical_url, normalize_text, source_reference


class ResearchAgent:
    """One local writer; every attempt is bounded and every iteration is checkpointed.

    Providers choose actions. Python owns permission checks, source-derived excerpts,
    guardrails, persistence and the workflow's completion transition.
    """

    def __init__(self, provider: ResearchProvider, tools: ResearchTools, settings: ResearchSettings,
                 emit: Callable[[str], None] | None = None, *,
                 knowledge_retriever: KnowledgeRetriever | None = None) -> None:
        self.provider = provider
        self.tools = tools
        self.settings = settings
        self.emit = emit or (lambda event: None)
        self.knowledge_retriever = knowledge_retriever

    def run(self, project: ProjectConfig, store: ArtifactStore,
            runtime_configuration: BaseModel | None = None) -> ResearchPackage:
        runtime = store.project_dir / ".runtime"
        runtime.mkdir(parents=True, exist_ok=True)
        lock = runtime / "research.lock"
        try:
            with lock.open("x", encoding="utf-8") as stream:
                stream.write("Research run active; remove only after confirming no runner is active.")
        except FileExistsError as exc:
            raise ValueError("Research lock exists; another run may be active. Inspect before removing it.") from exc
        try:
            if runtime_configuration is not None:
                write_json(runtime / "research_config.json", runtime_configuration, replace=True)
            return self._run(project, store, runtime)
        finally:
            lock.unlink(missing_ok=True)

    def _load_package(self, project: ProjectConfig, store: ArtifactStore) -> ResearchPackage | None:
        versions = store.list_versions("research")
        for version in reversed(versions):
            try:
                package = store.load("research", version, ResearchPackage)
            except (ValidationError, OSError):
                self.emit(f"Skipping invalid research checkpoint v{version}")
                continue
            if (package.project_id != project.project_id or package.topic != project.topic
                    or package.research_scope != project.research_scope):
                raise ValueError("Research checkpoint belongs to a different project/topic/scope")
            return package
        if versions:
            raise ValueError("No valid research checkpoint; refusing to discard prior work")
        return None

    def _run(self, project: ProjectConfig, store: ArtifactStore, runtime: Path) -> ResearchPackage:
        if store.project_dir.name != project.project_id:
            raise ValueError("Project ID must match the artifact directory")
        state_path = runtime / "state.json"
        machine = ProjectStateMachine(RuntimeState.model_validate_json(state_path.read_text(encoding="utf-8")))
        if machine.resume_state not in (ProjectState.CREATED, ProjectState.RESEARCHING, ProjectState.RESEARCH_COMPLETE):
            raise ValueError("Research cannot run from the current workflow stage")
        package = self._load_package(project, store)
        usage_path = runtime / "research_usage.json"
        if package and package.progress.status == ResearchRunStatus.COMPLETE:
            self._check_completion(package)
            self._finish_workflow(machine, state_path)
            self.emit("Research already complete; no provider calls made")
            return package
        if machine.state.current_state == ProjectState.RESEARCH_COMPLETE:
            raise ValueError("Workflow is complete but no valid completed research package exists")
        if package and not usage_path.exists():
            raise ValueError("Resume requires the runtime usage ledger; refusing to reset consumed budget")
        ledger = (UsageLedger.model_validate_json(usage_path.read_text(encoding="utf-8"))
                  if usage_path.exists() else UsageLedger(run_id=uuid4().hex))
        if package and ledger.run_id != package.progress.run_id:
            raise ValueError("Checkpoint and usage ledger run IDs do not match")
        if ledger.pending_request:
            ledger.unknown_usage = True
            ledger.pending_request = False
        ledger.persist(usage_path)
        if package is None:
            package = ResearchPackage(project_id=project.project_id, topic=project.topic,
                                      research_scope=project.research_scope,
                                      progress=ResearchProgress(run_id=ledger.run_id))
            self._checkpoint(package, store)
        if machine.state.current_state == ProjectState.FAILED:
            machine.recover()
        if machine.state.current_state == ProjectState.CREATED:
            machine.transition(ProjectState.RESEARCHING)
        write_json(state_path, machine.state, replace=True)
        write_json(runtime / "research_settings.json", self.settings, replace=True)
        # Seed old ledgers from the accepted artifact; never credit carry-forward on resume.
        ledger.progress_seen = sorted(set(ledger.progress_seen) | progress_marks(package))
        ledger.persist(usage_path)
        hard_budget = min(self.settings.hard_budget_usd, project.budget_usd)
        source_store = SourceStore(runtime / "sources")
        observation: tuple[ToolCall, str] | None = None
        while True:
            operation = "guardrails"
            try:
                if ledger.consecutive_no_progress >= self.settings.max_no_progress_checkpoints:
                    raise LimitReached("no_progress_limit")
                if ledger.iterations_started >= self.settings.max_iterations:
                    raise LimitReached("iteration_limit")
                ledger.iterations_started += 1
                ledger.persist(usage_path)
                package.progress = ResearchProgress(run_id=ledger.run_id, iterations=ledger.iterations_started,
                                                    stop_reason=package.progress.stop_reason)
                self.emit(f"Research iteration {ledger.iterations_started}")
                passages: dict[str, str] = {}
                latest_read: SourceSpans | None = None
                read_observations: dict[str, tuple[str, bool]] = {}
                span_reads: dict[str, SourceSpans] = {}
                seen_spans: dict[str, tuple[str, str]] = {}
                for turn in range(self.settings.max_turns_per_iteration):
                    soft = ledger.committed_budget_usd >= self.settings.soft_budget_usd
                    read_envelope = None
                    span_budget = None
                    if observation and observation[0].name == "read_source":
                        read_envelope = json.loads(observation[1])
                        span_budget = (self.settings.max_observation_chars
                            - len(json.dumps(observation[0].model_dump(), ensure_ascii=False))
                            - len(json.dumps(read_envelope, ensure_ascii=False, separators=(",", ":"))) + 1)
                    context, evidence_context = build_retrieval_context(project, package, self.settings, soft, {
                        "iterations": self.settings.max_iterations - ledger.iterations_started,
                        "searches": self.settings.max_searches - ledger.search_calls,
                        "source_reads": self.settings.max_source_reads - ledger.source_reads,
                        "turns_in_iteration": self.settings.max_turns_per_iteration - turn,
                        "reserved_budget_remaining_usd": max(0, hard_budget - ledger.committed_budget_usd),
                        "completion_min_facts": self.settings.min_facts,
                        "completion_min_sources": self.settings.min_sources,
                    }, list(passages), latest_read, previous_outcome=ledger.last_checkpoint_outcome,
                        observation_span_budget=span_budget, knowledge_retriever=self.knowledge_retriever)
                    if read_envelope is not None:
                        observation = (observation[0], json.dumps(
                            {**read_envelope, **evidence_context["latest_read_source"]},
                            ensure_ascii=False, separators=(",", ":")))
                    if soft:
                        self.emit("Soft budget reached; model asked to consolidate")
                    if observation:
                        size = len(json.dumps(observation[0].model_dump(), ensure_ascii=False)) + len(observation[1])
                        if size > self.settings.max_observation_chars:
                            raise LimitReached("observation_context_limit")
                    quote = self.provider.reserve_cost(context, observation, self.settings.max_output_tokens)
                    ledger.reserve(quote, hard_budget, "model", usage_path)
                    operation = "model_request"
                    reply = self.provider.decide(context, observation, self.settings.max_output_tokens)
                    ledger.record(reply.usage, usage_path)
                    call = reply.call
                    if call.name not in action_contracts():
                        raise ValueError("Unapproved research tool")
                    ledger.tool_calls += 1
                    ledger.persist(usage_path)
                    if call.name == "checkpoint_research":
                        operation = "artifact_validation"
                        try:
                            proposal = ResearchSelectionUpdate.model_validate(call.arguments)
                            facts = []
                            for fact_index, fact in enumerate(proposal.facts):
                                data = fact.model_dump()
                                # Only the same fact's last accepted evidence is trusted.
                                # Copy every persisted field; proposals contain IDs, never text/version overrides.
                                accepted = {(e.source_id, e.span_id): e
                                    for old in package.facts if old.fact_id == fact.fact_id
                                    for e in old.evidence if e.span_id is not None and e.source_version is not None}
                                data["evidence"] = []
                                for index, selection in enumerate(fact.evidence):
                                    prior = accepted.get((selection.source_id, selection.span_id))
                                    evidence = (prior.model_copy(deep=True) if prior is not None else
                                        resolve_selection(selection.source_id, selection.span_id, span_reads,
                                            seen_spans, ["facts", fact_index, "evidence", index, "span_id"]))
                                    data["evidence"].append(evidence)
                                facts.append(data)
                            update = ResearchUpdate(plan=proposal.plan, facts=facts,
                                                    recommendation=proposal.recommendation)
                            candidate = self._apply_update(package, update, passages, read_observations)
                        except (ValidationError, ArtifactValidationError) as exc:
                            diagnostic = validation_diagnostic(exc, run_id=ledger.run_id,
                                iteration=ledger.iterations_started, turn=turn + 1, recoverable=True)
                            operation = "validation_diagnostics"
                            self._record_validation(diagnostic, runtime)
                            operation = "artifact_validation"
                            # Keep the last valid package and this iteration's retrieved evidence.
                            # The next ordinary turn must pass the same budget/context checks.
                            observation = (call, json.dumps({
                                "status": "validation_error",
                                "errors": [issue.model_dump() for issue in diagnostic.errors],
                                "total_errors": diagnostic.total_errors,
                                "instruction": "Checkpoint rejected; no changes accepted. Correct the listed fields and resubmit checkpoint_research within the remaining limits. Do not invent evidence.",
                            }, ensure_ascii=False))
                            continue
                        candidate.progress.stop_reason = None
                        if update.recommendation == Recommendation.COMPLETE:
                            try:
                                self._check_completion(candidate)
                            except ValueError:
                                candidate.progress.stop_reason = "completion_requirements_not_met"
                            else:
                                candidate.progress.status = ResearchRunStatus.COMPLETE
                        marks = progress_marks(candidate, span_reads)
                        signals = progress_signals(marks, set(ledger.progress_seen))
                        coverage = completion_status(candidate, self.settings)
                        if not signals and candidate.progress.status != ResearchRunStatus.COMPLETE:
                            ledger.consecutive_no_progress += 1
                            feedback = {"status": "no_progress", "consecutive_attempts": ledger.consecutive_no_progress,
                                "coverage": coverage, "instruction": "Checkpoint added no new facts, sources, read representations, evidence, questions or coverage. No proposal changes accepted. Choose an action that advances coverage or COMPLETE if deterministic requirements are met."}
                            ledger.last_checkpoint_outcome = feedback
                            ledger.persist(usage_path)
                            self.emit(f"No research progress: {ledger.consecutive_no_progress}; {coverage}")
                            observation = (call, json.dumps(feedback, ensure_ascii=False))
                            if ledger.consecutive_no_progress >= self.settings.max_no_progress_checkpoints:
                                raise LimitReached("no_progress_limit")
                            continue
                        operation = "artifact_persistence"
                        self._checkpoint(candidate, store)
                        ledger.progress_seen = sorted(set(ledger.progress_seen) | marks)
                        ledger.consecutive_no_progress = 0
                        ledger.last_checkpoint_outcome = {"status": "checkpoint_accepted", "progress": signals,
                                                          "coverage": coverage}
                        ledger.persist(usage_path)
                        package = candidate
                        observation = None
                        if package.progress.status == ResearchRunStatus.COMPLETE:
                            operation = "workflow_transition"
                            self._finish_workflow(machine, state_path)
                            self._summary(package, ledger)
                            return package
                        break
                    if call.name == "search_web":
                        request = SearchRequest.model_validate(call.arguments)
                        if ledger.search_calls >= self.settings.max_searches:
                            raise LimitReached("search_limit")
                        ledger.reserve(self.tools.search_reserve_cost(request.query), hard_budget, "search", usage_path)
                        self.emit(f"Tool: search_web query={request.query}")
                        operation = "web_search"
                        result = self.tools.search_web(request.query)
                    else:
                        request = ReadRequest.model_validate(call.arguments)
                        source = next((s for s in package.sources if s.source_id == request.source_id), None)
                        if source is None:
                            raise ValueError("Only discovered sources may be read")
                        if ledger.source_reads >= self.settings.max_source_reads:
                            raise LimitReached("source_read_limit")
                        ledger.reserve(0, hard_budget, "read", usage_path)
                        self.emit(f"Tool: read_source {source.source_id}")
                        operation = "source_read"
                        result = self.tools.read_source(source, self.settings.max_page_chars)
                    ledger.record(result.usage, usage_path)
                    if call.name == "search_web":
                        if result.kind != "search":
                            raise ValueError("Search observations cannot supply fetched evidence")
                        result.text = ""
                    elif (result.kind != "source" or result.source_id != source.source_id
                          or len(result.sources) != 1
                          or canonical_url(str(result.sources[0].url)) != canonical_url(str(source.url))):
                        raise ValueError("Read observation must match the requested source")
                    result = self._merge_sources(package, result)
                    if result.kind == "source" and result.source_id:
                        read = make_spans(result.source_id, result.text, result.truncated)
                        # Persist the complete fetched representation before retrieval.
                        # Context selection never truncates text or regenerates span IDs.
                        operation = "source_persistence"
                        source_store.put(read)
                        output = json.dumps({"kind": "source", "sources": [
                            s.model_dump(mode="json") for s in result.sources]}, ensure_ascii=False)
                        span_reads[result.source_id] = read
                        for span_id in read.spans:
                            seen_spans[span_id] = (read.source_id, read.source_version)
                        passages[result.source_id] = read.text
                        read_observations[result.source_id] = (read.text, read.truncated)
                        latest_read = read
                    else:
                        output = result.model_dump_json(exclude={"usage"})
                    observation = (call, output)
                else:
                    raise LimitReached("turn_limit")
            except (LimitReached, ContextLimitError) as exc:
                package.progress = ResearchProgress(run_id=ledger.run_id, iterations=ledger.iterations_started,
                    status=ResearchRunStatus.LIMIT_REACHED, stop_reason=str(exc))
                self._checkpoint(package, store)
                # Research remains resumable, but this is not successful completion.
                write_json(state_path, machine.state, replace=True)
                self._summary(package, ledger)
                return package
            except Exception as exc:
                if operation == "model_request":
                    diagnostic = request_diagnostic(exc, run_id=ledger.run_id,
                        iteration=ledger.iterations_started, turn=turn + 1,
                        pending_request=ledger.pending_request)
                    self._record_validation(diagnostic, runtime)
                if isinstance(exc, ValidationError) and operation in (
                    "artifact_validation", "artifact_persistence", "workflow_transition"
                ):
                    diagnostic = validation_diagnostic(exc, run_id=ledger.run_id,
                        iteration=ledger.iterations_started, turn=turn + 1,
                        recoverable=False, operation=operation)
                    self._record_validation(diagnostic, runtime)
                # Third-party exception text can contain credentials, request bodies or pages.
                # Persist only a safe operational code, never raw exception messages.
                if ledger.pending_request:
                    ledger.unknown_usage = True
                ledger.persist(usage_path)
                package.progress = ResearchProgress(run_id=ledger.run_id, iterations=ledger.iterations_started,
                    status=ResearchRunStatus.FAILED, stop_reason=f"research_{operation}_failed")
                try:
                    self._checkpoint(package, store)
                finally:
                    if machine.state.current_state != ProjectState.FAILED:
                        machine.fail(f"research_{operation}_failed; inspect configuration/connectivity and retry research")
                    write_json(state_path, machine.state, replace=True)
                self._summary(package, ledger)
                return package

    def _record_validation(self, diagnostic: ValidationDiagnostic | RequestDiagnostic, runtime: Path) -> None:
        version = ArtifactStore(runtime).save("diagnostics", diagnostic)
        for line in diagnostic_lines(diagnostic):
            self.emit(line)
        self.emit(f"Research diagnostics: {runtime / 'diagnostics' / f'diagnostics_v{version}.json'}")

    def _merge_sources(self, package: ResearchPackage, result: ToolObservation) -> ToolObservation:
        known = {canonical_url(str(source.url)): source for source in package.sources}
        result = result.model_copy(deep=True)
        remapped = []
        for source in result.sources:
            url = canonical_url(str(source.url))
            normalized = source_reference(url, source.title)
            source = source.model_copy(update={"source_id": normalized.source_id, "url": normalized.url})
            if url in known and result.kind != "source":
                source = known[url]
            known[url] = source
            remapped.append(source)
        if result.source_id and len(remapped) == 1:
            result.source_id = remapped[0].source_id
        result.sources = remapped
        result.truncated |= len(result.text) > self.settings.max_page_chars
        result.text = result.text[:self.settings.max_page_chars]
        package.sources = list(known.values())
        return result

    def _apply_update(self, package: ResearchPackage, update: ResearchUpdate,
                      passages: dict[str, str],
                      read_observations: dict[str, tuple[str, bool]] | None = None) -> ResearchPackage:
        existing_gaps = {gap.gap_id: gap for gap in package.plan.gaps}
        new_gaps = {gap.gap_id: gap for gap in update.plan.gaps}
        for gap_index, (gap_id, old) in enumerate(existing_gaps.items()):
            if gap_id not in new_gaps or (new_gaps[gap_id].question, new_gaps[gap_id].critical) != (old.question, old.critical):
                raise ArtifactValidationError("gap_identity_changed", ["plan", "gaps", gap_index])
        facts = {fact.fact_id: fact for fact in package.facts}
        prior_evidence = {(e.source_id, e.excerpt) for f in package.facts for e in f.evidence}
        for fact_index, fact in enumerate(update.facts):
            if fact.fact_id in facts and facts[fact.fact_id].claim != fact.claim:
                raise ArtifactValidationError("fact_identity_changed", ["facts", fact_index, "fact_id"])
            for evidence_index, evidence in enumerate(fact.evidence):
                quote = normalize_text(evidence.excerpt)
                if (evidence.source_id, evidence.excerpt) not in prior_evidence:
                    if not quote or quote not in passages.get(evidence.source_id, ""):
                        raise ArtifactValidationError("evidence_not_source_derived",
                            ["facts", fact_index, "evidence", evidence_index, "excerpt"],
                            mismatch=evidence_mismatch(evidence.source_id, evidence.excerpt, passages,
                                raw_text=(read_observations or {}).get(evidence.source_id, (None, None))[0],
                                truncated=(read_observations or {}).get(evidence.source_id, (None, None))[1]))
            facts[fact.fact_id] = fact
        data = package.model_dump()
        data.update(plan=update.plan, facts=list(facts.values()))
        return ResearchPackage.model_validate(data)

    def _check_completion(self, package: ResearchPackage) -> None:
        if not completion_status(package, self.settings)["can_complete"]:
            raise ValueError("Completion requires coverage, enough facts and enough cited sources")

    def _checkpoint(self, package: ResearchPackage, store: ArtifactStore) -> None:
        # Revalidate after mutations before making the entire package durable.
        package = ResearchPackage.model_validate(package.model_dump())
        version = store.save("research", package)
        self.emit(f"Checkpoint research_v{version}: {len(package.facts)} facts, {len(package.sources)} sources")

    def _finish_workflow(self, machine: ProjectStateMachine, path: Path) -> None:
        if machine.state.current_state == ProjectState.FAILED:
            machine.recover()
        if machine.state.current_state == ProjectState.CREATED:
            machine.transition(ProjectState.RESEARCHING)
        if machine.state.current_state == ProjectState.RESEARCHING:
            machine.transition(ProjectState.RESEARCH_COMPLETE)
        write_json(path, machine.state, replace=True)

    def _summary(self, package: ResearchPackage, ledger: UsageLedger) -> None:
        self.emit(f"Research {package.progress.status}: {package.progress.stop_reason or 'coverage accepted'}; "
                  f"{ledger.model_calls} model calls, {ledger.search_calls} searches, {ledger.source_reads} reads; "
                  f"estimated USD {ledger.estimated_total_cost_usd:.4f}, "
                  f"reserved USD {ledger.committed_budget_usd:.4f}, unknown usage={ledger.unknown_usage}")
