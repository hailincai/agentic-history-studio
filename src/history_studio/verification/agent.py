"""Claim-bounded investigation and separate deterministic verification-provenance acceptance."""
import json
import hashlib

from history_studio.model_io import ModelProvider, ModelRequest, ModelResponse, NativeToolCall
from history_studio.models.verification_context import VerificationContext
from history_studio.models.verification import VerificationEvidence, VerificationResult
from history_studio.research.actions import ReadRequest, SearchRequest
from history_studio.research.boundaries import ResearchTools, ToolObservation
from history_studio.research.spans import SourceSpans, make_spans, resolve_selection
from history_studio.research.web_tools import canonical_url, normalize_text
from .context import (INSTRUCTIONS, OBSERVATION_INSTRUCTIONS, build_context, serialize_context,
                      serialize_observation_context)
from .tools import investigation_tools
from .investigation import InvestigationOutcome, InvestigationStopReason
from .submission import VerificationSubmission, VerificationSubmissionInput


class FactChecker:
    """Investigate one atomic claim through explicit boundaries and a bounded runtime loop."""

    def __init__(self, context: VerificationContext, tools: ResearchTools | None = None, *,
                 provider: ModelProvider | None = None) -> None:
        if not isinstance(context, VerificationContext):
            raise TypeError("FactChecker requires one VerificationContext")
        # Own a detached, validated snapshot instead of an alias to caller working state.
        self._context = VerificationContext.model_validate(context.model_dump(mode="json"))
        self.tools = tools
        self.provider = provider
        # Metadata lookup only: neither source discovery nor a read authorizes evidence here.
        self._sources = {source.source_id: source.model_copy(deep=True) for source in self._context.sources}
        # Runtime-only truth/authorization, never seeded from original research evidence.
        self._reads: dict[str, SourceSpans] = {}
        self._seen_spans: dict[str, tuple[str, str]] = {}
        self._read_urls: dict[str, str] = {}

    def prepare(self) -> str:
        """Return instructions followed by structured JSON; performs no model request."""
        return build_context(self._context)

    def tool_definitions(self) -> list[dict]:
        """Describe configured capabilities without invoking them or preparing a verdict."""
        return investigation_tools() if self.tools is not None else []

    def decide_next_action(self, max_output_tokens: int = 3000) -> ModelResponse:
        """One model decision, returned unchanged; no tool execution or verdict interpretation.

        Required tools apply only to this investigation-action stage, not a future terminal
        submission policy. With no configured tools, text is permitted using tool_choice=none.
        """
        if self.provider is None:
            raise RuntimeError("decide_next_action requires an injected ModelProvider")
        tools = self.tool_definitions()
        request = ModelRequest(instructions=INSTRUCTIONS, input=serialize_context(self._context),
                               tools=tools, tool_choice="required" if tools else "none",
                               max_output_tokens=max_output_tokens)
        return self.provider.decide(request)

    def decide_after_observation(self, observation: ToolObservation, *,
                                 max_output_tokens: int = 3000) -> ModelResponse:
        """One fresh reasoning turn over original context and one transient observation; no dispatch."""
        if self.provider is None:
            raise RuntimeError("decide_after_observation requires an injected ModelProvider")
        tools = self.tool_definitions()
        read = self._reads.get(observation.source_id) if isinstance(observation, ToolObservation) else None
        if read is not None and (observation.kind != "source" or
                                not normalize_text(observation.text).startswith(read.text)):
            read = None
        request = ModelRequest(instructions=INSTRUCTIONS + "\n" + OBSERVATION_INSTRUCTIONS,
            input=serialize_observation_context(self._context, observation, read), tools=tools,
            tool_choice="required" if tools else "none", max_output_tokens=max_output_tokens)
        return self.provider.decide(request)

    def investigate(self, *, max_steps: int = 4,
                    max_output_tokens: int = 3000) -> InvestigationOutcome:
        """Each step is one decision plus at most one action; never request beyond the ceiling.

        Keep runtime observations for reporting, but send only the current observation
        with original claim context. Limits/text/absence are not verification judgments.
        """
        if isinstance(max_steps, bool) or not isinstance(max_steps, int) or max_steps < 1:
            raise ValueError("max_steps must be a positive integer")
        # Metadata may persist, but previous investigations never authorize this one.
        self._reads = {}
        self._seen_spans = {}
        self._read_urls = {}
        observations = []
        for step in range(1, max_steps + 1):
            response = (self.decide_after_observation(observations[-1], max_output_tokens=max_output_tokens)
                        if observations else self.decide_next_action(max_output_tokens=max_output_tokens))
            if not isinstance(response, ModelResponse):
                raise TypeError("Investigation requires a ModelResponse")
            if response.status not in (None, "completed"):
                raise ValueError("Investigation requires a completed model response")
            if len(response.tool_calls) > 1:
                raise ValueError("Investigation permits exactly one tool call per executable decision")
            if not response.tool_calls:
                reason = (InvestigationStopReason.MODEL_TEXT if response.text.strip()
                          else InvestigationStopReason.NO_TOOL_CALL)
                return InvestigationOutcome(final_response=response, observations=observations,
                                            steps=step, stop_reason=reason)
            call = response.tool_calls[0]
            if call.name == "submit_verification":
                submission = self.submit_verification(call)
                return InvestigationOutcome(final_response=response, observations=observations,
                    steps=step, stop_reason=InvestigationStopReason.SUBMITTED, submission=submission)
            observations.append(self.execute_tool_call(call))
        return InvestigationOutcome(final_response=response, observations=observations,
                                    steps=max_steps, stop_reason=InvestigationStopReason.LIMIT_REACHED)

    def submit_verification(self, call: NativeToolCall) -> VerificationSubmission:
        """Validate one terminal semantic proposal; no external action or evidence acceptance."""
        if not isinstance(call, NativeToolCall) or call.name != "submit_verification":
            raise ValueError("Submission requires one submit_verification NativeToolCall")
        try:
            arguments = json.loads(call.arguments)
        except json.JSONDecodeError as exc:
            raise ValueError("Invalid submission arguments JSON") from exc
        if not isinstance(arguments, dict):
            raise ValueError("Submission arguments must be a JSON object")
        proposal = VerificationSubmissionInput.model_validate(arguments)
        if any(e.source_id not in self._sources for e in
               [*proposal.verification_evidence, *proposal.contradiction_evidence]):
            raise ValueError("Submission selects an unknown source")
        return VerificationSubmission(**proposal.model_dump(),
            research_fact_id=self._context.target_fact.fact_id,
            claim_snapshot=self._context.target_fact.claim)

    def finalize_submission(self, submission: VerificationSubmission) -> VerificationResult:
        """Authenticate and extract every selected read span; no semantic inference or I/O."""
        if not isinstance(submission, VerificationSubmission):
            raise TypeError("Finalization requires one VerificationSubmission")
        validated = VerificationSubmission.model_validate(submission.model_dump(mode="json"))
        target = self._context.target_fact
        if (validated.research_fact_id != target.fact_id or validated.claim_snapshot != target.claim):
            raise ValueError("Submission does not match the target fact and immutable claim")
        for read in self._reads.values():
            if read != make_spans(read.source_id, read.text, read.truncated):
                raise ValueError("Canonical investigation material is unavailable or invalid")
        data = validated.model_dump(mode="json")
        for role in ("verification_evidence", "contradiction_evidence"):
            materialized = []
            for index, selection in enumerate(getattr(validated, role)):
                if selection.source_id in self._reads and (
                        selection.source_id not in self._sources or self._read_urls.get(selection.source_id) !=
                        canonical_url(str(self._sources[selection.source_id].url))):
                    raise ValueError("Authorized read source identity changed")
                evidence = resolve_selection(selection.source_id, selection.span_id,
                    self._reads, self._seen_spans, [role, index, "span_id"],
                    known_source_ids=set(self._sources))
                materialized.append(VerificationEvidence(**evidence.model_dump()).model_dump(mode="json"))
            data[role] = materialized
        # Content-addressed runtime identity follows the project's deterministic hash convention.
        encoded = json.dumps(["verification-v1", data], ensure_ascii=False, sort_keys=True,
                             separators=(",", ":")).encode("utf-8")
        return VerificationResult(verification_id="V-" + hashlib.sha256(encoded).hexdigest(), **data)

    def execute_tool_call(self, call: NativeToolCall, *, max_chars: int = 8000) -> ToolObservation:
        """Execute one explicit search/read request and stop at investigation material.

        call_id stays on the caller-owned NativeToolCall for correlation. Observations retain
        their existing contract; no provenance identity is derived from the call ID.
        """
        if self.tools is None:
            raise RuntimeError("execute_tool_call requires configured ResearchTools")
        if not isinstance(call, NativeToolCall):
            raise TypeError("execute_tool_call requires one NativeToolCall")
        if call.name not in {"search_web", "read_source"}:
            raise ValueError("Unsupported Fact Checker investigation tool")
        try:
            arguments = json.loads(call.arguments)
        except json.JSONDecodeError as exc:
            raise ValueError("Invalid tool arguments JSON") from exc
        if not isinstance(arguments, dict):
            raise ValueError("Tool arguments must be a JSON object")
        if call.name == "search_web":
            request = SearchRequest.model_validate(arguments)
            result = self.tools.search_web(request.query)
            if not isinstance(result, ToolObservation) or result.kind != "search":
                raise ValueError("Search must return a search ToolObservation")
            for source in result.sources:
                self._sources[source.source_id] = source.model_copy(deep=True)
            return result
        request = ReadRequest.model_validate(arguments)
        source = self._sources.get(request.source_id)
        if source is None:
            raise ValueError("Read source_id must identify an original or discovered source")
        if isinstance(max_chars, bool) or not isinstance(max_chars, int) or max_chars <= 0:
            raise ValueError("max_chars must be a positive integer")
        result = self.tools.read_source(source.model_copy(deep=True), max_chars)
        if not isinstance(result, ToolObservation) or result.kind != "source" or result.source_id != request.source_id:
            raise ValueError("Read must return a source ToolObservation matching the requested source_id")
        if (len(result.sources) != 1 or result.sources[0].source_id != request.source_id or
                canonical_url(str(result.sources[0].url)) != canonical_url(str(source.url))):
            raise ValueError("Read metadata must match the requested source identity")
        read = make_spans(request.source_id, result.text[:max_chars],
                          result.truncated or len(result.text) > max_chars)
        if not read.spans:
            raise ValueError("Read contains no canonical investigation material")
        self._reads[request.source_id] = read
        self._read_urls[request.source_id] = canonical_url(str(source.url))
        for span_id in read.spans:
            self._seen_spans[span_id] = (read.source_id, read.source_version)
        return result
