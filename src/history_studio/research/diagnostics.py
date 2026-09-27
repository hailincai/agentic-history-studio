"""Safe checkpoint diagnostics: schema locations and trusted messages, never inputs."""
from __future__ import annotations

from typing import Literal, get_args

from pydantic import Field, ValidationError
from pydantic_core import ErrorType

from history_studio.models.base import Contract, Identifier
from history_studio.models.research_package import ResearchPackage
from .actions import ResearchUpdate, ResearchSelectionUpdate
from .evidence_diagnostics import EvidenceMismatch


class ValidationIssue(Contract):
    type: str
    location: list[str | int]
    message: str
    evidence_mismatch: EvidenceMismatch | None = None
    span_selection: dict[str, str | bool | None] | None = None


class ValidationDiagnostic(Contract):
    run_id: Identifier
    iteration: int = Field(ge=0)
    turn: int = Field(ge=1)
    operation: Literal["artifact_validation", "artifact_persistence", "workflow_transition"]
    error_class: Literal["ValidationError", "ArtifactValidationError"]
    recoverable: bool
    errors: list[ValidationIssue]
    total_errors: int = Field(ge=1)


DOMAIN_MESSAGES = {
    "source_not_read": "Read the selected source in this iteration before selecting evidence",
    "span_not_found": "Span ID does not exist in the current read source representation",
    "span_source_mismatch": "Span belongs to a different source_id",
    "stale_source_span": "Span belongs to an earlier source representation; select a current span",
    "gap_identity_changed": "Existing research gaps cannot be silently dropped or redefined",
    "fact_identity_changed": "A stable fact ID cannot be reassigned to a different claim",
    "evidence_not_source_derived": "Evidence must be an exact excerpt of a read source",
}


class ArtifactValidationError(ValueError):
    """An explicit deterministic checkpoint rejection with a safe, structured location."""

    def __init__(self, code: str, location: list[str | int], *,
                 mismatch: EvidenceMismatch | None = None, selection: dict | None = None) -> None:
        super().__init__(DOMAIN_MESSAGES[code])
        # IDs are model inputs too; apply the same bounded metadata sanitizer.
        selection = ({key: value if isinstance(value, bool) or value is None else safe_provider_field(value)
                      for key, value in selection.items()} if selection else None)
        self.issue = ValidationIssue(type=code, location=location, message=DOMAIN_MESSAGES[code],
                                     evidence_mismatch=mismatch, span_selection=selection)


# Only our static validator messages are eligible for display. Never stringify an
# arbitrary exception or include Pydantic's input/ctx (which can contain secrets).
CUSTOM_MESSAGES = frozenset({
    "Use one atomic claim, not a list or semicolon-separated claims",
    "A research fact requires evidence provenance",
    "A research fact requires a historical time expression",
    "End year requires a start year",
    "Historical range must be chronological",
    "YEAR requires a single year",
    "Range/century precision requires both bounds",
    "Gap IDs must be unique", "Fact IDs must be unique", "Atomic claims must be unique",
    "Source IDs must be unique", "Source URLs must be unique",
    "Package facts require structured time and evidence-only provenance",
    "Evidence references an unknown source", "Gap references an unknown fact",
    "Covered gaps require collected facts",
    "Complete research requires facts, sources and a plan",
    "Complete research cannot have unresolved critical gaps",
})

BUILTIN_MESSAGES = {
    "missing": "Field required",
    "extra_forbidden": "Extra inputs are not permitted",
    "string_type": "Input should be a valid string",
    "string_too_short": "String is shorter than the field minimum",
    "string_too_long": "String exceeds the field maximum length",
    "string_pattern_mismatch": "String does not match the required field pattern",
    "list_type": "Input should be a valid list",
    "dict_type": "Input should be a valid dictionary",
    "model_type": "Input should be a valid object for this contract",
    "model_attributes_type": "Input should be a valid object for this contract",
    "int_type": "Input should be an integer",
    "int_parsing": "Input cannot be parsed as an integer",
    "int_from_float": "Input must be an integer without a fractional part",
    "float_type": "Input should be a number",
    "float_parsing": "Input cannot be parsed as a number",
    "bool_type": "Input should be a boolean",
    "bool_parsing": "Input cannot be parsed as a boolean",
    "finite_number": "Input should be a finite number",
    "less_than_equal": "Input exceeds the field's inclusive upper bound",
    "less_than": "Input must be below the field's upper bound",
    "greater_than_equal": "Input is below the field's inclusive lower bound",
    "greater_than": "Input must be above the field's lower bound",
    "enum": "Input must be one of the enum values defined in the tool schema",
    "literal_error": "Input must match the literal value defined in the tool schema",
    "too_short": "Collection is smaller than the field minimum",
    "too_long": "Collection exceeds the field maximum length",
}


def _schema_fields() -> set[str]:
    fields = {"id", "notes", "researcher_confidence"}  # Legacy validation aliases.
    for contract in (ResearchUpdate, ResearchSelectionUpdate, ResearchPackage):
        schema = contract.model_json_schema()
        for node in [schema, *schema.get("$defs", {}).values()]:
            fields.update(node.get("properties", {}))
    return fields


def validation_diagnostic(exc: ValidationError | ArtifactValidationError, *, run_id: str,
                          iteration: int, turn: int, recoverable: bool,
                          operation: Literal["artifact_validation", "artifact_persistence", "workflow_transition"]
                          = "artifact_validation") -> ValidationDiagnostic:
    if isinstance(exc, ArtifactValidationError):
        issues = [exc.issue]
        total = 1
        error_class = "ArtifactValidationError"
    else:
        raw = exc.errors(include_input=False, include_context=False, include_url=False)
        total = len(raw)
        fields = _schema_fields()
        codes = frozenset(get_args(ErrorType))
        issues = []
        for error in raw[:12]:
            code = error["type"] if error["type"] in codes else "validation_error"
            message = BUILTIN_MESSAGES.get(code, "Input does not satisfy the field contract")
            custom = error["msg"].removeprefix("Value error, ")
            if code == "value_error" and custom in CUSTOM_MESSAGES:
                message = custom
            # Extra field names can themselves contain keys or model reasoning.
            location = [part if isinstance(part, int) or part in fields else "<unknown_field>"
                        for part in error["loc"]]
            issues.append(ValidationIssue(type=code, location=location, message=message))
        error_class = "ValidationError"
    return ValidationDiagnostic(run_id=run_id, iteration=iteration, turn=turn, operation=operation,
                                error_class=error_class, recoverable=recoverable,
                                errors=issues, total_errors=total)


def diagnostic_lines(diagnostic: ValidationDiagnostic | RequestDiagnostic) -> list[str]:
    if isinstance(diagnostic, RequestDiagnostic):
        details = ", ".join(f"{key}={value}" for key in
            ("http_status", "provider_type", "provider_code", "provider_param", "response_status", "incomplete_reason")
            if (value := getattr(diagnostic, key)) is not None)
        return [f"Model request failed (iteration {diagnostic.iteration}, turn {diagnostic.turn}, "
                f"{diagnostic.error_class}, {diagnostic.category}):",
                f"  [{diagnostic.code}]: {diagnostic.message}",
                f"  pending_request={diagnostic.pending_request}" + (f"; {details}" if details else "")] + (
                [f"  Provider: {diagnostic.provider_message}"] if diagnostic.provider_message else [])

    lines = [f"Checkpoint validation rejected (iteration {diagnostic.iteration}, turn {diagnostic.turn}, "
             f"{diagnostic.operation}, {diagnostic.error_class}):"]
    for issue in diagnostic.errors:
        location = ".".join(str(part) for part in issue.location) or "$"
        lines.append(f"  {location} [{issue.type}]: {issue.message}")
        if issue.span_selection is not None:
            import json
            lines.append("  Span selection: " + json.dumps(issue.span_selection))
        if issue.evidence_mismatch is not None:
            lines.append("  Evidence mismatch: " + issue.evidence_mismatch.model_dump_json())
    omitted = diagnostic.total_errors - len(diagnostic.errors)
    if omitted:
        lines.append(f"  {omitted} additional errors omitted; correct these fields first")
    return lines


class ModelResponseError(RuntimeError):
    """Local response-contract failure; arbitrary response content stays out of diagnostics."""

    def __init__(self, code: str, *, status=None, reason=None) -> None:
        super().__init__(code)
        self.code = code
        self.status = status
        self.reason = reason


class RequestDiagnostic(Contract):
    run_id: Identifier
    iteration: int = Field(ge=0)
    turn: int = Field(ge=1)
    operation: Literal["model_request"] = "model_request"
    error_class: str
    category: Literal["api", "connectivity", "model_response", "other"]
    code: str
    message: str
    http_status: int | None = None
    provider_code: str | None = Field(default=None, max_length=128)
    provider_type: str | None = Field(default=None, max_length=128)
    provider_param: str | None = Field(default=None, max_length=256)
    provider_message: str | None = Field(default=None, max_length=1000)
    response_status: str | None = None
    incomplete_reason: str | None = None
    pending_request: bool


RESPONSE_MESSAGES = {
    "incomplete_model_response": "Model response was not completed",
    "expected_one_native_function_call": "Response must contain exactly one native function call",
    "invalid_function_arguments_json": "Native function arguments are not valid JSON",
    "invalid_model_reply": "Native function call or usage does not satisfy the reply contract",
}
def safe_provider_field(value: object, *, path: bool = False) -> str | None:
    import re

    limit = 256 if path else 128
    pattern = r"[A-Za-z_$][A-Za-z0-9_.$/\[\]-]*" if path else r"[A-Za-z][A-Za-z0-9_-]*"
    if (not isinstance(value, str) or len(value) > limit
            or not re.fullmatch(pattern, value) or re.search(r"(?i)sk-|bearer|authorization", value)):
        return None
    return value


def safe_provider_message(value: object) -> str | None:
    import re
    import unicodedata

    if not isinstance(value, str):
        return None
    # Work on a bounded prefix, but redact before the final output truncation.
    value = value[:8192]
    value = re.sub(r"\x1b\[[0-?]*[ -/]*[@-~]", "", value)
    value = "".join(c if not unicodedata.category(c).startswith("C") else " " for c in value)
    # Discard payload/header/reasoning dumps, including their remaining content.
    value = re.split(r"(?i)(?:authorization|proxy-authorization|(?:request|response)[ _-]*(?:headers|body)|"
                     r"headers|hidden[ _-]*reasoning|chain[ _-]*of[ _-]*thought|reasoning)\s*[:=]", value)[0]
    value = re.sub(r"(?is)<(?:think|reasoning)>.*", "[redacted]", value)
    value = re.sub(r"(?is)\{.*", "[payload omitted]", value)
    value = re.sub(r"(?i)\bsk-[A-Za-z0-9_-]*", "[redacted]", value)
    value = re.sub(r"(?i)\bBearer\s+[^\s,;]+", "Bearer [redacted]", value)
    value = re.sub(r"(?i)(?:[A-Za-z_]*api[_-]?key|access[_-]?token|password|secret)"
                   r"\s*[:=]\s*(?:\"[^\"]*\"|'[^']*'|[^\s,;]+)", "[redacted]", value)
    value = " ".join(value.split())
    return (value[:997] + "..." if len(value) > 1000 else value) or None


def request_diagnostic(exc: Exception, *, run_id: str, iteration: int, turn: int,
                       pending_request: bool) -> RequestDiagnostic:
    import json
    import openai

    # Never stringify exceptions or serialize request/response objects. Provider
    # metadata is individually validated; only the structured message is sanitized.
    data = dict(run_id=run_id, iteration=iteration, turn=turn, pending_request=pending_request,
                error_class="Exception", category="other", code="unexpected_exception",
                message="Unexpected exception during model request; raw details suppressed")
    for cls in (openai.APITimeoutError, openai.APIConnectionError,
                openai.APIResponseValidationError, openai.BadRequestError,
                openai.AuthenticationError, openai.PermissionDeniedError, openai.NotFoundError,
                openai.ConflictError, openai.UnprocessableEntityError, openai.RateLimitError,
                openai.InternalServerError, openai.APIStatusError, openai.APIError,
                ModelResponseError, json.JSONDecodeError, ValidationError,
                OSError, RuntimeError, ValueError, TypeError, AttributeError):
        if isinstance(exc, cls):
            data["error_class"] = cls.__name__
            break
    if isinstance(exc, openai.APITimeoutError):
        data.update(category="connectivity", code="request_timeout", message="Provider request timed out")
    elif isinstance(exc, openai.APIConnectionError):
        data.update(category="connectivity", code="connection_failed", message="Provider connection failed")
    elif isinstance(exc, openai.APIResponseValidationError):
        data.update(category="model_response", code="invalid_api_response", message="SDK rejected the API response schema")
    elif isinstance(exc, openai.APIError):
        data.update(category="api", code="provider_api_error", message="Provider API rejected or failed the request")
        data["provider_code"] = safe_provider_field(exc.code)
        data["provider_type"] = safe_provider_field(exc.type)
        data["provider_param"] = safe_provider_field(exc.param, path=True)
        # SDK .message can be a formatted full body. Read only body.message;
        # never fall back to str(exc), repr(exc), response.text or request data.
        body = exc.body
        if isinstance(body, dict):
            data["provider_message"] = safe_provider_message(body.get("message"))
    if isinstance(exc, (openai.APIStatusError, openai.APIResponseValidationError)):
        if type(exc.status_code) is int and 100 <= exc.status_code <= 599:
            data["http_status"] = exc.status_code
    if isinstance(exc, ModelResponseError):
        code = exc.code if exc.code in RESPONSE_MESSAGES else "invalid_model_reply"
        data.update(category="model_response", code=code, message=RESPONSE_MESSAGES[code])
        if exc.status in ("completed", "incomplete", "failed", "cancelled", "queued", "in_progress"):
            data["response_status"] = exc.status
        if exc.reason in ("max_output_tokens", "content_filter"):
            data["incomplete_reason"] = exc.reason
    elif isinstance(exc, json.JSONDecodeError):
        data.update(category="model_response", code="invalid_function_arguments_json",
                    message=RESPONSE_MESSAGES["invalid_function_arguments_json"])
    return RequestDiagnostic(**data)
