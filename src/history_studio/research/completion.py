"""Exact Research completion commit evidence; never generation authority.

The Research exclusive lock must be held by callers. An intent is immutable and
retained after commit, so recovery can authenticate the same publication again.
"""
import hashlib
import json
from pathlib import Path
from typing import Literal

from pydantic import Field

from history_studio.models import ArtifactReference
from history_studio.models.base import Contract, Identifier, Text
from history_studio.workflow import RuntimeState
from history_studio.storage.artifact_store import write_json
from .usage import UsageLedger


def digest(value) -> str:
    return hashlib.sha256(json.dumps(value, sort_keys=True, ensure_ascii=False,
                                    separators=(",", ":"), allow_nan=False).encode("utf-8")).hexdigest()


class CompletionIntent(Contract):
    model_config = {"frozen": True}
    version: Literal[1] = 1
    project_id: Identifier
    run_id: Identifier
    candidate_ref: ArtifactReference
    candidate_sha256: Text
    sources_sha256: Text
    project_sha256: Text
    settings_sha256: Text
    state_before: RuntimeState
    state_after: RuntimeState
    ledger_before: UsageLedger
    ledger_after: UsageLedger
    commit_sha256: Text

    def verify_commit(self):
        if self.commit_sha256 != digest(self.model_dump(mode="json", exclude={"commit_sha256"})):
            raise ValueError("Research completion commit identity differs")


class CompletionPublicationError(ValueError):
    """Safe user-facing message; original exception remains in __cause__."""
    def __init__(self, operation):
        self.operation = operation
        super().__init__(f"Research completion publication pending ({operation}); "
                         "retry Research after checking local storage; no new request authorized")


class CompletionErrorDetail(Contract):
    error_class: Text
    errno: int | None = None
    winerror: int | None = None


class CompletionDiagnostic(Contract):
    operation: Text
    stage: Literal["research_completion"] = "research_completion"
    recoverable: bool = True
    errors: list[CompletionErrorDetail] = Field(default_factory=list)


def diagnostic(operation, exception):
    """Only trusted class categories and numeric OS codes, never messages/paths."""
    details, seen = [], set()
    while exception is not None and id(exception) not in seen and len(details) < 8:
        seen.add(id(exception))
        category = next((kind.__name__ for kind in (PermissionError, FileNotFoundError, OSError,
                        ValueError, RuntimeError) if isinstance(exception, kind)), "Exception")
        codes = {name: value if type(value) is int else None
                 for name in ("errno", "winerror") for value in [getattr(exception, name, None)]}
        details.append(CompletionErrorDetail(error_class=category, **codes))
        exception = exception.__cause__ or (None if exception.__suppress_context__ else exception.__context__)
    return CompletionDiagnostic(operation=operation, errors=details)


def read_bytes(path: Path):
    if any(item.is_symlink() or item.is_junction() for item in (path, path.parent)):
        raise ValueError("Research completion paths must not be links or junctions")
    return path.read_bytes()


def read_model(path: Path, model):
    return model.model_validate_json(read_bytes(path))


def publish_intent(path: Path, intent: CompletionIntent):
    intent.verify_commit()
    write_json(path, intent)
