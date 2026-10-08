"""Exact per-asset execution evidence; never manifest or budget authority."""
import hashlib
import json
from datetime import datetime, timezone
from pathlib import Path
from typing import Literal
from uuid import uuid4

from pydantic import AwareDatetime, Field

from history_studio.budget import ledger_lock
from history_studio.models import ArtifactReference
from history_studio.models.base import Contract, Text
from history_studio.storage.artifact_store import write_json


def digest(value) -> str:
    return hashlib.sha256(json.dumps(value, sort_keys=True, ensure_ascii=False,
                                    separators=(",", ":")).encode("utf-8")).hexdigest()


class RecoveryScope(Contract):
    project_id: Text
    storyboard_ref: ArtifactReference
    script_ref: ArtifactReference
    storyboard_sha256: Text
    script_sha256: Text
    providers: dict
    media_root: Text


class RecoveryEntry(Contract):
    asset_key: Text
    request_sha256: Text
    dispatch_id: Text
    dispatched_at: AwareDatetime
    state: Literal["UNCERTAIN", "COMPLETED"]
    result_type: Text
    result: dict | None = None
    result_sha256: str | None = None


class NoChargeReconciliation(Contract):
    asset_key: Text
    dispatch_id: Text
    request_sha256: Text
    evidence: Text
    decided_by: Text
    decided_at: AwareDatetime


class RecoveryJournal(Contract):
    version: Literal[1] = 1
    scope: RecoveryScope
    scope_sha256: Text
    entries: dict[str, RecoveryEntry] = Field(default_factory=dict)
    no_charge_reconciliations: list[NoChargeReconciliation] = Field(default_factory=list)


class UncertainMediaRequest(ValueError):
    """Operator evidence is required before another dispatch can be authorized."""


def provider_identity(provider):
    if provider is None:
        return None
    identity = {"class": f"{type(provider).__module__}.{type(provider).__qualname__}"}
    # Known request settings only, never client credentials or mutable counters.
    for field in ("model", "voice", "size", "quality", "seconds", "pricing", "usd_per_million_characters"):
        value = getattr(provider, field, None)
        if value is not None:
            identity[field] = value.model_dump(mode="json") if isinstance(value, Contract) else value
    data = getattr(provider, "data", None)
    if isinstance(data, bytes):
        identity["synthetic_data_sha256"] = hashlib.sha256(data).hexdigest()
    return identity


class AssetRecovery:
    def __init__(self, project_dir: Path, scope: RecoveryScope):
        self.path = project_dir / ".runtime/media_recovery.json"
        self.lock_path = project_dir / ".runtime/media_recovery.lock"
        self.scope = scope
        self.scope_hash = digest(scope.model_dump(mode="json"))
        with ledger_lock(self.lock_path):
            if self.path.exists():
                self._load()
            else:
                write_json(self.path, RecoveryJournal(scope=scope, scope_sha256=self.scope_hash))

    def _load(self):
        for path in (self.path, self.path.parent, self.lock_path):
            if path.is_symlink() or path.is_junction():
                raise ValueError("Media recovery paths must not be links or junctions")
        journal = RecoveryJournal.model_validate_json(self.path.read_text(encoding="utf-8"))
        if journal.scope != self.scope or journal.scope_sha256 != self.scope_hash:
            raise ValueError("Media recovery scope differs from exact approved inputs/configuration")
        for key, entry in journal.entries.items():
            if key != entry.asset_key or (entry.state == "COMPLETED") != (entry.result is not None):
                raise ValueError("Invalid media recovery entry")
            if entry.state == "COMPLETED" and entry.result_sha256 != digest(entry.result):
                raise ValueError("Media recovery completion evidence hash mismatch")
        return journal

    def confirm_no_charge(self, *, key, dispatch_id, evidence, decided_by):
        """Explicit operator decision only; does NOT reconcile/refund the budget.

        The operator must verify provider-side non-billing/non-completion and no
        active writer. Verified remote results are NOT adopted by this API.
        """
        with ledger_lock(self.lock_path):
            journal = self._load()
            for prior in journal.no_charge_reconciliations:
                if prior.dispatch_id == dispatch_id:
                    if (prior.asset_key, prior.evidence, prior.decided_by) != (key, evidence, decided_by):
                        raise ValueError("Conflicting no-charge reconciliation")
                    return
            entry = journal.entries.get(key)
            if entry is None or entry.dispatch_id != dispatch_id or entry.state != "UNCERTAIN":
                raise ValueError("No-charge decision requires the exact uncertain dispatch")
            # Same deterministic paths as narration/visual services, never scan.
            parts = key.split(":")
            identity = digest([self.scope.storyboard_ref.model_dump(mode="json"),
                               self.scope.script_ref.model_dump(mode="json"), parts[-1]])
            version = self.scope.storyboard_ref.version
            if parts[0] == "narration":
                relative = f"audio/storyboard-v{version}/{identity}.wav"
            elif parts[0] == "image":
                role = "intermediate/" if len(parts) == 3 else ""
                relative = f"images/storyboard-v{version}/{role}{identity}.png"
            elif parts[0] == "video":
                relative = f"video/storyboard-v{version}/{identity}.mp4"
            else:
                raise ValueError("Unknown reconciliation asset type")
            if (Path(self.scope.media_root) / relative).exists():
                raise ValueError("Published bytes require verified-result reconciliation; cannot declare no-charge")
            decision = NoChargeReconciliation(asset_key=key, dispatch_id=dispatch_id,
                request_sha256=entry.request_sha256, evidence=evidence, decided_by=decided_by,
                decided_at=datetime.now(timezone.utc))
            journal.no_charge_reconciliations.append(decision)
            del journal.entries[key]
            write_json(self.path, journal, replace=True)

    def perform(self, *, key, request, result_type, execute, validate):
        request_hash = digest([self.scope_hash, key, request])
        with ledger_lock(self.lock_path):
            journal = self._load()
            previous = journal.entries.get(key)
            if previous is not None:
                if previous.request_sha256 != request_hash or previous.result_type != result_type.__name__:
                    raise ValueError("Media recovery request identity mismatch")
                if previous.state != "COMPLETED":
                    raise UncertainMediaRequest("Media dispatch outcome uncertain; explicit operator reconciliation required")
                result = result_type.model_validate(previous.result)
            else:
                journal.entries[key] = RecoveryEntry(asset_key=key, request_sha256=request_hash,
                    dispatch_id=uuid4().hex, dispatched_at=datetime.now(timezone.utc),
                    state="UNCERTAIN", result_type=result_type.__name__)
                write_json(self.path, journal, replace=True)
                result = None
            dispatch_id = journal.entries[key].dispatch_id
        if result is not None:
            validate(result)
            return result
        # A failure here (even before dispatch) conservatively retains UNCERTAIN.
        result = execute()
        result = result_type.model_validate(result.model_dump(mode="json"))
        validate(result)
        with ledger_lock(self.lock_path):
            journal = self._load()
            entry = journal.entries[key]
            if entry.request_sha256 != request_hash or entry.dispatch_id != dispatch_id or entry.state != "UNCERTAIN":
                raise ValueError("Media completion must match its exact dispatch intent")
            entry.result = result.model_dump(mode="json")
            entry.result_sha256 = digest(entry.result)
            entry.state = "COMPLETED"
            write_json(self.path, journal, replace=True)
        return result
