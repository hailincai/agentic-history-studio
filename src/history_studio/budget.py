"""One project budget, serialized local writers, conservative in-flight accounting.

USD amounts are configured-price accounting, never provider invoices. Requests are
reserved before execution; missing usage, exceptions and crashes retain the bound.
Only explicit operator reconciliation can release uncertain reservations. Research
run ledgers are diagnostic/narrower limits and are not debited here a second time.
"""
from contextlib import contextmanager
from contextvars import ContextVar
from datetime import datetime, timezone
from decimal import Decimal
import hashlib
import json
import os
from pathlib import Path
import time
from typing import Literal
from uuid import uuid4

from pydantic import AwareDatetime, Field

from history_studio.models.base import Contract, Text
from history_studio.models.project import ProjectConfig
from history_studio.storage.artifact_store import write_json


class BudgetError(ValueError):
    """Safe operational error without request bodies or credential values."""


class BudgetExceeded(BudgetError):
    settlement = None


_research_accounting = ContextVar("research_accounting", default=None)


@contextmanager
def research_accounting(identity):
    """Correlate one durable Research reservation, without changing budget authority."""
    token = _research_accounting.set(identity)
    try:
        yield
    finally:
        _research_accounting.reset(token)


class UnsupportedPrice(BudgetError):
    pass


def usd(value) -> Decimal:
    if isinstance(value, bool):
        raise BudgetError("USD must be a finite nonnegative number")
    try:
        result = Decimal(str(value))
    except Exception as exc:
        raise BudgetError("USD must be a finite nonnegative number") from exc
    if not result.is_finite() or result < 0:
        raise BudgetError("USD must be a finite nonnegative number")
    return result


class BudgetReconciliation(Contract):
    project_id: Text
    historical_committed_usd: Decimal = Field(ge=0, allow_inf_nan=False)
    research_run_id: str | None = None
    evidence: Text
    decided_by: Text
    decided_at: AwareDatetime


class BudgetEvent(Contract):
    action: Literal["initialize", "reserve", "settle", "release"]
    request_id: Text
    amount_usd: Decimal = Field(ge=0, allow_inf_nan=False)
    stage: Text
    operation: Text
    model: Text
    request_sha256: Text
    basis: dict
    at: AwareDatetime


class BudgetLedger(Contract):
    schema_version: Literal[1] = 1
    project_id: Text
    events: list[BudgetEvent] = Field(min_length=1)


@contextmanager
def ledger_lock(path: Path, timeout_seconds: float = 10):
    """OS-owned byte/flock lock; process death releases it, file is never deleted."""
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("a+b") as stream:
        stream.seek(0, os.SEEK_END)
        if stream.tell() == 0:
            stream.write(b"0")
            stream.flush()
        deadline = time.monotonic() + timeout_seconds
        while True:
            try:
                stream.seek(0)
                if os.name == "nt":
                    import msvcrt
                    msvcrt.locking(stream.fileno(), msvcrt.LK_NBLCK, 1)
                else:
                    import fcntl
                    fcntl.flock(stream.fileno(), fcntl.LOCK_EX | fcntl.LOCK_NB)
                break
            except OSError:
                if time.monotonic() >= deadline:
                    raise BudgetError("Project budget lock is busy; no provider call authorized")
                time.sleep(0.01)
        try:
            yield
        finally:
            stream.seek(0)
            if os.name == "nt":
                msvcrt.locking(stream.fileno(), msvcrt.LK_UNLCK, 1)
            else:
                fcntl.flock(stream.fileno(), fcntl.LOCK_UN)


def fold(ledger: BudgetLedger):
    """Revalidate audit transitions; compute amounts instead of trusting counters."""
    if ledger.events[0].action != "initialize":
        raise BudgetError("Budget history lacks initialization")
    committed = ledger.events[0].amount_usd
    pending, closed = {}, {}
    for event in ledger.events[1:]:
        key = event.request_id
        if event.action == "reserve":
            if key in pending or key in closed or event.amount_usd <= 0:
                raise BudgetError("Invalid or duplicate budget reservation history")
            pending[key] = event
        elif event.action in ("settle", "release"):
            reservation = pending.pop(key, None)
            if reservation is None:
                raise BudgetError("Budget history closes an unknown reservation")
            if any(getattr(event, field) != getattr(reservation, field)
                   for field in ("stage", "operation", "model", "request_sha256")):
                raise BudgetError("Budget settlement must preserve exact reservation authority")
            if event.action == "release" and event.amount_usd != 0:
                raise BudgetError("Released reservation must have zero charge")
            if event.action == "release" and not (event.basis.get("evidence") and event.basis.get("decided_by")):
                raise BudgetError("Released reservation lacks explicit operator evidence")
            committed += event.amount_usd
            closed[key] = event
        else:
            raise BudgetError("Budget initialization may not be replayed")
    return committed, pending, closed


class ProjectBudget:
    def __init__(self, project_dir: Path, *, stage: str):
        self.project_dir = Path(project_dir).resolve()
        self.stage = stage
        self.path = self.project_dir / ".runtime/project_budget.json"
        self.lock_path = self.project_dir / ".runtime/project_budget.lock"

    def _project(self):
        project = ProjectConfig.model_validate_json((self.project_dir / "project.json").read_text(encoding="utf-8"))
        if project.project_id != self.project_dir.name:
            raise BudgetError("Project budget must match the persisted project")
        for path in (self.path, self.lock_path, self.path.parent, self.project_dir):
            if path.is_symlink() or path.is_junction():
                raise BudgetError("Project budget paths must not be links or junctions")
        return project

    def _load(self, project):
        if not self.path.exists():
            raise BudgetError("Project budget ledger missing; explicit initialization/reconciliation required")
        ledger = BudgetLedger.model_validate_json(self.path.read_text(encoding="utf-8"))
        if ledger.project_id != project.project_id:
            raise BudgetError("Budget ledger project identity mismatch")
        fold(ledger)
        return ledger

    def _initialize(self, project, reconciliation):
        history = any(any((self.project_dir / kind).glob("*_v*.json")) for kind in
                      ("research", "verification", "story", "script", "storyboard", "media", "assembly"))
        usage_path = self.project_dir / ".runtime/research_usage.json"
        state_path = self.project_dir / ".runtime/state.json"
        state = json.loads(state_path.read_text(encoding="utf-8"))
        history |= state.get("current_state") != "CREATED" or usage_path.exists()
        amount, basis = Decimal(0), {"policy": "new CREATED project without paid history"}
        if reconciliation is None and history:
            raise BudgetError("Historical paid activity has no project ledger; explicit budget reconciliation required")
        if reconciliation is not None:
            record = BudgetReconciliation.model_validate(reconciliation.model_dump())
            if record.project_id != project.project_id:
                raise BudgetError("Reconciliation must match the exact project")
            if usage_path.exists():
                from history_studio.research.usage import UsageLedger
                usage = UsageLedger.model_validate_json(usage_path.read_text(encoding="utf-8"))
                if record.research_run_id != usage.run_id:
                    raise BudgetError("Reconciliation must preserve the exact Research run_id")
                minimum = max(usd(usage.committed_budget_usd), usd(usage.estimated_total_cost_usd))
                if record.historical_committed_usd < minimum:
                    raise BudgetError("Reconciled historical amount must cover Research reservations and recorded usage")
            amount = record.historical_committed_usd
            basis = record.model_dump(mode="json")
        ledger = BudgetLedger(project_id=project.project_id, events=[BudgetEvent(
            action="initialize", request_id="historical", amount_usd=amount, stage="reconciliation",
            operation="initialize", model="none", request_sha256="none", basis=basis,
            at=datetime.now(timezone.utc))])
        write_json(self.path, ledger)
        return ledger

    def initialize(self, reconciliation: BudgetReconciliation):
        with ledger_lock(self.lock_path):
            project = self._project()
            if self.path.exists():
                raise BudgetError("Project budget is already initialized; history cannot be reset")
            return self._initialize(project, reconciliation)

    def initialize_new(self):
        """Only project creation invokes zero-history initialization automatically."""
        with ledger_lock(self.lock_path):
            project = self._project()
            if self.path.exists():
                raise BudgetError("Project budget is already initialized")
            return self._initialize(project, None)

    def snapshot(self):
        with ledger_lock(self.lock_path):
            project = self._project()
            ledger = self._load(project)
            committed, pending, _ = fold(ledger)
            outstanding = sum((event.amount_usd for event in pending.values()), Decimal(0))
            return dict(cap_usd=usd(project.budget_usd), committed_usd=committed,
                        outstanding_usd=outstanding, remaining_usd=max(Decimal(0), usd(project.budget_usd) - committed - outstanding),
                        outstanding_request_ids=sorted(pending))

    def reserve(self, *, request_id: str, operation: str, model: str, request_sha256: str,
                maximum_usd, basis: dict):
        maximum = usd(maximum_usd)
        if maximum <= 0:
            raise UnsupportedPrice("Paid request requires a positive known cost bound")
        with ledger_lock(self.lock_path):
            project = self._project()
            ledger = self._load(project)
            committed, pending, closed = fold(ledger)
            if request_id in pending or request_id in closed or request_id == "historical":
                raise BudgetError("Request identity already reserved or closed; refusing duplicate execution")
            outstanding = sum((event.amount_usd for event in pending.values()), Decimal(0))
            if committed + outstanding + maximum > usd(project.budget_usd):
                raise BudgetExceeded("Project budget exhausted; provider request blocked")
            basis = dict(basis)
            if self.stage == "research" and _research_accounting.get() is not None:
                basis["research_accounting_id"] = _research_accounting.get()
            ledger.events.append(BudgetEvent(action="reserve", request_id=request_id, amount_usd=maximum,
                stage=self.stage, operation=operation, model=model, request_sha256=request_sha256,
                basis=basis, at=datetime.now(timezone.utc)))
            write_json(self.path, ledger, replace=True)

    def close(self, request_id: str, *, amount_usd, basis: dict, release: bool = False):
        """Idempotent exact settlement; explicit evidence is mandatory for releases."""
        amount = usd(amount_usd)
        if release and (amount != 0 or not basis.get("evidence") or not basis.get("decided_by")):
            raise BudgetError("Release requires explicit operator evidence of no charge")
        action = "release" if release else "settle"
        with ledger_lock(self.lock_path):
            project = self._project()
            ledger = self._load(project)
            _, pending, closed = fold(ledger)
            if request_id in closed:
                previous = closed[request_id]
                if (previous.action, previous.amount_usd, previous.basis) != (action, amount, basis):
                    raise BudgetError("Conflicting duplicate budget settlement")
                return
            reservation = pending.get(request_id)
            if reservation is None:
                raise BudgetError("Cannot settle an unknown request identity")
            ledger.events.append(BudgetEvent(action=action, request_id=request_id, amount_usd=amount,
                stage=reservation.stage, operation=reservation.operation, model=reservation.model,
                request_sha256=reservation.request_sha256, basis=basis, at=datetime.now(timezone.utc)))
            write_json(self.path, ledger, replace=True)
            if amount > reservation.amount_usd:
                exc = BudgetExceeded("Reported cost exceeded reservation; recorded overrun, further calls remain capped")
                exc.settlement = ledger.events[-1].model_copy(deep=True)
                raise exc

    def research_settlement(self, identity):
        """Read only an exact correlated settlement; legacy history has no inferred match."""
        with ledger_lock(self.lock_path):
            ledger = self._load(self._project())
            _, _, closed = fold(ledger)
            reservations = [event for event in ledger.events if event.action == "reserve"
                            and event.stage == "research"
                            and event.basis.get("research_accounting_id") == identity]
            if len(reservations) > 1:
                raise BudgetError("Ambiguous Research accounting identity")
            if not reservations:
                return None
            event = closed.get(reservations[0].request_id)
            if event is None or event.action != "settle":
                return None
            return event.model_copy(deep=True)

    def _execute(self, operation, model, request, maximum, basis, call, authenticate):
        identity = uuid4().hex
        digest = hashlib.sha256(json.dumps(request, ensure_ascii=False, sort_keys=True).encode("utf-8")).hexdigest()
        self.reserve(request_id=identity, operation=operation, model=model, request_sha256=digest,
                     maximum_usd=maximum, basis=basis)
        # Never hold the ledger lock over provider execution or auto-refund failure.
        response = call()
        amount, evidence = authenticate(response)
        self.close(identity, amount_usd=amount, basis=evidence)
        return response

    def responses(self, request: dict, call, *, input_rate, output_rate,
                  overhead_tokens=4096, search_content_tokens=0, search_call_usd=0):
        rates = usd(input_rate), usd(output_rate)
        if not all(rates) or type(overhead_tokens) is not int or overhead_tokens < 4096:
            raise UnsupportedPrice("Responses requires positive model prices and a conservative framing bound")
        output_cap = request.get("max_output_tokens")
        if type(output_cap) is not int or output_cap <= 0:
            raise UnsupportedPrice("Responses requires an explicit output token cap")
        search = search_content_tokens != 0
        if search or search_call_usd != 0 or any(tool.get("type") == "web_search" for tool in request.get("tools", [])):
            raise UnsupportedPrice("Hosted search blocked: reported search-content billing semantics and hard input bound are unverified")
        value = request.get("input")
        if not isinstance(value, str) and not (isinstance(value, list) and all(
                isinstance(item, dict) and (isinstance(item.get("content"), str)
                    or item.get("type") in ("function_call", "function_call_output")) for item in value)):
            raise UnsupportedPrice("Only text/function-call Responses input has a supported token bound")
        tool_types = [tool.get("type") for tool in request.get("tools", [])]
        if any(kind != "function" for kind in tool_types):
            raise UnsupportedPrice("Unpriced hosted tools are disabled for paid Responses")
        input_bound = len(json.dumps(request, ensure_ascii=False).encode("utf-8")) + overhead_tokens + search_content_tokens
        maximum = (input_bound * rates[0] + output_cap * rates[1]) / Decimal(1000000) + usd(search_call_usd)
        pricing = dict(input_usd_per_million=str(rates[0]), output_usd_per_million=str(rates[1]),
                       input_token_bound=input_bound, output_token_cap=output_cap,
                       search_content_tokens=search_content_tokens, search_call_usd=str(usd(search_call_usd)))
        def authenticated(response):
            model = getattr(response, "model", None)
            expected = request["model"]
            if not isinstance(model, str) or not (model == expected or model.startswith(expected + "-")):
                raise BudgetError("Response model does not match the reserved pricing rule")
            if not isinstance(getattr(response, "id", None), str) or not response.id:
                raise BudgetError("Response identity missing; reservation retained")
            usage = getattr(response, "usage", None)
            counts = getattr(usage, "input_tokens", None), getattr(usage, "output_tokens", None)
            if any(type(count) is not int or count < 0 for count in counts):
                raise BudgetError("Response usage unknown; reservation retained for reconciliation")
            searches = sum(getattr(item, "type", None) == "web_search_call" for item in response.output)
            if searches != int(search):
                raise BudgetError("Unexpected hosted search count; reservation retained")
            amount = ((counts[0] + search_content_tokens) * rates[0] + counts[1] * rates[1]) / Decimal(1000000) + usd(search_call_usd)
            return amount, dict(response_id=response.id, input_tokens=counts[0], output_tokens=counts[1],
                                pricing=pricing, accounting="usage at configured rates; hosted block conservatively added; not an invoice")
        return self._execute("hosted_search" if search else "responses", request["model"], request,
                             maximum, pricing, call, authenticated)

    def tts(self, *, model, voice, text, usd_per_million_characters, call):
        if model not in ("tts-1", "tts-1-hd") or usd_per_million_characters is None:
            raise UnsupportedPrice("Guarded TTS supports tts-1/tts-1-hd only with explicit per-million-character pricing; token-billed TTS is disabled")
        rate = usd(usd_per_million_characters)
        if rate <= 0:
            raise UnsupportedPrice("TTS character price must be positive")
        # UTF-8 bytes conservatively bound character/code-unit billing for Chinese.
        maximum = len(text.encode("utf-8")) * rate / Decimal(1000000)
        request = dict(model=model, voice=voice, input=text, response_format="wav")
        pricing = dict(usd_per_million_characters=str(rate), utf8_byte_bound=len(text.encode("utf-8")))
        def authenticated(data):
            from history_studio.media.audio import measure_wav_duration
            measure_wav_duration(data)
            return maximum, dict(pricing=pricing, accounting="conservative configured character bound; no provider usage/invoice supplied")
        return self._execute("tts", model, request, maximum, pricing, call, authenticated)

    def disabled(self, operation: str):
        raise UnsupportedPrice(f"Guarded {operation} is disabled: no enforceable output-usage price bound or durable asynchronous charge rule")


def attach_budget(client, project_dir: Path, stage: str):
    client._history_studio_budget = ProjectBudget(project_dir, stage=stage)
    return client


def require_budget(client) -> ProjectBudget:
    budget = getattr(client, "_history_studio_budget", None)
    if not isinstance(budget, ProjectBudget):
        raise BudgetError("Paid provider requires an exact project budget binding")
    if getattr(client, "max_retries", None) != 0:
        raise BudgetError("Paid SDK automatic retries must be disabled")
    return budget
