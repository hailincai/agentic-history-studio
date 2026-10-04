"""Completed-fact resume with actual FactChecker boundaries and fake dependencies only."""
import json

import pytest
from pydantic import ValidationError

from history_studio.models import ArtifactReference, VerificationPackage, add_verification_result, create_verification_package
from history_studio.models.research_package import ResearchPackage
from history_studio.research.spans import make_spans
from history_studio.storage.artifact_store import ArtifactStore
from history_studio.verification import FactCheckingRunner, FactCheckingStopReason, FactCheckingPhase
from test_fact_checker_dispatch import native
from test_fact_checker_investigation import SequenceProvider, decision
from test_verification_context import research_ref
from test_verification_evidence import TextTools
from test_verification_package import research_data, result_for
from test_verification_submission import payload, terminal


IDS = ["RF-other", "RF-target", "RF-third"]


def setup_store(tmp_path, *, empty=False):
    store = ArtifactStore(tmp_path / "test")
    research = research_data()
    if empty:
        data = research.model_dump(mode="json")
        data["facts"] = []
        research = ResearchPackage.model_validate(data)
    for _ in range(4):
        store.save("research", research)
    return store, research


class Dependencies:
    def __init__(self, plans=None):
        self.plans = plans or {}
        self.contexts, self.providers, self.tools = [], [], []

    def provider(self, context):
        self.contexts.append(context.model_copy(deep=True))
        result = SequenceProvider(self.plans.get(context.target_fact.fact_id, [decision(terminal(payload()))]))
        self.providers.append(result)
        return result

    def tool(self, context):
        result = TextTools()
        self.tools.append(result)
        return result

    def runner(self):
        return FactCheckingRunner(provider_factory=self.provider, tools_factory=self.tool)


def seed(store, research, version=4, indexes=()):
    package = create_verification_package(research, research_input_ref=research_ref(version))
    for index in indexes:
        package = add_verification_result(package, result_for(package, index))
    saved = store.save("verification", package)
    return package, saved


def test_new_stage_runs_in_research_order_and_checkpoints_every_fact(tmp_path):
    store, research = setup_store(tmp_path)
    before = research.model_dump_json()
    deps = Dependencies()
    outcome = deps.runner().run(store, research_input_ref=research_ref(4))
    assert outcome.stop_reason == FactCheckingStopReason.COMPLETE and outcome.is_complete
    assert outcome.completed_count == 3 and outcome.verification_version == 3
    assert [c.target_fact.fact_id for c in deps.contexts] == IDS
    assert all(c.research_input_ref == research_ref(4) for c in deps.contexts)
    assert len({id(t) for t in deps.tools}) == 3
    assert store.list_versions("verification") == [1, 2, 3]
    for version in (1, 2, 3):
        package = store.load("verification", version, VerificationPackage)
        assert package.completed_fact_ids == IDS[:version]
        assert package.pending_fact_ids == IDS[version:]
        assert all(r.research_input_ref == research_ref(4) for r in package.results)
    assert outcome.package == store.load_latest("verification", VerificationPackage)
    assert research.model_dump_json() == before
    # Runner neither creates nor changes workflow runtime state.
    assert not (store.project_dir / ".runtime").exists()
    schema = json.dumps(outcome.model_json_schema())
    assert all(x not in schema for x in ("ToolObservation", "ModelRequest", "ModelResponse", "NativeToolCall"))
    files = list(store.project_dir.rglob("*.json"))
    assert len(files) == 7 and all(p.parent.name in ("research", "verification") for p in files)


def test_canonical_verified_result_passes_actual_read_submit_finalize_pipeline(tmp_path):
    store, _ = setup_store(tmp_path)
    text = TextTools().text
    canonical = make_spans("SRC-unrelated", text)
    data = payload("VERIFIED", source_id="SRC-unrelated")
    data["verification_evidence"][0]["span_id"] = next(iter(canonical.spans))
    deps = Dependencies({IDS[0]: [decision(native("read_source", '{"source_id":"SRC-unrelated"}')),
                                decision(terminal(data))]})
    outcome = deps.runner().run(store, research_input_ref=research_ref(4))
    assert outcome.is_complete and outcome.completed_count == 3
    result = outcome.package.results[0]
    assert result.status.value == "VERIFIED" and result.verification_evidence[0].excerpt == canonical.text
    assert result.research_input_ref == research_ref(4)
    assert len(deps.providers[0].requests) == 2 and len(deps.tools[0].calls) == 1
    saved = store.load("verification", 1, VerificationPackage).model_dump(mode="json")
    assert set(saved) == {"schema_version", "research_input_ref", "research_facts", "results"}
    assert '"current_observation"' not in json.dumps(saved)
    assert '"read_authorization"' not in json.dumps(saved)


@pytest.mark.parametrize("failure, expected, phase", [
    ("limit", "LIMIT_REACHED", None), ("text", "MODEL_TEXT", None), ("none", "NO_TOOL_CALL", None),
    ("provider", "FAILED", "INVESTIGATION"), ("tool", "FAILED", "INVESTIGATION"),
    ("malformed", "FAILED", "INVESTIGATION"), ("json", "FAILED", "INVESTIGATION"),
    ("finalize", "FAILED", "FINALIZATION"),
])
def test_runtime_stops_leave_first_fact_pending_without_fake_verdict_or_advancement(tmp_path, failure, expected, phase):
    store, _ = setup_store(tmp_path)
    bad = payload("VERIFIED", source_id="SRC-unrelated")
    plans = {
        "limit": [decision(native())], "text": [decision(text="UNVERIFIED ordinary prose")],
        "none": [decision()], "provider": [RuntimeError("Do not persist raw provider detail")],
        "tool": [decision(native())], "malformed": [decision(terminal(bad | {"verification_evidence": []}))],
        "json": [decision(native("submit_verification", "{bad"))], "finalize": [decision(terminal(bad))],
    }
    deps = Dependencies({IDS[0]: plans[failure]})
    if failure == "tool":
        def broken_tools(context):
            tools = TextTools()
            def fail(query):
                raise OSError("Do not persist raw fetched contents")
            tools.search_web = fail
            return tools
        runner = FactCheckingRunner(provider_factory=deps.provider, tools_factory=broken_tools)
    else:
        runner = deps.runner()
    outcome = runner.run(store, research_input_ref=research_ref(4), max_steps=1)
    assert outcome.stop_reason.value == expected
    assert (outcome.failure_phase.value if outcome.failure_phase else None) == phase
    assert outcome.stopped_fact_id == IDS[0] and outcome.completed_count == 0
    assert outcome.package.results == () and not outcome.is_complete
    assert outcome.package.pending_fact_ids == IDS
    assert outcome.verification_version is None and store.list_versions("verification") == []
    assert [c.target_fact.fact_id for c in deps.contexts] == [IDS[0]]
    assert "Do not persist" not in outcome.model_dump_json()
    fresh = Dependencies()
    resumed = fresh.runner().run(store, research_input_ref=research_ref(4))
    assert resumed.completed_count == 3 and fresh.contexts[0].target_fact.fact_id == IDS[0]


def test_partial_success_restart_skips_only_successfully_saved_facts(tmp_path):
    store, _ = setup_store(tmp_path)
    deps = Dependencies({IDS[1]: [RuntimeError("Interruption")]})
    first = deps.runner().run(store, research_input_ref=research_ref(4))
    assert first.completed_count == 1 and first.verification_version == 1 and not first.is_complete
    assert first.package.completed_fact_ids == IDS[:1] and first.stopped_fact_id == IDS[1]
    old_path = store.project_dir / "verification" / "verification_v1.json"
    old_bytes = old_path.read_bytes()
    fresh = Dependencies()
    second = fresh.runner().run(store, research_input_ref=research_ref(4))
    assert second.completed_count == 2 and second.is_complete and second.verification_version == 3
    assert [c.target_fact.fact_id for c in fresh.contexts] == IDS[1:]
    assert old_path.read_bytes() == old_bytes
    already = Dependencies()
    done = already.runner().run(store, research_input_ref=research_ref(4))
    assert done.completed_count == 0 and done.verification_version == 3 and done.is_complete
    assert already.contexts == [] and store.list_versions("verification") == [1, 2, 3]


def test_latest_global_other_snapshot_does_not_hide_latest_compatible_package(tmp_path):
    store, research = setup_store(tmp_path)
    seed(store, research, 4, (0,))
    compatible, version = seed(store, research, 4, (0, 1))
    seed(store, research, 3, (0, 1, 2))
    deps = Dependencies()
    outcome = deps.runner().run(store, research_input_ref=research_ref(4))
    assert [c.target_fact.fact_id for c in deps.contexts] == [IDS[2]]
    assert outcome.completed_count == 1 and outcome.verification_version == 4
    assert outcome.package.results[:2] == compatible.results
    assert store.load("verification", version, VerificationPackage) == compatible
    assert all(r.research_input_ref == research_ref(4) for r in outcome.package.results)


def test_other_research_version_is_never_relabelled_or_resumed(tmp_path):
    store, research = setup_store(tmp_path)
    old, _ = seed(store, research, 3, (0, 1, 2))
    deps = Dependencies()
    outcome = deps.runner().run(store, research_input_ref=research_ref(4))
    assert outcome.completed_count == 3 and [c.target_fact.fact_id for c in deps.contexts] == IDS
    assert store.load("verification", 1, VerificationPackage) == old
    assert all(r.research_input_ref == research_ref(4) for r in outcome.package.results)


def test_save_failure_leaves_fact_pending_and_restarts_from_it(tmp_path, monkeypatch):
    store, _ = setup_store(tmp_path)
    original = store.save
    def fail(artifact_type, artifact):
        if artifact_type == "verification":
            raise OSError("Publication failed before write")
        return original(artifact_type, artifact)
    monkeypatch.setattr(store, "save", fail)
    deps = Dependencies()
    outcome = deps.runner().run(store, research_input_ref=research_ref(4))
    assert outcome.stop_reason == FactCheckingStopReason.FAILED
    assert outcome.failure_phase == FactCheckingPhase.PERSISTENCE
    assert outcome.completed_count == 0 and outcome.package.results == ()
    assert store.list_versions("verification") == [] and len(deps.contexts) == 1
    monkeypatch.setattr(store, "save", original)
    fresh = Dependencies()
    resumed = fresh.runner().run(store, research_input_ref=research_ref(4))
    assert fresh.contexts[0].target_fact.fact_id == IDS[0] and resumed.completed_count == 3


def test_crash_after_successful_publication_is_recovered_from_disk(tmp_path, monkeypatch):
    store, _ = setup_store(tmp_path)
    original = store.save
    def crash(artifact_type, artifact):
        version = original(artifact_type, artifact)
        if artifact_type == "verification" and version == 2:
            raise KeyboardInterrupt("Process stopped after publication")
        return version
    monkeypatch.setattr(store, "save", crash)
    with pytest.raises(KeyboardInterrupt):
        Dependencies().runner().run(store, research_input_ref=research_ref(4))
    assert store.load("verification", 2, VerificationPackage).completed_fact_ids == IDS[:2]
    monkeypatch.setattr(store, "save", original)
    deps = Dependencies()
    outcome = deps.runner().run(store, research_input_ref=research_ref(4))
    assert outcome.completed_count == 1 and [c.target_fact.fact_id for c in deps.contexts] == IDS[2:]


def test_interrupted_read_is_not_persisted_and_fact_restarts_without_read_authorization(tmp_path):
    store, _ = setup_store(tmp_path)
    class CrashProvider:
        def __init__(self):
            self.calls = 0
        def decide(self, request):
            self.calls += 1
            if self.calls == 1:
                return decision(native("read_source", '{"source_id":"SRC-unrelated"}'))
            raise KeyboardInterrupt("Crash after read before submission")
    with pytest.raises(KeyboardInterrupt):
        FactCheckingRunner(provider_factory=lambda c: CrashProvider(), tools_factory=lambda c: TextTools()).run(
            store, research_input_ref=research_ref(4))
    assert store.list_versions("verification") == []
    # A new invocation cannot select the prior process's read span without rereading it.
    canonical = make_spans("SRC-unrelated", TextTools().text)
    bad = payload("VERIFIED", source_id="SRC-unrelated")
    bad["verification_evidence"][0]["span_id"] = next(iter(canonical.spans))
    deps = Dependencies({IDS[0]: [decision(terminal(bad))]})
    outcome = deps.runner().run(store, research_input_ref=research_ref(4))
    assert outcome.failure_phase == FactCheckingPhase.FINALIZATION and outcome.completed_count == 0
    assert outcome.package.pending_fact_ids == IDS


def test_zero_facts_completes_without_dependencies_or_empty_checkpoint(tmp_path):
    store, _ = setup_store(tmp_path, empty=True)
    deps = Dependencies()
    outcome = deps.runner().run(store, research_input_ref=research_ref(4))
    assert outcome.is_complete and outcome.stop_reason == FactCheckingStopReason.COMPLETE
    assert outcome.completed_count == 0 and outcome.verification_version is None
    assert deps.contexts == [] and store.list_versions("verification") == []


@pytest.mark.parametrize("mismatch", ["claim", "order", "missing"])
def test_matching_reference_with_wrong_membership_fails_before_any_model_call(tmp_path, mismatch):
    store, research = setup_store(tmp_path)
    package = create_verification_package(research, research_input_ref=research_ref(4))
    data = package.model_dump(mode="json")
    if mismatch == "claim":
        data["research_facts"][0]["claim_snapshot"] = "A changed assertion."
    elif mismatch == "order":
        data["research_facts"].reverse()
    else:
        data["research_facts"].pop()
    store.save("verification", VerificationPackage.model_validate(data))
    deps = Dependencies()
    with pytest.raises(ValueError, match="membership"):
        deps.runner().run(store, research_input_ref=research_ref(4))
    assert deps.contexts == []


def test_out_of_order_existing_results_do_not_change_pending_selection(tmp_path):
    store, research = setup_store(tmp_path)
    seed(store, research, 4, (2,))
    deps = Dependencies()
    outcome = deps.runner().run(store, research_input_ref=research_ref(4))
    assert [c.target_fact.fact_id for c in deps.contexts] == IDS[:2]
    assert outcome.completed_count == 2 and outcome.package.completed_fact_ids == IDS


@pytest.mark.parametrize("changes", [{"max_steps": 0}, {"max_steps": True}, {"max_output_tokens": 0}])
def test_invalid_limits_fail_before_dependency_construction(tmp_path, changes):
    store, _ = setup_store(tmp_path)
    deps = Dependencies()
    with pytest.raises(ValueError):
        deps.runner().run(store, research_input_ref=research_ref(4), **changes)
    assert deps.contexts == []


def test_invalid_project_type_and_missing_exact_research_version_fail_before_model_calls(tmp_path):
    store, _ = setup_store(tmp_path)
    deps = Dependencies()
    for ref in (ArtifactReference(project_id="other", artifact_type="research", version=4),
                ArtifactReference(project_id="test", artifact_type="story", version=4)):
        with pytest.raises(ValueError):
            deps.runner().run(store, research_input_ref=ref)
    with pytest.raises(FileNotFoundError):
        deps.runner().run(store, research_input_ref=research_ref(99))
    assert deps.contexts == []


def test_invalid_verification_checkpoint_fails_closed_instead_of_discarding_prior_work(tmp_path):
    store, research = setup_store(tmp_path)
    seed(store, research, 4, (0,))
    (store.project_dir / "verification" / "verification_v2.json").write_text('{}', encoding="utf-8")
    deps = Dependencies()
    with pytest.raises(ValidationError):
        deps.runner().run(store, research_input_ref=research_ref(4))
    assert deps.contexts == []
