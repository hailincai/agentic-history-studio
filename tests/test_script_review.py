import json

import pytest

from history_studio.models import ScriptPackage
from history_studio.workflow import RuntimeState, ProjectState as S, ApprovalRecord, ProjectStateMachine
from history_studio.workflow.script import ScriptWorkflow
from history_studio.workflow.script_review import load_script_review, script_review_lines, apply_script_review
from history_studio.storage.artifact_store import write_json
from test_script_workflow import prepare, read, terminal
from test_script_generation import FakeProvider


def setup_review(tmp_path):
    project, store, _ = prepare(tmp_path)
    ScriptWorkflow(provider_factory=lambda ctx: FakeProvider([terminal()])).run(project, store)
    original = store.load("script", 1, ScriptPackage)
    for _ in range(4):
        store.save("script", original)
    data = read(store).model_dump(mode="json")
    data["artifacts"]["script"]["version"] = 4
    # Simulate stale downstream lineage to prove every decision invalidates it.
    for key in ("approved_script", "storyboard", "approved_storyboard"):
        data["artifacts"][key] = dict(project_id="project", artifact_type=key.removeprefix("approved_"), version=1)
    write_json(store.project_dir / ".runtime/state.json", RuntimeState.model_validate(data), replace=True)
    return store


def record(decision="APPROVED", version=4):
    return ApprovalRecord(project_id="project", stage="script", artifact_type="script", artifact_version=version,
        decision=decision, feedback="Human reviewed the narration", decided_at="2026-10-05T12:00:00Z",
        decided_by="Reviewer", decision_source="human")


@pytest.mark.parametrize("decision,destination", [("APPROVED", S.SCRIPT_APPROVED),
    ("REVISION_REQUESTED", S.SCRIPT_GENERATING), ("REJECTED", S.SCRIPT_GENERATING)])
def test_exact_decision_audit_and_immutable_artifacts(tmp_path, monkeypatch, decision, destination):
    from history_studio.storage.artifact_store import ArtifactStore
    store = setup_review(tmp_path)
    before = read(store)
    original = store.load("script", 4, ScriptPackage)
    monkeypatch.setattr(ArtifactStore, "load_latest", lambda *args: pytest.fail("No latest lookup"))
    package = load_script_review(before, store)
    lines = script_review_lines(before, package)
    assert lines[0] == "Review project/script:v4"
    assert any("story:v1" in line for line in lines)
    assert any("HISTORICAL" in line and "segment_01" in line for line in lines)
    assert any("Approved narration" in line for line in lines)
    assert "Story beat: b1" in lines and "Fact: f1" in lines
    machine = ProjectStateMachine(before)
    result = machine.apply_human_decision(record(decision), store)
    write_json(store.project_dir / ".runtime/state.json", result, replace=True)
    assert result.current_state == destination and result.artifacts.script == before.artifacts.script
    assert result.artifacts.approved_script == (before.artifacts.script if decision == "APPROVED" else None)
    assert result.artifacts.storyboard is result.artifacts.approved_storyboard is None
    for key in ("research", "verification", "approved_verification", "story", "approved_story"):
        assert getattr(result.artifacts, key) == getattr(before.artifacts, key)
    assert store.list_versions("script") == [1, 2, 3, 4, 5]
    assert store.load("script", 4, ScriptPackage) == original
    assert store.load("approvals", 1, ApprovalRecord) == record(decision)


@pytest.mark.parametrize("failure", ["state", "missing", "type", "project", "version"])
def test_review_fails_closed(tmp_path, failure):
    store = setup_review(tmp_path)
    data = read(store).model_dump(mode="json")
    if failure == "state":
        data["current_state"] = "SCRIPT_GENERATING"
    elif failure == "missing":
        data["artifacts"]["script"] = None
    else:
        field, value = {"type": ("artifact_type", "research"), "project": ("project_id", "foreign"),
                        "version": ("version", 99)}[failure]
        data["artifacts"]["script"][field] = value
    with pytest.raises((ValueError, OSError)):
        load_script_review(RuntimeState.model_validate(data), store)


@pytest.mark.parametrize("changes", [{"artifact_version": 3}, {"artifact_version": 5},
    {"project_id": "foreign"}, {"stage": "story", "artifact_type": "story"}, {"decision_source": "agent"}])
def test_stale_wrong_and_malformed_decisions_rejected(tmp_path, changes):
    store = setup_review(tmp_path)
    with pytest.raises(ValueError):
        apply_script_review(store.project_dir, record().model_copy(update=changes))
    assert store.list_versions("approvals") == []


@pytest.mark.parametrize("failure", ["provenance", "fact"])
def test_reloaded_script_cross_artifact_validation(tmp_path, failure):
    store = setup_review(tmp_path)
    data = store.load("script", 4, ScriptPackage).model_dump(mode="json")
    if failure == "provenance":
        data["story_input_ref"]["version"] = 2
    else:
        data["sections"][0]["segments"][0]["grounding"]["research_fact_ids"] = ["unknown"]
    (store.project_dir / "script/script_v4.json").write_text(json.dumps(data))
    with pytest.raises(ValueError):
        apply_script_review(store.project_dir, record())
    assert not store.list_versions("approvals")


def test_audit_state_crash_does_not_adopt_approval(tmp_path, monkeypatch):
    import history_studio.workflow.script_review as module
    store = setup_review(tmp_path)
    data = read(store).model_dump(mode="json")
    data["artifacts"]["approved_script"] = None
    write_json(store.project_dir / ".runtime/state.json", RuntimeState.model_validate(data), replace=True)
    monkeypatch.setattr(module, "write_json", lambda *args, **kwargs: (_ for _ in ()).throw(OSError("state failure")))
    with pytest.raises(OSError):
        apply_script_review(store.project_dir, record())
    assert read(store).current_state == S.WAITING_SCRIPT_APPROVAL
    assert read(store).artifacts.approved_script is None
    assert store.list_versions("approvals") == [1]
    with pytest.raises(ValueError, match="already"):
        apply_script_review(store.project_dir, record())


@pytest.mark.parametrize("decision", ["APPROVED", "REVISION_REQUESTED", "REJECTED"])
def test_cli_exact_review_decision_no_generation(tmp_path, capsys, monkeypatch, decision):
    from history_studio.cli import main
    from history_studio.script import ScriptWriter
    store = setup_review(tmp_path)
    monkeypatch.setattr(ScriptWriter, "generate", lambda *args, **kwargs: pytest.fail("No regeneration"))
    assert main(["--projects-dir", str(tmp_path), "review", "project", "script"]) == 0
    assert "Review project/script:v4" in capsys.readouterr().out
    path = tmp_path / "decision.json"
    path.write_text(record(decision).model_dump_json())
    assert main(["--projects-dir", str(tmp_path), "review", "project", "script", "--decision-file", str(path)]) == 0
    assert read(store).current_state == (S.SCRIPT_APPROVED if decision == "APPROVED" else S.SCRIPT_GENERATING)


def test_only_story_and_script_loaded_and_no_provider(tmp_path, monkeypatch):
    from history_studio.storage.artifact_store import ArtifactStore
    from openai.resources.responses import Responses
    store = setup_review(tmp_path)
    original = ArtifactStore.load

    def bounded_load(self, kind, version, contract):
        assert kind in ("script", "story", "approvals")
        return original(self, kind, version, contract)

    monkeypatch.setattr(ArtifactStore, "load", bounded_load)
    monkeypatch.setattr(Responses, "create", lambda *args, **kwargs: pytest.fail("No model calls"))
    assert apply_script_review(store.project_dir, record()).current_state == S.SCRIPT_APPROVED
