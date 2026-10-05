import json

import pytest

from history_studio.models import ArtifactReference, StoryPackage
from history_studio.workflow import RuntimeState, ProjectState as S, ApprovalRecord, ProjectStateMachine
from history_studio.workflow.story import StoryWorkflow
from history_studio.workflow.story_review import load_story_review, story_review_lines, apply_story_review
from history_studio.storage.artifact_store import write_json
from test_story_workflow import prepare, read, terminal
from test_story_generation import FakeProvider


def setup_review(tmp_path):
    project, store, _ = prepare(tmp_path)
    StoryWorkflow(provider_factory=lambda ctx: FakeProvider([terminal()])).run(project, store)
    package = store.load("story", 1, StoryPackage)
    for _ in range(4):
        store.save("story", package)
    state = read(store)
    data = state.model_dump(mode="json")
    data["artifacts"]["story"]["version"] = 4
    write_json(store.project_dir / ".runtime/state.json", RuntimeState.model_validate(data), replace=True)
    return store


def record(decision="APPROVED", version=4):
    return ApprovalRecord(project_id="test", stage="story", artifact_type="story", artifact_version=version,
        decision=decision, feedback="Human reviewed this blueprint", decided_at="2026-10-05T12:00:00Z",
        decided_by="Reviewer", decision_source="human")


@pytest.mark.parametrize("decision,destination", [("APPROVED", S.STORY_APPROVED),
    ("REVISION_REQUESTED", S.STORY_GENERATING), ("REJECTED", S.STORY_GENERATING)])
def test_exact_review_decision_audit_and_immutable_artifacts(tmp_path, monkeypatch, decision, destination):
    store = setup_review(tmp_path)
    before = read(store)
    original = store.load("story", 4, StoryPackage)
    monkeypatch.setattr(store, "load_latest", lambda *args: pytest.fail("No latest lookup"))
    package = load_story_review(before, store)
    lines = story_review_lines(before, package)
    assert lines[0] == "Review test/story:v4"
    assert any("verification:v2" in line for line in lines)
    assert any("RF-target" in line and "VERIFIED" in line for line in lines)
    assert any("Chronology RF-target" in line for line in lines)
    result = apply_story_review(store.project_dir, record(decision))
    assert result.current_state == destination and result.artifacts.story == before.artifacts.story
    assert result.artifacts.approved_story == (before.artifacts.story if decision == "APPROVED" else None)
    assert result.artifacts.approved_verification == before.artifacts.approved_verification
    assert result.artifacts.research == before.artifacts.research
    assert result.artifacts.verification == before.artifacts.verification
    assert read(store) == result
    assert store.list_versions("story") == [1, 2, 3, 4, 5]
    assert store.load("story", 4, StoryPackage) == original
    audit = store.load("approvals", 1, ApprovalRecord)
    assert audit == record(decision) and audit.artifact_version == 4


@pytest.mark.parametrize("failure", ["state", "missing", "type", "project", "version"])
def test_review_fails_closed(tmp_path, failure):
    store = setup_review(tmp_path)
    data = read(store).model_dump(mode="json")
    if failure == "state":
        data["current_state"] = "STORY_GENERATING"
    elif failure == "missing":
        data["artifacts"]["story"] = None
    else:
        field, value = {"type": ("artifact_type", "research"), "project": ("project_id", "foreign"),
                        "version": ("version", 99)}[failure]
        data["artifacts"]["story"][field] = value
    with pytest.raises((ValueError, OSError)):
        load_story_review(RuntimeState.model_validate(data), store)


@pytest.mark.parametrize("failure", ["provenance", "status", "chronology", "unknown"])
def test_inconsistent_exact_story_fails_closed(tmp_path, failure):
    store = setup_review(tmp_path)
    package = store.load("story", 4, StoryPackage).model_dump(mode="json")
    beat = package["plan"]["sections"][0]["beats"][0]
    if failure == "provenance":
        package["verification_input_ref"]["version"] = 3
    elif failure == "status":
        beat["fact_refs"][0].update(status="PARTIALLY_VERIFIED", use="QUALIFIED", qualification="hedge")
    elif failure == "chronology":
        beat["fact_chronology"][0]["historical_time"]["display"] = "Invented date"
    else:
        beat["fact_refs"][0]["research_fact_id"] = "unknown"
        beat["fact_chronology"][0]["research_fact_id"] = "unknown"
    (store.project_dir / "story/story_v4.json").write_text(json.dumps(package))
    with pytest.raises(ValueError):
        apply_story_review(store.project_dir, record())
    assert not store.list_versions("approvals")


def test_stale_decision_and_changed_state_rejected(tmp_path):
    store = setup_review(tmp_path)
    with pytest.raises(ValueError, match="exact bound"):
        apply_story_review(store.project_dir, record(version=5))
    assert not store.list_versions("approvals")
    apply_story_review(store.project_dir, record())
    with pytest.raises(ValueError):
        apply_story_review(store.project_dir, record())


def test_revision_clears_stale_downstream_bindings(tmp_path):
    store = setup_review(tmp_path)
    data = read(store).model_dump(mode="json")
    for field in ("approved_story", "script", "approved_script", "storyboard", "approved_storyboard"):
        data["artifacts"][field] = dict(project_id="test", artifact_type=field.removeprefix("approved_"), version=1)
    write_json(store.project_dir / ".runtime/state.json", RuntimeState.model_validate(data), replace=True)
    result = apply_story_review(store.project_dir, record("REVISION_REQUESTED"))
    assert all(getattr(result.artifacts, field) is None for field in
        ("approved_story", "script", "approved_script", "storyboard", "approved_storyboard"))


def test_audit_state_crash_requires_reconciliation_not_replay(tmp_path, monkeypatch):
    import history_studio.workflow.story_review as module
    store = setup_review(tmp_path)
    monkeypatch.setattr(module, "write_json", lambda *args, **kwargs: (_ for _ in ()).throw(OSError("state failure")))
    with pytest.raises(OSError):
        apply_story_review(store.project_dir, record())
    assert read(store).current_state == S.WAITING_STORY_APPROVAL
    assert read(store).artifacts.approved_story is None
    assert store.list_versions("approvals") == [1]
    with pytest.raises(ValueError, match="already"):
        apply_story_review(store.project_dir, record())


def test_cli_exact_review_and_decision_delegation(tmp_path, capsys, monkeypatch):
    from history_studio.cli import main
    import history_studio.workflow.story_review as module
    store = setup_review(tmp_path)
    assert main(["--projects-dir", str(tmp_path), "review", "test", "story"]) == 0
    assert "Review test/story:v4" in capsys.readouterr().out
    path = tmp_path / "decision.json"
    path.write_text(record().model_dump_json())
    real = module.apply_story_review
    calls = []
    def spy(directory, decision):
        calls.append(decision)
        return real(directory, decision)
    monkeypatch.setattr(module, "apply_story_review", spy)
    assert main(["--projects-dir", str(tmp_path), "review", "test", "story", "--decision-file", str(path)]) == 0
    assert calls == [record()] and read(store).current_state == S.STORY_APPROVED
