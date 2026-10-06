import json

import pytest

from history_studio.models import StoryboardPackage
from history_studio.workflow import RuntimeState, ProjectState as S, ApprovalRecord, ProjectStateMachine
from history_studio.workflow.storyboard_review import (
    load_storyboard_review, storyboard_review_lines, apply_storyboard_review,
)
from history_studio.visual_director import build_visual_director_context
from history_studio.storage.artifact_store import write_json
from test_storyboard_workflow import prepare, read, workflow


def setup_review(tmp_path):
    project, store, _ = prepare(tmp_path)
    workflow().run(project, store)
    original = store.load("storyboard", 1, StoryboardPackage)
    for _ in range(4):
        store.save("storyboard", original)
    data = read(store).model_dump(mode="json")
    data["artifacts"]["storyboard"]["version"] = 4
    data["artifacts"]["approved_storyboard"] = dict(project_id="project", artifact_type="storyboard", version=1)
    write_json(store.project_dir / ".runtime/state.json", RuntimeState.model_validate(data), replace=True)
    return store


def record(decision="APPROVED", version=4):
    return ApprovalRecord(project_id="project", stage="storyboard", artifact_type="storyboard", artifact_version=version,
        decision=decision, feedback="Human reviewed visual semantics", decided_at="2026-10-05T12:00:00Z",
        decided_by="Reviewer", decision_source="human")


@pytest.mark.parametrize("decision,destination", [("APPROVED", S.STORYBOARD_APPROVED),
    ("REVISION_REQUESTED", S.STORYBOARD_GENERATING), ("REJECTED", S.STORYBOARD_GENERATING)])
def test_exact_decision_immutable_history_and_upstream_preservation(tmp_path, monkeypatch, decision, destination):
    from history_studio.storage.artifact_store import ArtifactStore
    store = setup_review(tmp_path)
    before = read(store)
    original = store.load("storyboard", 4, StoryboardPackage)
    monkeypatch.setattr(ArtifactStore, "load_latest", lambda *args: pytest.fail("No latest lookup"))
    result = ProjectStateMachine(before).apply_human_decision(record(decision), store)
    assert result.current_state == destination
    assert result.artifacts.storyboard == before.artifacts.storyboard
    assert result.artifacts.approved_storyboard == (before.artifacts.storyboard if decision == "APPROVED" else None)
    for field in ("research", "verification", "approved_verification", "story", "approved_story", "script", "approved_script"):
        assert getattr(result.artifacts, field) == getattr(before.artifacts, field)
    assert store.list_versions("storyboard") == [1, 2, 3, 4, 5]
    assert store.load("storyboard", 4, StoryboardPackage) == original
    assert store.load("approvals", 1, ApprovalRecord) == record(decision)


def test_exact_review_load_and_render_no_upstream_or_discovery(tmp_path, monkeypatch):
    from history_studio.storage.artifact_store import ArtifactStore
    from openai.resources.responses import Responses
    store = setup_review(tmp_path)
    state = read(store)
    load = ArtifactStore.load
    calls = []

    def exact_load(self, kind, version, model):
        assert (kind, version) in (("script", 1), ("storyboard", 4))
        calls.append((kind, version))
        return load(self, kind, version, model)

    def forbidden(*args, **kwargs):
        pytest.fail("No discovery, model or generation")

    monkeypatch.setattr(ArtifactStore, "load", exact_load)
    monkeypatch.setattr(ArtifactStore, "load_latest", forbidden)
    monkeypatch.setattr(ArtifactStore, "list_versions", forbidden)
    monkeypatch.setattr(Responses, "create", forbidden)
    package = load_storyboard_review(state, store)
    context = build_visual_director_context(store, script_input_ref=state.artifacts.approved_script)
    lines = storyboard_review_lines(state, package, context)
    assert lines[0] == "Review project/storyboard:v4"
    assert "Script provenance: project/script:v1" in lines
    text = "\n".join(lines)
    for fragment in ("Section z", "Source segment z", "日期尚不确定", "Shot shot1", "HISTORICAL",
                     "STRUCTURAL", "A traveler with an invented motive", "Unverified costume",
                     "IMAGE_TO_VIDEO", "WIDE", "PUSH_IN", "3.5", "Story beat: beat", "Fact: f2", "Fact: f1"):
        assert fragment in text
    assert text.index("Shot shot1") < text.index("Shot shot2") < text.index("Shot shot3")
    assert calls == [("storyboard", 4), ("script", 1), ("script", 1)]


@pytest.mark.parametrize("failure", ["state", "missing", "approved_missing", "type", "project", "version"])
def test_review_fails_closed(tmp_path, failure):
    store = setup_review(tmp_path)
    data = read(store).model_dump(mode="json")
    if failure == "state":
        data["current_state"] = "STORYBOARD_GENERATING"
    elif failure == "missing":
        data["artifacts"]["storyboard"] = None
    elif failure == "approved_missing":
        data["artifacts"]["approved_script"] = None
    else:
        key, value = {"type": ("artifact_type", "script"), "project": ("project_id", "foreign"),
                      "version": ("version", 99)}[failure]
        data["artifacts"]["storyboard"][key] = value
    with pytest.raises((ValueError, OSError)):
        load_storyboard_review(RuntimeState.model_validate(data), store)


@pytest.mark.parametrize("changes", [{"artifact_version": 3}, {"artifact_version": 5},
    {"project_id": "foreign"}, {"stage": "script", "artifact_type": "script"}, {"decision_source": "agent"}])
def test_stale_wrong_and_malformed_decisions(tmp_path, changes):
    store = setup_review(tmp_path)
    before = read(store)
    with pytest.raises(ValueError):
        apply_storyboard_review(store.project_dir, record().model_copy(update=changes))
    assert read(store) == before and store.list_versions("approvals") == []


@pytest.mark.parametrize("failure", ["provenance", "segment", "kind", "coverage", "title", "json"])
def test_corrupt_bound_artifact_cannot_be_reviewed_or_approved(tmp_path, failure):
    store = setup_review(tmp_path)
    data = store.load("storyboard", 4, StoryboardPackage).model_dump(mode="json")
    if failure == "provenance":
        data["script_input_ref"]["version"] = 2
    elif failure == "segment":
        data["sections"][0]["shots"][0]["source_segment_id"] = "unknown"
    elif failure == "kind":
        data["sections"][0]["shots"][0]["kind"] = "STRUCTURAL"
    elif failure == "coverage":
        data["sections"][0]["shots"].pop()
    elif failure == "title":
        data["title"] = "Changed"
    path = store.project_dir / "storyboard/storyboard_v4.json"
    path.write_text("{" if failure == "json" else json.dumps(data))
    before = read(store)
    with pytest.raises(ValueError):
        load_storyboard_review(before, store)
    with pytest.raises(ValueError):
        apply_storyboard_review(store.project_dir, record())
    assert read(store) == before and not store.list_versions("approvals")


def test_reloads_integrity_before_decision_and_rejects_changed_candidate(tmp_path):
    store = setup_review(tmp_path)
    load_storyboard_review(read(store), store)
    data = read(store).model_dump(mode="json")
    data["artifacts"]["storyboard"]["version"] = 5
    write_json(store.project_dir / ".runtime/state.json", RuntimeState.model_validate(data), replace=True)
    with pytest.raises(ValueError, match="exact bound"):
        apply_storyboard_review(store.project_dir, record(version=4))
    assert not store.list_versions("approvals")


def test_audit_state_crash_requires_reconciliation(tmp_path, monkeypatch):
    import history_studio.workflow.storyboard_review as module
    store = setup_review(tmp_path)
    data = read(store).model_dump(mode="json")
    data["artifacts"]["approved_storyboard"] = None
    write_json(store.project_dir / ".runtime/state.json", RuntimeState.model_validate(data), replace=True)
    monkeypatch.setattr(module, "write_json", lambda *args, **kwargs: (_ for _ in ()).throw(OSError("state failed")))
    with pytest.raises(OSError):
        apply_storyboard_review(store.project_dir, record())
    assert read(store).current_state == S.WAITING_STORYBOARD_APPROVAL
    assert read(store).artifacts.approved_storyboard is None
    assert store.list_versions("approvals") == [1]
    with pytest.raises(ValueError, match="already"):
        apply_storyboard_review(store.project_dir, record())


@pytest.mark.parametrize("decision", ["APPROVED", "REVISION_REQUESTED", "REJECTED"])
def test_cli_exact_review_and_external_human_decision(tmp_path, capsys, monkeypatch, decision):
    from history_studio.cli import main
    from history_studio.visual_director import VisualDirector
    store = setup_review(tmp_path)
    monkeypatch.setattr(VisualDirector, "generate", lambda *args, **kwargs: pytest.fail("No generation"))
    assert main(["--projects-dir", str(tmp_path), "review", "project", "storyboard"]) == 0
    assert "Review project/storyboard:v4" in capsys.readouterr().out
    assert store.list_versions("approvals") == []
    path = tmp_path / "decision.json"
    path.write_text(record(decision).model_dump_json())
    assert main(["--projects-dir", str(tmp_path), "review", "project", "storyboard", "--decision-file", str(path)]) == 0
    assert read(store).current_state == (S.STORYBOARD_APPROVED if decision == "APPROVED" else S.STORYBOARD_GENERATING)


def test_render_escapes_and_bounds_prose(tmp_path):
    store = setup_review(tmp_path)
    state = read(store)
    package = load_storyboard_review(state, store)
    package.sections[0].shots[0].generation_prompt = "Line\nForged label\x1b" + "x" * 1100
    context = build_visual_director_context(store, script_input_ref=state.artifacts.approved_script)
    line = next(line for line in storyboard_review_lines(state, package, context) if line.startswith("Generation prompt:"))
    assert "\\n" in line and "\\u001b" in line and "[truncated]" in line
    assert "\n" not in line
