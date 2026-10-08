"""Real creative requests and local publication; no SDK clients or network calls."""
import json

import pytest
from pydantic import ValidationError

from history_studio.models import GenerationMethod, ProductionBrief, ProjectConfig, StoryboardPackage
from history_studio.models.production_brief import load_production_brief
from history_studio.storage.artifact_store import write_json
from history_studio.workflow import ProjectState as S, ProjectStateMachine, InvalidTransitionError
from history_studio.visual_director import StoryboardSubmission, finalize_storyboard_submission, validate_storyboard_integrity
from test_storyboard_submission import submission
from test_visual_director_preparation import context as visual_context


def configure(store, *, methods=("STATIC_IMAGE",)):
    project = ProjectConfig(project_id=store.project_dir.name, topic="李白", language="zh-CN",
                            target_duration_minutes=0.75, allowed_generation_methods=methods)
    write_json(store.project_dir / "project.json", project, replace=True)
    return project


def static_proposal():
    data = submission().model_dump(mode="json")
    for section in data["sections"]:
        for shot in section["shots"]:
            shot["generation_method"] = "STATIC_IMAGE"
    return StoryboardSubmission.model_validate(data)


def test_project_defaults_and_immutable_seconds_derivation(tmp_path):
    legacy = ProjectConfig.model_validate(dict(project_id="li_bai", topic="李白"))
    assert legacy.allowed_generation_methods == tuple(GenerationMethod)
    assert ProductionBrief.from_project(legacy).target_duration_seconds == 300
    project = ProjectConfig(project_id="li_bai", topic="李白", language="zh-CN",
                            target_duration_minutes=0.75, allowed_generation_methods=["STATIC_IMAGE"])
    root = tmp_path / "li_bai"
    write_json(root / "project.json", project)
    brief = load_production_brief(root, project_id="li_bai")
    assert brief == ProductionBrief.from_project(project)
    assert brief.target_duration_seconds == 45 and brief.language == "zh-CN"
    assert brief.allowed_generation_methods == (GenerationMethod.STATIC_IMAGE,)
    with pytest.raises(ValidationError, match="frozen"):
        brief.language = "en"
    with pytest.raises(ValueError, match="exact workflow project"):
        load_production_brief(root, project_id="other")
    raw = project.model_dump(mode="json")
    raw.pop("allowed_generation_methods")
    (root / "project.json").write_text(json.dumps(raw), encoding="utf-8")
    assert load_production_brief(root, project_id="li_bai").allowed_generation_methods == tuple(GenerationMethod)


@pytest.mark.parametrize("methods", [[], ["NOT_A_METHOD"], ["STATIC_IMAGE", "STATIC_IMAGE"]])
def test_invalid_method_policies_rejected(methods):
    with pytest.raises(ValidationError):
        ProjectConfig(project_id="test", topic="李白", allowed_generation_methods=methods)
    with pytest.raises(ValidationError):
        ProductionBrief(topic="李白", language="zh-CN", target_duration_seconds=45,
                        allowed_generation_methods=methods)


@pytest.mark.parametrize("stage", ["story", "script", "storyboard"])
def test_persisted_brief_reaches_real_request_and_keeps_exact_authority(tmp_path, monkeypatch, stage):
    if stage == "story":
        from test_story_workflow import prepare, terminal
        from test_story_generation import FakeProvider
        from history_studio.workflow.story import StoryWorkflow as Workflow
        gate, binding, input_field = S.WAITING_STORY_APPROVAL, "approved_verification", "verification_input_ref"
        _, store, authority = prepare(tmp_path)
        provider = FakeProvider([terminal()])
    elif stage == "script":
        from test_script_workflow import prepare, terminal
        from test_script_generation import FakeProvider
        from history_studio.workflow.script import ScriptWorkflow as Workflow
        gate, binding, input_field = S.WAITING_SCRIPT_APPROVAL, "approved_story", "story_input_ref"
        _, store, authority = prepare(tmp_path)
        provider = FakeProvider([terminal()])
    else:
        from test_storyboard_workflow import prepare
        from test_visual_director_generation import FakeProvider, response, call
        from history_studio.workflow.storyboard import StoryboardWorkflow as Workflow
        gate, binding, input_field = S.WAITING_STORYBOARD_APPROVAL, "approved_script", "script_input_ref"
        _, store, authority = prepare(tmp_path)
        provider = FakeProvider([response([call(static_proposal().model_dump_json())])])
    project = configure(store)
    state_path = store.project_dir / ".runtime/state.json"
    before = json.loads(state_path.read_text(encoding="utf-8"))["artifacts"]
    monkeypatch.setattr(store, "load_latest", lambda *args: pytest.fail("No latest authority"))
    runner = Workflow(provider_factory=lambda ctx: provider)
    # A stale caller's production settings must not override project.json.
    outcome = runner.run(project.model_copy(update={"language": "en", "target_duration_minutes": 20}), store)
    assert outcome.error_type is None
    assert outcome.state.current_state == gate
    request = provider.requests[0]
    payload = json.loads(request.input)
    assert payload["production_brief"] == ProductionBrief.from_project(project).model_dump(mode="json")
    assert payload[input_field] == authority.model_dump(mode="json")
    assert "production_brief" in request.instructions
    assert getattr(outcome.state.artifacts, binding) == authority
    assert getattr(outcome.state.artifacts, f"approved_{stage}") is None
    with pytest.raises(InvalidTransitionError):
        ProjectStateMachine(outcome.state).transition(getattr(S, f"{stage.upper()}_APPROVED"))
    after = outcome.state.artifacts.model_dump(mode="json")
    for field, value in before.items():
        if field != stage:
            assert after[field] == value
    version = getattr(outcome.state.artifacts, stage).version
    persisted = (store.project_dir / stage / f"{stage}_v{version}.json").read_text(encoding="utf-8")
    assert "production_brief" not in persisted
    # Existing bound output remains authoritative after a configuration change.
    state_bytes = state_path.read_bytes()
    configure(store, methods=("TEXT_TO_VIDEO",))
    assert runner.run(project, store).stage is None
    assert len(provider.requests) == 1 and state_path.read_bytes() == state_bytes


@pytest.mark.parametrize("method", ["IMAGE_TO_VIDEO", "TEXT_TO_VIDEO"])
def test_static_only_rejects_video_without_weakening_coverage(method):
    source = visual_context()
    source.production_brief = ProductionBrief.from_project(ProjectConfig(
        project_id="project", topic="李白", allowed_generation_methods=["STATIC_IMAGE"]))
    accepted = finalize_storyboard_submission(source, static_proposal())
    assert validate_storyboard_integrity(source, accepted).is_valid
    data = static_proposal().model_dump(mode="json")
    data["sections"][0]["shots"][0]["generation_method"] = method
    with pytest.raises(ValueError, match=f"{method}.*allowed_generation_methods"):
        finalize_storyboard_submission(source, StoryboardSubmission.model_validate(data))
    unrestricted = visual_context()
    candidate = finalize_storyboard_submission(unrestricted, StoryboardSubmission.model_validate(data))
    report = validate_storyboard_integrity(source, candidate)
    assert [issue.code.value for issue in report.issues] == ["GENERATION_METHOD_NOT_ALLOWED"]
    assert report.issues[0].shot_id == "shot1"
    missing = static_proposal().model_dump(mode="json")
    missing["sections"][0]["shots"].pop()
    with pytest.raises(ValueError, match="cover every"):
        finalize_storyboard_submission(source, StoryboardSubmission.model_validate(missing))


def test_rejected_method_preserves_workflow_and_recovers_with_exact_script(tmp_path):
    from test_storyboard_workflow import prepare
    from test_visual_director_generation import FakeProvider, response, call
    from history_studio.workflow.storyboard import StoryboardWorkflow
    _, store, authority = prepare(tmp_path)
    project = configure(store)
    provider = FakeProvider([response([call()]), response([call(static_proposal().model_dump_json())])])
    runner = StoryboardWorkflow(provider_factory=lambda ctx: provider)
    failed = runner.run(project, store)
    assert failed.state.current_state == S.FAILED and failed.state.failed_state == S.STORYBOARD_GENERATING
    assert failed.error_type == "GenerationMethodNotAllowedError"
    assert failed.validation_report.issues[0].code.value == "GENERATION_METHOD_NOT_ALLOWED"
    assert "IMAGE_TO_VIDEO" in failed.validation_report.issues[0].message
    assert "STATIC_IMAGE" in failed.validation_report.issues[0].message
    assert failed.state.artifacts.approved_script == authority
    assert failed.state.artifacts.storyboard is None and failed.state.artifacts.approved_storyboard is None
    assert store.list_versions("storyboard") == []
    recovered = runner.run(project, store)
    assert recovered.state.current_state == S.WAITING_STORYBOARD_APPROVAL
    assert recovered.state.artifacts.approved_storyboard is None
    package = store.load("storyboard", recovered.state.artifacts.storyboard.version, StoryboardPackage)
    assert package.script_input_ref == authority


def test_legacy_project_allows_broad_methods_in_real_workflow(tmp_path):
    from test_storyboard_workflow import prepare, workflow
    project, store, authority = prepare(tmp_path)
    raw = project.model_dump(mode="json")
    raw.pop("allowed_generation_methods")
    (store.project_dir / "project.json").write_text(json.dumps(raw), encoding="utf-8")
    outcome = workflow().run(project, store)
    assert outcome.state.current_state == S.WAITING_STORYBOARD_APPROVAL
    assert outcome.stage.package.script_input_ref == authority
    assert {shot.generation_method for section in outcome.stage.package.sections for shot in section.shots} == {
        GenerationMethod.IMAGE_TO_VIDEO}
