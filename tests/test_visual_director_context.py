import json

import pytest
from pydantic import ValidationError

from history_studio.models import (
    ArtifactReference, Script, ScriptPackage, ScriptScene, ScriptSection, ScriptSegment,
    ScriptSegmentKind, VisualDirectorContext, VisualDirectorContextSection,
    VisualDirectorContextSegment,
)
from history_studio.storage.artifact_store import ArtifactNotFoundError, ArtifactStore
from history_studio.visual_director import build_visual_director_context


def setup(tmp_path):
    store = ArtifactStore(tmp_path / "project")
    script = ScriptPackage(story_input_ref=ArtifactReference(
        project_id="project", artifact_type="story", version=99), title="Approved title", sections=[
            ScriptSection(section_id="z", title="Opening", segments=[
                ScriptSegment(segment_id="z", kind="HISTORICAL", narration="  Approved narration  ",
                              grounding=dict(story_beat_id="beat", research_fact_ids=["f2", "f1"])),
                ScriptSegment(segment_id="a", kind="STRUCTURAL", narration="Next chapter")]),
            ScriptSection(section_id="a", title="Closing", segments=[
                ScriptSegment(segment_id="middle", kind="HISTORICAL", narration="Closing narration",
                              grounding=dict(story_beat_id="beat2", research_fact_ids=["f3"]))])])
    store.save("script", script)
    return store, ArtifactReference(project_id="project", artifact_type="script", version=1), script


def test_projection_identity_grounding_order_and_serialization(tmp_path):
    store, ref, script = setup(tmp_path)
    context = build_visual_director_context(store, script_input_ref=ref)
    assert isinstance(context, VisualDirectorContext)
    assert context.script_input_ref == ref
    assert context.title == script.title
    assert [sec.section_id for sec in context.sections] == ["z", "a"]
    assert [sec.title for sec in context.sections] == ["Opening", "Closing"]
    assert [seg.segment_id for seg in context.sections[0].segments] == ["z", "a"]
    for projected, source in zip(context.sections, script.sections):
        assert isinstance(projected, VisualDirectorContextSection)
        assert projected.model_dump() == source.model_dump()
        for segment, original in zip(projected.segments, source.segments):
            assert isinstance(segment, VisualDirectorContextSegment)
            assert segment.model_dump() == original.model_dump()
    historical, structural = context.sections[0].segments
    assert historical.kind == ScriptSegmentKind.HISTORICAL
    assert historical.narration == "Approved narration"
    assert historical.grounding.story_beat_id == "beat"
    assert historical.grounding.research_fact_ids == ("f2", "f1")
    assert structural.kind == ScriptSegmentKind.STRUCTURAL
    assert structural.grounding is None
    assert VisualDirectorContext.model_validate_json(context.model_dump_json()) == context
    assert build_visual_director_context(store, script_input_ref=ref).model_dump_json() == context.model_dump_json()


def test_only_exact_script_loaded_without_discovery_or_upstream(tmp_path, monkeypatch):
    store, ref, script = setup(tmp_path)
    newer = script.model_copy(deep=True)
    newer.title = "Newer title"
    assert store.save("script", newer) == 2
    load = store.load
    calls = []

    def exact_load(artifact_type, version, model_type):
        calls.append((artifact_type, version, model_type))
        assert artifact_type == "script"
        return load(artifact_type, version, model_type)

    def forbidden(*args, **kwargs):
        pytest.fail("Context must not discover versions")

    monkeypatch.setattr(store, "load", exact_load)
    monkeypatch.setattr(store, "load_latest", forbidden)
    monkeypatch.setattr(store, "list_versions", forbidden)
    assert build_visual_director_context(store, script_input_ref=ref).title == "Approved title"
    ref2 = ArtifactReference(project_id="project", artifact_type="script", version=2)
    assert build_visual_director_context(store, script_input_ref=ref2).title == "Newer title"
    for artifact_type in ("story", "verification", "research"):
        assert not (store.project_dir / artifact_type).exists()
    assert calls == [("script", 1, ScriptPackage), ("script", 2, ScriptPackage)]


@pytest.mark.parametrize("changes,error", [
    ({"artifact_type": "story"}, ValueError), ({"artifact_type": "Script"}, ValueError),
    ({"project_id": "other"}, ValueError), ({"version": 99}, ArtifactNotFoundError),
    ({"version": True}, ValidationError), ({"version": "1"}, ValidationError),
])
def test_invalid_reference_fails_closed(tmp_path, changes, error):
    store, ref, _ = setup(tmp_path)
    with pytest.raises(error):
        build_visual_director_context(store, script_input_ref=ref.model_copy(update=changes))


def test_missing_exact_version_never_falls_back(tmp_path):
    store, _, script = setup(tmp_path)
    assert store.save("script", script) == 2
    with pytest.raises(ArtifactNotFoundError):
        build_visual_director_context(store, script_input_ref=ArtifactReference(
            project_id="project", artifact_type="script", version=3))


def test_detached_nested_values(tmp_path, monkeypatch):
    store, ref, script = setup(tmp_path)
    monkeypatch.setattr(store, "load", lambda *args: script)
    context = build_visual_director_context(store, script_input_ref=ref)
    assert context.script_input_ref is not ref
    assert context.sections[0] is not script.sections[0]
    segment = context.sections[0].segments[0]
    assert segment.grounding is not script.sections[0].segments[0].grounding
    context.title = "Working title"
    context.sections[0].title = "Working section"
    segment.narration = "Working narration"
    segment.grounding.story_beat_id = "working_beat"
    segment.grounding.research_fact_ids = ("working_fact",)
    assert script.title == "Approved title"
    assert script.sections[0].title == "Opening"
    assert script.sections[0].segments[0].narration == "Approved narration"
    assert script.sections[0].segments[0].grounding.model_dump() == dict(
        story_beat_id="beat", research_fact_ids=("f2", "f1"))


def test_exact_context_boundary(tmp_path):
    store, ref, _ = setup(tmp_path)
    data = build_visual_director_context(store, script_input_ref=ref).model_dump(mode="json")
    assert set(data) == {"script_input_ref", "title", "sections"}
    assert set(data["script_input_ref"]) == {"project_id", "artifact_type", "version"}
    for section in data["sections"]:
        assert set(section) == {"section_id", "title", "segments"}
        for segment in section["segments"]:
            assert set(segment) == {"segment_id", "kind", "narration", "grounding"}
            if segment["grounding"] is not None:
                assert set(segment["grounding"]) == {"story_beat_id", "research_fact_ids"}
    for field in ("story_input_ref", "research_package", "verification_package", "story_package",
                  "source", "evidence", "workflow_state", "approval", "provider", "media_path",
                  "actual_duration_seconds", "cost", "schema_version"):
        with pytest.raises(ValidationError, match="Extra inputs"):
            VisualDirectorContext.model_validate(data | {field: {}})


@pytest.mark.parametrize("failure", ["section_ids", "segment_ids", "grounding", "structural", "project", "json"])
def test_malformed_script_rejected_without_repair(tmp_path, failure):
    store, ref, script = setup(tmp_path)
    data = script.model_dump(mode="json")
    sections = data["sections"]
    if failure == "section_ids":
        sections[1]["section_id"] = sections[0]["section_id"]
    elif failure == "segment_ids":
        sections[1]["segments"][0]["segment_id"] = "z"
    elif failure == "grounding":
        sections[0]["segments"][0]["grounding"] = None
    elif failure == "structural":
        sections[0]["segments"][0]["kind"] = "STRUCTURAL"
    elif failure == "project":
        data["story_input_ref"]["project_id"] = "other"
    content = "{" if failure == "json" else json.dumps(data)
    path = store.project_dir / "script" / "script_v1.json"
    path.write_text(content, encoding="utf-8")
    with pytest.raises(ValueError):
        build_visual_director_context(store, script_input_ref=ref)
    assert path.read_text(encoding="utf-8") == content


def test_context_validates_its_own_shape(tmp_path):
    store, ref, _ = setup(tmp_path)
    data = build_visual_director_context(store, script_input_ref=ref).model_dump(mode="json")
    data["script_input_ref"]["artifact_type"] = "story"
    with pytest.raises(ValidationError, match="script artifact"):
        VisualDirectorContext.model_validate(data)
    data["script_input_ref"]["artifact_type"] = "script"
    data["sections"][1]["segments"][0]["segment_id"] = "z"
    with pytest.raises(ValidationError, match="Segment IDs"):
        VisualDirectorContext.model_validate(data)


def test_legacy_script_is_usable_but_not_authority(tmp_path):
    legacy = Script(title="Legacy", target_duration_seconds=5, scenes=[ScriptScene(
        scene_id="scene", sequence=1, narration="Legacy narration", fact_ids=["fact"], duration_seconds=5)])
    assert Script.model_validate_json(legacy.model_dump_json()) == legacy
    store = ArtifactStore(tmp_path / "project")
    store.save("script", legacy)
    with pytest.raises(ValidationError):
        build_visual_director_context(store, script_input_ref=ArtifactReference(
            project_id="project", artifact_type="script", version=1))
