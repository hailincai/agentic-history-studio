import json

import pytest
from pydantic import ValidationError

from history_studio.models import CameraMotion, GenerationMethod, ShotFraming, VisualDirectorContext
from history_studio.visual_director import VisualDirector
from history_studio.visual_director.preparation import INSTRUCTIONS, build_context, serialize_context


def context():
    return VisualDirectorContext(script_input_ref=dict(
        project_id="project", artifact_type="script", version=7), title="Approved title", sections=[
            dict(section_id="z", title="Opening", segments=[
                dict(segment_id="z", kind="HISTORICAL", narration="  日期尚不确定  ",
                     grounding=dict(story_beat_id="beat", research_fact_ids=["f2", "f1"])),
                dict(segment_id="a", kind="STRUCTURAL", narration="Next chapter")]),
            dict(section_id="a", title="Closing", segments=[
                dict(segment_id="closing", kind="STRUCTURAL", narration="Conclusion")])])


def decode(prepared):
    assert prepared.startswith(INSTRUCTIONS + "\n")
    return json.loads(prepared[len(INSTRUCTIONS) + 1:])


def test_complete_bounded_model_input():
    source = context()
    data = decode(VisualDirector(source).prepare())
    assert data == source.model_dump(mode="json")
    assert data["script_input_ref"] == dict(project_id="project", artifact_type="script", version=7)
    assert [section["section_id"] for section in data["sections"]] == ["z", "a"]
    historical, structural = data["sections"][0]["segments"]
    assert [historical["segment_id"], structural["segment_id"]] == ["z", "a"]
    assert historical["kind"] == "HISTORICAL"
    assert historical["narration"] == "日期尚不确定"
    assert historical["grounding"] == dict(story_beat_id="beat", research_fact_ids=["f2", "f1"])
    assert structural["kind"] == "STRUCTURAL" and structural["grounding"] is None
    assert "日期尚不确定" in serialize_context(source)


def test_deterministic_detached_nonmutating_preparation():
    source = context()
    before = source.model_dump_json()
    director = VisualDirector(source)
    prepared = director.prepare()
    assert prepared == director.prepare() == VisualDirector(source).prepare() == build_context(source)
    assert source.model_dump_json() == before
    source.title = "Caller title"
    source.sections[0].segments[0].narration = "Caller narration"
    source.sections[0].segments[0].grounding.story_beat_id = "caller_beat"
    assert director.prepare() == prepared


@pytest.mark.parametrize("value", [{}, [], "context", None])
def test_requires_typed_context(value):
    for operation in (VisualDirector, serialize_context, build_context):
        with pytest.raises(TypeError, match="one VisualDirectorContext"):
            operation(value)


def test_revalidates_mutated_nested_context():
    source = context()
    source.sections[1].segments[0].segment_id = "z"
    for operation in (VisualDirector, serialize_context, build_context):
        with pytest.raises(ValidationError, match="Segment IDs"):
            operation(source)


def test_allowed_metadata_matches_contract_enums():
    values = json.loads(INSTRUCTIONS.split("Allowed contract values:\n", 1)[1])
    assert values == dict(generation_method=[v.value for v in GenerationMethod],
                          framing=[v.value for v in ShotFraming], camera_motion=[v.value for v in CameraMotion])
    guidance = " ".join(INSTRUCTIONS.split())
    for method, intent in (("STATIC_IMAGE", "motion adds little value"),
                           ("IMAGE_TO_VIDEO", "controlled visual composition"),
                           ("TEXT_TO_VIDEO", "direct motion generation")):
        assert method in guidance and intent in guidance


def test_no_execution_or_enrichment(monkeypatch):
    from openai.resources.responses import Responses
    from history_studio.storage.artifact_store import ArtifactStore

    def forbidden(*args, **kwargs):
        pytest.fail("Preparation must not call a model or access artifacts")

    monkeypatch.setattr(Responses, "create", forbidden)
    for method in ("load", "load_latest", "list_versions", "save"):
        monkeypatch.setattr(ArtifactStore, method, forbidden)
    director = VisualDirector(context())
    data = decode(director.prepare())
    assert set(data) == {"script_input_ref", "title", "sections"}
    for section in data["sections"]:
        assert set(section) == {"section_id", "title", "segments"}
        for segment in section["segments"]:
            assert set(segment) == {"segment_id", "kind", "narration", "grounding"}
            if segment["grounding"]:
                assert set(segment["grounding"]) == {"story_beat_id", "research_fact_ids"}
    for name in ("provider", "generate", "tool_definitions", "submit_storyboard", "finalize_submission",
                 "estimate_duration", "validate_grounding", "run"):
        assert not hasattr(director, name)


@pytest.mark.parametrize("fragments", [
    ("visual freedom", "not factual freedom", "complete immediate semantic authority"),
    ("Do not research", "general historical knowledge or memory", "provenance markers"),
    ("Do not choose fact subsets", "new ResearchFact IDs", "shot -> source_segment_id"),
    ("1..N shots", "every segment must receive at least one planned shot", "exactly one source_segment_id"),
    ("Do not merge multiple Script segments", "Preserve approved section IDs", "Script section/segment order"),
    ("HISTORICAL Script segments must produce HISTORICAL shots", "STRUCTURAL Script segments must produce STRUCTURAL shots"),
    ("Ordinary cinematic detail is allowed", "composition", "lighting", "Every visible pixel need not"),
    ("unsupported historical propositions", "family relationships", "motives", "emotions presented as historical fact"),
    ("generic, non-identifying, period-compatible", "historically restrained", "Preserve narration's uncertainty"),
    ("you may depict that person", "identity does not establish appearance or behavior", "clothing rank"),
    ("Structural shots are not a backdoor", "organizational maps", "unsupported geographic claims"),
    ("visual_description is a concise human-readable", "generation_prompt is a detailed media-generation instruction",
     "must not smuggle in new historical propositions"),
    ("only from the contract values", "creative metadata, not historical grounding"),
    ("planning-only, not actual TTS duration", "Do not require mathematical equality", "Actual timing belongs to Phase 7"),
    ("Later Runtime must authenticate", "coverage and order", "Human Storyboard Gate", "do not prove historical truth"),
])
def test_required_instruction_semantics(fragments):
    guidance = " ".join(INSTRUCTIONS.split())
    assert all(fragment in guidance for fragment in fragments)
