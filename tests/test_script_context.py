import json

import pytest
from pydantic import ValidationError

from history_studio.models import (
    ArtifactReference, HistoricalTime, ScriptContext, ScriptContextBeat, ScriptContextSection,
    StoryFactChronology, StoryFactReference, StoryFactUse, StoryNarrativeBeat,
    StoryPackage, StorySection, StoryStructure, VerificationStatus,
)
from history_studio.script import build_script_context
from history_studio.storage.artifact_store import ArtifactNotFoundError, ArtifactStore


def setup(tmp_path):
    store = ArtifactStore(tmp_path / "project")
    refs = [StoryFactReference(research_fact_id=fact_id, status=status, use=use,
                              qualification=qualification) for fact_id, status, use, qualification in (
        ("f1", "VERIFIED", "AFFIRMATIVE", None),
        ("f2", "PARTIALLY_VERIFIED", "QUALIFIED", "Exact approved qualification"),
        ("f3", "DISPUTED", "DISPUTE", "Exact approved dispute"))]
    times = [HistoricalTime(display="Unknown", precision="UNKNOWN"),
             HistoricalTime(display="circa 700", precision="APPROXIMATE"),
             HistoricalTime(display="701–725", start_year=701, end_year=725, precision="RANGE")]
    beats = [StoryNarrativeBeat(beat_id="b1", narrative_role="Opening", summary="Approved framing",
        fact_refs=refs[:2], fact_chronology=[StoryFactChronology(
            research_fact_id="f2", historical_time=times[1]), StoryFactChronology(
            research_fact_id="f1", historical_time=times[0])], uncertainty_notes=["Approved uncertainty"]),
        StoryNarrativeBeat(beat_id="b2", narrative_role="Dispute", summary="Approved dispute framing",
            fact_refs=refs[2:], fact_chronology=[StoryFactChronology(
                research_fact_id="f3", historical_time=times[2])]),
        StoryNarrativeBeat(beat_id="transition", kind="structural", narrative_role="Transition",
            summary="Move to the next chapter")]
    story = StoryPackage(verification_input_ref=ArtifactReference(
        project_id="project", artifact_type="verification", version=99),
        plan=StoryStructure(title="Approved title", narrative_thesis="Approved thesis", sections=[
            StorySection(section_id="s1", purpose="Opening purpose", beats=beats[:1]),
            StorySection(section_id="s2", purpose="Closing purpose", beats=beats[1:])]))
    store.save("story", story)
    return store, ArtifactReference(project_id="project", artifact_type="story", version=1), story


def test_exact_projection_and_authorized_beat_membership(tmp_path):
    store, ref, story = setup(tmp_path)
    context = build_script_context(store, story_input_ref=ref)
    assert context.story_input_ref == ref
    assert context.title == story.plan.title
    assert context.narrative_thesis == story.plan.narrative_thesis
    assert [s.section_id for s in context.sections] == ["s1", "s2"]
    assert [s.purpose for s in context.sections] == [s.purpose for s in story.plan.sections]
    for projected_section, source_section in zip(context.sections, story.plan.sections):
        assert isinstance(projected_section, ScriptContextSection)
        for projected, source in zip(projected_section.beats, source_section.beats):
            assert isinstance(projected, ScriptContextBeat)
            assert projected.model_dump() == source.model_dump()
            assert projected is not source
    first = context.sections[0].beats[0]
    assert [f.research_fact_id for f in first.fact_refs] == ["f1", "f2"]
    assert first.fact_refs[0].status == VerificationStatus.VERIFIED
    assert first.fact_refs[0].use == StoryFactUse.AFFIRMATIVE
    assert first.fact_refs[1].status == VerificationStatus.PARTIALLY_VERIFIED
    assert first.fact_refs[1].use == StoryFactUse.QUALIFIED
    assert first.fact_refs[1].qualification == "Exact approved qualification"
    disputed = context.sections[1].beats[0].fact_refs[0]
    assert (disputed.status, disputed.use, disputed.qualification) == (
        VerificationStatus.DISPUTED, StoryFactUse.DISPUTE, "Exact approved dispute")
    assert first.uncertainty_notes == ("Approved uncertainty",)
    assert first.fact_chronology == story.plan.sections[0].beats[0].fact_chronology
    assert [c.research_fact_id for c in first.fact_chronology] == ["f2", "f1"]
    assert first.fact_chronology[0].historical_time != first.fact_chronology[1].historical_time
    assert "historical_time" not in first.model_dump()
    structural = context.sections[1].beats[1]
    assert structural.beat_id == "transition" and structural.kind == "structural"
    assert structural.fact_refs == structural.fact_chronology == ()
    assert ScriptContext.model_validate_json(context.model_dump_json()) == context


def test_only_exact_story_load_no_discovery_or_upstream(tmp_path, monkeypatch):
    store, ref, _ = setup(tmp_path)
    load = store.load
    calls = []

    def exact_load(artifact_type, version, model_type):
        calls.append((artifact_type, version, model_type))
        assert artifact_type == "story"
        return load(artifact_type, version, model_type)

    def forbidden(*args, **kwargs):
        pytest.fail("Context must not discover artifact versions")

    monkeypatch.setattr(store, "load", exact_load)
    monkeypatch.setattr(store, "load_latest", forbidden)
    monkeypatch.setattr(store, "list_versions", forbidden)
    first = build_script_context(store, story_input_ref=ref)
    second = build_script_context(store, story_input_ref=ref)
    assert first == second
    assert calls == [("story", 1, StoryPackage)] * 2
    # Upstream verification v99 and every ResearchPackage are absent.
    assert not (store.project_dir / "verification").exists()
    assert not (store.project_dir / "research").exists()


def test_newer_story_cannot_replace_exact_reference(tmp_path):
    store, ref, story = setup(tmp_path)
    newer = story.model_copy(deep=True)
    newer.plan.title = "Newer title"
    assert store.save("story", newer) == 2
    assert build_script_context(store, story_input_ref=ref).title == "Approved title"
    context = build_script_context(store, story_input_ref=ref.model_copy(update={"version": 2}))
    assert context.story_input_ref.version == 2
    assert story.schema_version == 1
    assert "schema_version" not in context.model_dump()
    (store.project_dir / "story" / "story_v1.json").unlink()
    with pytest.raises(ArtifactNotFoundError):
        build_script_context(store, story_input_ref=ref)


@pytest.mark.parametrize("changes,error", [
    ({"artifact_type": "verification"}, ValueError),
    ({"artifact_type": "research"}, ValueError),
    ({"project_id": "other"}, ValueError),
    ({"version": 99}, ArtifactNotFoundError),
    ({"version": True}, ValidationError),
])
def test_invalid_reference_fails_closed(tmp_path, changes, error):
    store, ref, _ = setup(tmp_path)
    with pytest.raises(error):
        build_script_context(store, story_input_ref=ref.model_copy(update=changes))


def test_detachment_including_nested_values(tmp_path, monkeypatch):
    store, ref, story = setup(tmp_path)
    monkeypatch.setattr(store, "load", lambda *args: story)
    context = build_script_context(store, story_input_ref=ref)
    assert context is not story and context is not story.plan
    assert context.story_input_ref is not ref
    beat = context.sections[0].beats[0]
    beat.fact_refs[1].qualification = "Working copy change"
    beat.fact_chronology[0].historical_time.display = "Working copy date"
    assert story.plan.sections[0].beats[0].fact_refs[1].qualification == "Exact approved qualification"
    assert story.plan.sections[0].beats[0].fact_chronology[0].historical_time.display == "circa 700"
    assert build_script_context(store, story_input_ref=ref).sections[0].beats[0].model_dump() == (
        story.plan.sections[0].beats[0].model_dump())


def test_context_boundary(tmp_path):
    store, ref, _ = setup(tmp_path)
    data = build_script_context(store, story_input_ref=ref).model_dump(mode="json")
    assert set(data) == {"story_input_ref", "title", "narrative_thesis", "sections", "production_brief"}
    assert data["production_brief"] is None
    payload = json.dumps(data)
    for hidden in ("verification_input_ref", "claim_snapshot", "sources", "evidence",
                   "research_confidence", "research_plan", "runtime_state", "workflow_history",
                   "model_responses", "transcripts", "schema_version", "f99"):
        assert hidden not in payload
    for hidden in ("research_package", "verification_package", "runtime_state"):
        with pytest.raises(ValidationError, match="Extra inputs"):
            ScriptContext.model_validate(data | {hidden: {}})


@pytest.mark.parametrize("failure", ["section_ids", "beat_ids", "missing_refs", "chronology",
                                     "structural", "project", "status"])
def test_corrupt_story_fails_closed_without_repair(tmp_path, failure):
    store, ref, story = setup(tmp_path)
    data = story.model_dump(mode="json")
    sections = data["plan"]["sections"]
    beat = sections[0]["beats"][0]
    if failure == "section_ids":
        sections[1]["section_id"] = sections[0]["section_id"]
    elif failure == "beat_ids":
        sections[1]["beats"][0]["beat_id"] = beat["beat_id"]
    elif failure == "missing_refs":
        beat["fact_refs"] = []
    elif failure == "chronology":
        beat["fact_chronology"].pop()
    elif failure == "structural":
        beat["kind"] = "structural"
    elif failure == "project":
        data["verification_input_ref"]["project_id"] = "other"
    else:
        beat["fact_refs"][0]["status"] = "REJECTED"
    path = store.project_dir / "story" / "story_v1.json"
    content = json.dumps(data)
    path.write_text(content, encoding="utf-8")
    with pytest.raises(ValueError):
        build_script_context(store, story_input_ref=ref)
    assert path.read_text(encoding="utf-8") == content


def test_context_reuses_story_internal_shape_rules(tmp_path):
    store, ref, _ = setup(tmp_path)
    data = build_script_context(store, story_input_ref=ref).model_dump(mode="json")
    data["sections"][1]["beats"][0]["beat_id"] = "b1"
    with pytest.raises(ValidationError, match="Beat IDs"):
        ScriptContext.model_validate(data)
