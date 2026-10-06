import pytest
from pydantic import ValidationError

from history_studio.models import StoryboardPackage
from history_studio.visual_director import StoryboardSubmission, finalize_storyboard_submission
from test_visual_director_preparation import context


def shot(shot_id="shot1", source_segment_id="z", kind="HISTORICAL", **changes):
    return dict(shot_id=shot_id, source_segment_id=source_segment_id, kind=kind,
                visual_description="  A traveler with an invented motive  ",
                generation_prompt="Unverified costume, named location, dialogue and emotions",
                generation_method="IMAGE_TO_VIDEO", framing="WIDE", camera_motion="PUSH_IN",
                estimated_duration_seconds=3.5) | changes


def submission():
    return StoryboardSubmission(title="Proposed title", sections=[
        dict(section_id="z", title="Proposed opening", shots=[shot(), shot("shot2", "a", "STRUCTURAL")]),
        dict(section_id="a", title="Proposed closing", shots=[shot("shot3", "closing", "STRUCTURAL")])])


def finalize_data(data):
    return finalize_storyboard_submission(context(), StoryboardSubmission.model_validate(data))


def test_valid_finalization_canonical_titles_provenance_and_detachment():
    source, proposal = context(), submission()
    before = source.model_dump_json(), proposal.model_dump_json()
    result = finalize_storyboard_submission(source, proposal)
    assert isinstance(result, StoryboardPackage)
    assert result.script_input_ref == source.script_input_ref
    assert result.script_input_ref is not source.script_input_ref
    assert result.script_input_ref.version == 7 and result.schema_version == 1
    assert result.title == source.title != proposal.title
    assert [s.title for s in result.sections] == [s.title for s in source.sections]
    for accepted, proposed in zip(result.sections, proposal.sections):
        assert [s.model_dump() for s in accepted.shots] == [s.model_dump() for s in proposed.shots]
    assert before == (source.model_dump_json(), proposal.model_dump_json())
    assert StoryboardPackage.model_validate_json(result.model_dump_json()) == result
    assert finalize_storyboard_submission(source, proposal) == result
    result.sections[0].shots[0].generation_prompt = "Output change"
    assert proposal.sections[0].shots[0].generation_prompt != "Output change"


def test_minimal_one_section_one_segment():
    source = context()
    data = source.model_dump(mode="json")
    data["sections"] = data["sections"][:1]
    data["sections"][0]["segments"] = data["sections"][0]["segments"][:1]
    source = type(source).model_validate(data)
    proposal = StoryboardSubmission(title="Proposal", sections=[
        dict(section_id="z", title="Proposal section", shots=[shot()])])
    assert len(finalize_storyboard_submission(source, proposal).sections[0].shots) == 1


def test_multiple_shots_per_segment_preserve_agent_order():
    data = submission().model_dump(mode="json")
    data["sections"][0]["shots"].insert(1, shot("earlier_id"))
    result = finalize_data(data)
    assert [s.shot_id for s in result.sections[0].shots] == ["shot1", "earlier_id", "shot2"]
    assert [s.source_segment_id for s in result.sections[0].shots] == ["z", "z", "a"]


@pytest.mark.parametrize("failure,message", [
    ("unknown_section", "section must exist"), ("missing_section", "every approved Script section"),
    ("section_order", "section order"), ("missing_segment", "every approved Script segment"),
    ("unknown_segment", "submitted Script section"), ("wrong_section", "submitted Script section"),
    ("historical_kind", "segment kind"), ("structural_kind", "segment kind"),
    ("backward", "segment order"), ("reordered", "segment order"),
])
def test_authority_failures(failure, message):
    data = submission().model_dump(mode="json")
    first = data["sections"][0]
    if failure == "unknown_section":
        first["section_id"] = "unknown"
    elif failure == "missing_section":
        data["sections"].pop()
    elif failure == "section_order":
        data["sections"].reverse()
    elif failure == "missing_segment":
        first["shots"].pop()
    elif failure == "unknown_segment":
        first["shots"][0]["source_segment_id"] = "unknown"
    elif failure == "wrong_section":
        first["shots"][0]["source_segment_id"] = "closing"
    elif failure == "historical_kind":
        first["shots"][0]["kind"] = "STRUCTURAL"
    elif failure == "structural_kind":
        first["shots"][1]["kind"] = "HISTORICAL"
    elif failure == "backward":
        first["shots"].append(shot("returning"))
    else:
        first["shots"].reverse()
    proposal = StoryboardSubmission.model_validate(data)
    before = proposal.model_dump_json()
    with pytest.raises(ValueError, match=message):
        finalize_storyboard_submission(context(), proposal)
    assert proposal.model_dump_json() == before


@pytest.mark.parametrize("failure", ["section", "local_shot", "global_shot"])
def test_duplicate_ids(failure):
    data = submission().model_dump(mode="json")
    if failure == "section":
        data["sections"][1]["section_id"] = "z"
    elif failure == "local_shot":
        data["sections"][0]["shots"][1]["shot_id"] = "shot1"
    else:
        data["sections"][1]["shots"][0]["shot_id"] = "shot1"
    with pytest.raises(ValidationError, match="must be unique"):
        StoryboardSubmission.model_validate(data)


@pytest.mark.parametrize("level,field,value", [
    ("package", "script_input_ref", {}), ("package", "version", 99),
    ("package", "schema_version", 1), ("package", "approval", {}),
    ("shot", "story_beat_id", "beat"), ("shot", "research_fact_ids", ["f1"]),
    ("shot", "status", "VERIFIED"), ("shot", "evidence", []),
    ("shot", "historical_time", {}), ("shot", "script_input_ref", {}),
])
def test_proposal_cannot_author_authoritative_metadata(level, field, value):
    data = submission().model_dump(mode="json")
    target = data if level == "package" else data["sections"][0]["shots"][0]
    target[field] = value
    with pytest.raises(ValidationError, match="Extra inputs"):
        StoryboardSubmission.model_validate(data)


@pytest.mark.parametrize("changes", [
    {"source_segment_id": ""}, {"source_segment_id": None},
    {"visual_description": " "}, {"generation_prompt": " "},
    {"estimated_duration_seconds": 0}, {"estimated_duration_seconds": -1},
    {"estimated_duration_seconds": float("inf")}, {"estimated_duration_seconds": float("nan")},
    {"generation_method": "UNKNOWN"}, {"framing": "UNKNOWN"}, {"camera_motion": "UNKNOWN"},
])
def test_local_shape_validation(changes):
    data = submission().model_dump(mode="json")
    data["sections"][0]["shots"][0].update(changes)
    with pytest.raises(ValidationError):
        StoryboardSubmission.model_validate(data)


def test_source_segment_required_and_no_duplicated_authority_in_output():
    data = submission().model_dump(mode="json")
    del data["sections"][0]["shots"][0]["source_segment_id"]
    with pytest.raises(ValidationError):
        StoryboardSubmission.model_validate(data)
    result = finalize_storyboard_submission(context(), submission()).model_dump(mode="json")
    assert set(result) == {"schema_version", "script_input_ref", "title", "sections"}
    assert set(result["sections"][0]["shots"][0]) == {
        "shot_id", "kind", "source_segment_id", "visual_description", "generation_prompt",
        "generation_method", "framing", "camera_motion", "estimated_duration_seconds"}


def test_no_external_access_or_semantic_policing(monkeypatch):
    from history_studio.storage.artifact_store import ArtifactStore
    from openai.resources.responses import Responses

    def forbidden(*args, **kwargs):
        pytest.fail("Finalization must not access artifacts or models")

    for method in ("load", "load_latest", "list_versions", "save"):
        monkeypatch.setattr(ArtifactStore, method, forbidden)
    monkeypatch.setattr(Responses, "create", forbidden)
    result = finalize_storyboard_submission(context(), submission())
    assert result.sections[0].shots[0].visual_description == "A traveler with an invented motive"
    assert result.sections[0].shots[0].generation_prompt == submission().sections[0].shots[0].generation_prompt


def test_unvalidated_copies_and_mutated_context_fail_closed():
    source, proposal = context(), submission()
    for version in (True, "7"):
        invalid_ref = source.script_input_ref.model_copy(update={"version": version})
        with pytest.raises(ValidationError):
            finalize_storyboard_submission(source.model_copy(update={"script_input_ref": invalid_ref}), proposal)
    bad_shot = proposal.sections[0].shots[0].model_copy(update={"estimated_duration_seconds": -1})
    bad_section = proposal.sections[0].model_copy(update={"shots": (bad_shot,)})
    with pytest.raises(ValidationError):
        finalize_storyboard_submission(source, proposal.model_copy(update={"sections": (bad_section,)}))
    source.sections[1].segments[0].segment_id = "z"
    with pytest.raises(ValidationError, match="Segment IDs"):
        finalize_storyboard_submission(source, proposal)


@pytest.mark.parametrize("source,proposal", [(None, submission()), (context(), {})])
def test_requires_typed_inputs(source, proposal):
    with pytest.raises(TypeError, match="VisualDirectorContext and StoryboardSubmission"):
        finalize_storyboard_submission(source, proposal)
