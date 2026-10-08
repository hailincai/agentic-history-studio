"""P8-A pure planning against explicit exact authorities, entirely offline."""
import builtins
import inspect

import pytest
from pydantic import ValidationError

from history_studio.assembly import build_subtitle_cues, build_timeline_plan, serialize_srt
from history_studio.assembly.planning import allocate_shots, duration_ms
from history_studio.models import (
    ArtifactReference, MediaPackage, ScriptPackage, StoryboardPackage,
    SegmentTiming, ShotTiming, SubtitleCue, TimelinePlan,
)
from test_media_integrity import inputs, reference


def authorities():
    return dict(media_input_ref=reference("media", 7), **inputs())


def single(*, duration=9.3, weights=(4, 2)):
    values = authorities()
    board = values["storyboard"].model_dump(mode="json")
    # Existing fixture's first shot is SEG-A; two later shots are SEG-Z.
    shots = board["sections"][0]["shots"][1:]
    shots = shots[:len(weights)]
    for shot, weight in zip(shots, weights):
        shot["estimated_duration_seconds"] = weight
    board["sections"][0]["shots"] = shots
    values["storyboard"] = StoryboardPackage.model_validate(board)
    media = values["package"].model_dump(mode="json")
    media["narration_assets"] = media["narration_assets"][:1]
    media["narration_assets"][0]["duration_seconds"] = duration
    media["visual_assets"] = media["visual_assets"][1:1 + len(weights)]
    values["package"] = MediaPackage.model_validate(media)
    return values


def cues(values, plan):
    return build_subtitle_cues(plan=plan, script_input_ref=values["script_input_ref"], script=values["script"])


def test_single_segment_single_shot():
    values = single(weights=(4,))
    plan = build_timeline_plan(**values)
    assert [(shot.start_ms, shot.end_ms) for shot in plan.shots] == [(0, 9300)]
    assert plan.total_duration_ms == 9300
    assert len(cues(values, plan)) == 1


def test_weighted_actual_narration_and_exact_srt():
    values = single()
    script = values["script"].model_dump(mode="json")
    script["sections"][0]["segments"][0]["narration"] = "Exact café narration!"
    values["script"] = ScriptPackage.model_validate(script)
    plan = build_timeline_plan(**values)
    assert [(shot.start_ms, shot.end_ms) for shot in plan.shots] == [(0, 6200), (6200, 9300)]
    subtitle = cues(values, plan)
    assert subtitle[0].text == "Exact café narration!"
    assert serialize_srt(subtitle) == "1\n00:00:00,000 --> 00:00:09,300\nExact café narration!\n\n".encode("utf-8")


def test_multiple_segments_mixed_methods_structural_and_presentation_order():
    values = authorities()
    plan = build_timeline_plan(**values)
    assert [(segment.segment_id, segment.start_ms, segment.end_ms) for segment in plan.segments] == [
        ("SEG-Z", 0, 250), ("SEG-A", 250, 500)]
    expected = [shot.shot_id for shot in values["storyboard"].sections[0].shots]
    assert [shot.shot_id for shot in plan.shots] == expected[1:] + expected[:1]
    assert {visual.generation_method.value for visual in values["package"].visual_assets} == {
        "STATIC_IMAGE", "IMAGE_TO_VIDEO", "TEXT_TO_VIDEO"}
    assert {shot.kind.value for shot in values["storyboard"].sections[0].shots} == {"STRUCTURAL"}
    subtitle = cues(values, plan)
    assert [(cue.start_ms, cue.end_ms) for cue in subtitle] == [(0, 250), (250, 500)]
    assert serialize_srt(subtitle).count(b" --> ") == 2


@pytest.mark.parametrize("seconds,expected", [(0.0005, 1), (1.2345, 1235), (9.3, 9300), (0.00149, 1)])
def test_documented_round_half_up(seconds, expected):
    assert duration_ms(seconds) == expected


@pytest.mark.parametrize("total,weights,expected", [
    (10, [1, 1, 1], [4, 3, 3]), (7, [4, 2], [5, 2]),
    (3, [1, 1], [2, 1]), (2, [1, 1], [1, 1]),
    (9300, [4, 2], [6200, 3100]),
])
def test_allocation_conservation_and_stable_ties(total, weights, expected):
    assert allocate_shots(total, weights) == expected
    assert sum(expected) == total


@pytest.mark.parametrize("seconds", [0, -1, float("nan"), float("inf"), -float("inf"), 0.00049])
def test_invalid_or_submillisecond_durations(seconds):
    with pytest.raises(ValueError):
        duration_ms(seconds)
    values = single()
    narration = values["package"].narration_assets[0].model_copy(update={"duration_seconds": seconds})
    values["package"] = values["package"].model_copy(update={"narration_assets": (narration,)})
    with pytest.raises(ValueError):
        build_timeline_plan(**values)


@pytest.mark.parametrize("duration,weights", [(0.001, (1, 1)), (0.002, (1000, 1)), (0.0001, (1,))])
def test_tiny_intervals_fail_without_silent_repair(duration, weights):
    with pytest.raises(ValueError, match="millisecond"):
        build_timeline_plan(**single(duration=duration, weights=weights))


@pytest.mark.parametrize("field,change", [
    ("narration_assets", lambda items: items.pop()),
    ("narration_assets", lambda items: items.reverse()),
    ("narration_assets", lambda items: items.append(dict(items[0], segment_id="unused", asset=dict(items[0]["asset"], asset_id="extra")))),
    ("narration_assets", lambda items: items[0].update(segment_id="unknown")),
    ("visual_assets", lambda items: items.pop()),
    ("visual_assets", lambda items: items.reverse()),
    ("visual_assets", lambda items: items.append(dict(items[0], shot_id="extra", asset=dict(items[0]["asset"], asset_id="extra")))),
    ("visual_assets", lambda items: items[0].update(source_segment_id="SEG-Z")),
    ("visual_assets", lambda items: items[0].update(generation_method="TEXT_TO_VIDEO")),
    ("visual_assets", lambda items: items[0]["asset"].update(media_type="VIDEO")),
])
def test_missing_extra_order_ownership_and_method_failures(field, change):
    values = authorities()
    data = values["package"].model_dump(mode="json")
    change(data[field])
    values["package"] = MediaPackage.model_validate(data)
    with pytest.raises(ValueError, match="Invalid assembly authorities"):
        build_timeline_plan(**values)


@pytest.mark.parametrize("field,kind", [("storyboard_input_ref", "storyboard"), ("script_input_ref", "script")])
def test_mismatched_exact_provenance(field, kind):
    values = authorities()
    values[field] = reference(kind, 2)
    with pytest.raises(ValueError, match="PROVENANCE_MISMATCH"):
        build_timeline_plan(**values)


@pytest.mark.parametrize("kind,project", [("script", "project"), ("media", "foreign")])
def test_media_reference_type_and_lineage(kind, project):
    values = authorities()
    values["media_input_ref"] = ArtifactReference(project_id=project, artifact_type=kind, version=7)
    with pytest.raises(ValueError, match="Media reference"):
        build_timeline_plan(**values)


def test_wrong_storyboard_segment_and_foreign_script_lineage():
    for source in ("board", "script"):
        values = authorities()
        if source == "board":
            data = values["storyboard"].model_dump(mode="json")
            data["sections"][0]["shots"][0]["source_segment_id"] = "unknown"
            values["storyboard"] = StoryboardPackage.model_validate(data)
        else:
            data = values["script"].model_dump(mode="json")
            data["story_input_ref"]["project_id"] = "foreign"
            values["script"] = ScriptPackage.model_validate(data)
        with pytest.raises(ValueError):
            build_timeline_plan(**values)


def test_roundtrip_and_deep_immutable_contracts():
    values = authorities()
    plan = build_timeline_plan(**values)
    assert TimelinePlan.model_validate_json(plan.model_dump_json()) == plan
    for contract in (plan, plan.segments[0], plan.shots[0], plan.media_input_ref,
                     plan.segments[0].narration_asset, cues(values, plan)[0]):
        assert type(contract).model_validate_json(contract.model_dump_json()) == contract
        field = next(iter(type(contract).model_fields))
        with pytest.raises(ValidationError):
            setattr(contract, field, getattr(contract, field))
    assert set(SegmentTiming.model_fields) == {"start_ms", "end_ms", "segment_id", "narration_asset"}
    assert set(ShotTiming.model_fields) == {"start_ms", "end_ms", "shot_id", "source_segment_id", "visual_asset"}


@pytest.mark.parametrize("change", [
    lambda data: data.update(total_duration_ms=501),
    lambda data: data["segments"][0].update(start_ms=1),
    lambda data: data["shots"][0].update(end_ms=0),
    lambda data: data["shots"][0].update(start_ms=1),
    lambda data: data["shots"][0].update(source_segment_id="unknown"),
    lambda data: data["shots"].reverse(),
    lambda data: data["segments"][0].update(start_ms=0.0),
])
def test_plan_contract_rejects_gaps_drift_unknown_and_noninteger_timing(change):
    data = build_timeline_plan(**authorities()).model_dump(mode="json")
    change(data)
    with pytest.raises(ValidationError):
        TimelinePlan.model_validate(data)


def test_exact_script_required_for_subtitles():
    values = authorities()
    plan = build_timeline_plan(**values)
    with pytest.raises(ValueError, match="exact Script"):
        build_subtitle_cues(plan=plan, script_input_ref=reference("script", 2), script=values["script"])
    data = values["script"].model_dump(mode="json")
    data["sections"].reverse()
    with pytest.raises(ValueError, match="presentation order"):
        build_subtitle_cues(plan=plan, script_input_ref=values["script_input_ref"], script=ScriptPackage.model_validate(data))


def test_empty_script_narration_fails_at_authority_boundary():
    values = authorities()
    section = values["script"].sections[0]
    segment = section.segments[0].model_copy(update={"narration": " "})
    values["script"] = values["script"].model_copy(update={"sections": (
        section.model_copy(update={"segments": (segment, *section.segments[1:])}),)})
    with pytest.raises(ValueError):
        build_timeline_plan(**values)


@pytest.mark.parametrize("weight", [0, -1, float("nan"), float("inf")])
def test_invalid_relative_weights_fail(weight):
    with pytest.raises(ValueError, match="positive and finite"):
        allocate_shots(10, [weight, 1])


def test_fractional_canonical_durations_conserve_cumulative_offsets():
    values = authorities()
    narration = values["package"].narration_assets
    values["package"] = values["package"].model_copy(update={"narration_assets": (
        narration[0].model_copy(update={"duration_seconds": 1.2345}),
        narration[1].model_copy(update={"duration_seconds": 2.3455}))})
    plan = build_timeline_plan(**values)
    assert [(segment.start_ms, segment.end_ms) for segment in plan.segments] == [(0, 1235), (1235, 3581)]
    assert plan.shots[-1].end_ms == plan.total_duration_ms == 3581


def test_srt_newlines_unicode_hour_timestamps_and_wording():
    text = "α\r\n\r\nExact wording.\rLast line"
    cue = SubtitleCue(index=1, start_ms=3661002, end_ms=3662003, text=text)
    assert cue.text == text
    assert serialize_srt((cue,)) == "1\n01:01:01,002 --> 01:01:02,003\nα\n \nExact wording.\nLast line\n\n".encode("utf-8")


@pytest.mark.parametrize("update", [dict(index=2), dict(start_ms=-1), dict(end_ms=0), dict(text=" "), dict(text="bad\x00text")])
def test_invalid_cues_and_indexes(update):
    cue = SubtitleCue(index=1, start_ms=0, end_ms=1000, text="Exact")
    with pytest.raises(ValueError):
        serialize_srt((cue.model_copy(update=update),))


def test_empty_and_overlapping_srt_rejected():
    with pytest.raises(ValueError):
        serialize_srt(())
    with pytest.raises(ValueError, match="nonoverlapping"):
        serialize_srt((SubtitleCue(index=1, start_ms=0, end_ms=10, text="one"),
                       SubtitleCue(index=2, start_ms=9, end_ms=20, text="two")))


def test_pure_planning_has_no_discovery_storage_provider_or_rendering(monkeypatch):
    from history_studio.storage import ArtifactStore, MediaStore
    import history_studio.assembly.planning as module

    values = authorities()
    def forbidden(*args, **kwargs):
        raise AssertionError("Planning must not access external runtime")
    monkeypatch.setattr(builtins, "open", forbidden)
    monkeypatch.setattr(ArtifactStore, "load", forbidden)
    monkeypatch.setattr(ArtifactStore, "save", forbidden)
    monkeypatch.setattr(MediaStore, "read_bytes", forbidden)
    plan = build_timeline_plan(**values)
    assert serialize_srt(cues(values, plan))
    assert all(name not in vars(module) for name in ("subprocess", "ArtifactStore", "MediaStore", "OpenAI"))
    assert "latest" not in inspect.getsource(module)
