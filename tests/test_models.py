from datetime import datetime, timezone

import pytest
from pydantic import ValidationError

from history_studio.models import (
    ProjectConfig, SourceReference, ResearchFact, VerifiedFact,
    Script, ScriptScene, StoryBeat, StoryPlan, Shot, Storyboard,
)
from history_studio.workflow import ApprovalRecord


def source() -> SourceReference:
    return SourceReference(source_id="s1", title="唐诗", url="https://example.org/book",
                           source_type="BOOK", accessed_at=datetime.now(timezone.utc))


@pytest.mark.parametrize("changes", [
    {"project_id": "../escape"}, {"topic": " "}, {"budget_usd": -1},
    {"target_duration_minutes": 0}, {"output_width": 0}, {"output_height": True},
    {"budget_usd": float("inf")}, {"unknown": 1},
])
def test_invalid_project(changes: dict) -> None:
    with pytest.raises(ValidationError):
        ProjectConfig(**({"project_id": "li_bai", "topic": "李白"} | changes))


@pytest.mark.parametrize("confidence", [-0.01, 1.01, float("nan"), float("inf")])
def test_confidence_bounds(confidence: float) -> None:
    with pytest.raises(ValidationError):
        ResearchFact(id="f1", claim="claim", time_period="701", category="life",
                     sources=[source()], researcher_confidence=confidence)
    with pytest.raises(ValidationError):
        VerifiedFact(fact_id="f1", claim="claim", status="VERIFIED", confidence=confidence,
                     reasoning="reason")


@pytest.mark.parametrize("confidence", [0, 1])
def test_confidence_endpoints(confidence: float) -> None:
    fact = ResearchFact(id="f1", claim="李白", time_period="唐代", category="life",
                        sources=[source()], researcher_confidence=confidence)
    assert ResearchFact.model_validate_json(fact.model_dump_json()) == fact
    assert VerifiedFact(fact_id="f1", claim="claim", status="DISPUTED", confidence=confidence,
                        reasoning="reason").confidence == confidence


def test_source_requires_url_and_aware_timestamp() -> None:
    for changes in ({"url": "invalid"}, {"accessed_at": "2026-01-01T00:00:00"}):
        with pytest.raises(ValidationError):
            SourceReference.model_validate(source().model_dump() | changes)


def scene(scene_id: str = "s1", sequence: int = 1) -> ScriptScene:
    return ScriptScene(scene_id=scene_id, sequence=sequence, narration="李白", fact_ids=["f1"],
                       duration_seconds=10)


def test_script_structure_and_total() -> None:
    script = Script(title="李白", scenes=[scene(), scene("s2", 2)], target_duration_seconds=30)
    assert script.total_estimated_duration_seconds == 20
    for scenes in ([scene(), scene("s1", 2)], [scene(), scene("s2", 1)],
                   [scene("s2", 2), scene()], []):
        with pytest.raises(ValidationError):
            Script(title="title", scenes=scenes, target_duration_seconds=30)


def test_story_order() -> None:
    beat = StoryBeat(sequence=1, title="Early life", time_period="701–725", purpose="Introduce",
                     fact_ids=["f1"], target_seconds=12)
    fields = dict(title="title", thesis="thesis", central_question="why?", hook="hook",
                  ending="ending", target_duration_seconds=20)
    assert StoryPlan(beats=[beat], **fields).total_estimated_duration_seconds == 12
    with pytest.raises(ValidationError):
        StoryPlan(beats=[beat, beat], **fields)
    with pytest.raises(ValidationError):
        StoryPlan(beats=[beat.model_copy(update={"sequence": 2}), beat], **fields)


def test_storyboard_structure() -> None:
    shot = Shot(shot_id="shot1", scene_id="scene1", sequence=1, start_seconds=0,
                duration_seconds=5, visual_description="river", location="China", period="Tang",
                generation_method="STATIC_IMAGE", camera_motion="none", prompt="river")
    next_shot = shot.model_copy(update={"shot_id": "shot2", "sequence": 2, "start_seconds": 5})
    assert len(Storyboard(shots=[shot, next_shot], estimated_media_cost_usd=0).shots) == 2
    for changes in ({"start_seconds": 4}, {"shot_id": "shot1"}, {"sequence": 1}):
        with pytest.raises(ValidationError):
            Storyboard(shots=[shot, next_shot.model_copy(update=changes)], estimated_media_cost_usd=0)
    with pytest.raises(ValidationError):
        Shot.model_validate(shot.model_dump() | {"duration_seconds": -1})


def test_approval_requires_explicit_human() -> None:
    data = dict(project_id="li_bai", stage="facts", artifact_type="facts", artifact_version=1,
                decision="APPROVED", decided_at=datetime.now(timezone.utc), decided_by="reviewer")
    with pytest.raises(ValidationError):
        ApprovalRecord(**data)
    with pytest.raises(ValidationError):
        ApprovalRecord(**data, decision_source="agent")
    with pytest.raises(ValidationError):
        ApprovalRecord(**(data | {"artifact_type": "story"}), decision_source="human")
