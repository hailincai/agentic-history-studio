"""Deterministic planning: no artifact discovery, files, providers or rendering.

Callers authenticate payloads loaded by the explicit exact references. Reference
equality checks lineage, not content authentication or approval. Binary hashes
remain a pre-render concern.
"""
from decimal import Decimal
from fractions import Fraction
import math

from history_studio.media.validation import validate_media_package
from history_studio.models import ArtifactReference, MediaPackage, ScriptPackage, StoryboardPackage
from history_studio.models.assembly import SegmentTiming, ShotTiming, SubtitleCue, TimelinePlan


def duration_ms(seconds: float) -> int:
    """Round decimal string seconds to nearest ms, ties up; never clamp to 1."""
    if not math.isfinite(seconds) or seconds <= 0:
        raise ValueError("Narration duration must be positive and finite")
    # Fractions avoid Decimal context precision limits for large finite values.
    value = Fraction(Decimal(str(seconds))) * 1000
    result = (2 * value.numerator + value.denominator) // (2 * value.denominator)
    if result <= 0:
        raise ValueError("Narration duration cannot form a positive millisecond interval")
    return result


def allocate_shots(total: int, weights: list[float]) -> list[int]:
    """Largest remainders, ties in shot order; reject zero-length allocations.

    Allocate the full measured interval proportionally, without adding a minimum
    baseline that would distort the approved relative weights.
    """
    if total < len(weights):
        raise ValueError("Segment has fewer milliseconds than shots")
    if not weights or any(not math.isfinite(weight) or weight <= 0 for weight in weights):
        raise ValueError("Shot estimates must be positive and finite")
    exact = [Fraction(Decimal(str(weight))) for weight in weights]
    weight_sum = sum(exact)
    quotas = [total * weight / weight_sum for weight in exact]
    allocated = [quota.numerator // quota.denominator for quota in quotas]
    order = sorted(range(len(weights)), key=lambda i: (-(quotas[i] - allocated[i]), i))
    for i in order[:total - sum(allocated)]:
        allocated[i] += 1
    if any(value <= 0 for value in allocated):
        raise ValueError("Weighted shot allocation cannot form positive millisecond intervals")
    return allocated


def build_timeline_plan(*, media_input_ref: ArtifactReference, package: MediaPackage,
                        storyboard_input_ref: ArtifactReference, storyboard: StoryboardPackage,
                        script_input_ref: ArtifactReference, script: ScriptPackage) -> TimelinePlan:
    """Plan exact supplied authorities; represented segments follow Script order."""
    media_ref = ArtifactReference.model_validate(media_input_ref.model_dump())
    storyboard_input_ref = ArtifactReference.model_validate(storyboard_input_ref.model_dump())
    script_input_ref = ArtifactReference.model_validate(script_input_ref.model_dump())
    if media_ref.artifact_type != "media" or media_ref.project_id != storyboard_input_ref.project_id:
        raise ValueError("Media reference must identify media in the exact source project")
    report = validate_media_package(package=package, storyboard_input_ref=storyboard_input_ref,
                                    storyboard=storyboard, script_input_ref=script_input_ref, script=script)
    if not report.is_valid:
        raise ValueError("Invalid assembly authorities: " + "; ".join(issue.code.value for issue in report.issues))
    # Reparse mutable upstream contracts and unsafe model_copy inputs before use.
    package = MediaPackage.model_validate(package.model_dump())
    storyboard = StoryboardPackage.model_validate(storyboard.model_dump())
    shots = [shot for section in storyboard.sections for shot in section.shots]
    visuals = {visual.shot_id: visual.asset for visual in package.visual_assets}
    segments, timings = [], []
    offset = 0
    for narration in package.narration_assets:
        length = duration_ms(narration.duration_seconds)
        selected = [shot for shot in shots if shot.source_segment_id == narration.segment_id]
        lengths = allocate_shots(length, [shot.estimated_duration_seconds for shot in selected])
        segments.append(SegmentTiming(segment_id=narration.segment_id, start_ms=offset,
                                      end_ms=offset + length, narration_asset=narration.asset))
        for shot, shot_length in zip(selected, lengths):
            timings.append(ShotTiming(shot_id=shot.shot_id, source_segment_id=shot.source_segment_id,
                                      start_ms=offset, end_ms=offset + shot_length,
                                      visual_asset=visuals[shot.shot_id]))
            offset += shot_length
    return TimelinePlan(media_input_ref=media_ref, storyboard_input_ref=storyboard_input_ref,
                        script_input_ref=script_input_ref, segments=segments, shots=timings,
                        total_duration_ms=offset)


def build_subtitle_cues(*, plan: TimelinePlan, script_input_ref: ArtifactReference,
                        script: ScriptPackage) -> tuple[SubtitleCue, ...]:
    """One cue per represented segment, preserving exact Script narration text."""
    plan = TimelinePlan.model_validate(plan.model_dump())
    script = ScriptPackage.model_validate(script.model_dump())
    if (script_input_ref != plan.script_input_ref
            or script.story_input_ref.project_id != script_input_ref.project_id):
        raise ValueError("Subtitles require the exact Script reference in the same project")
    by_id = {segment.segment_id: segment for section in script.sections for segment in section.segments}
    represented = [segment.segment_id for segment in plan.segments]
    if [identifier for identifier in by_id if identifier in represented] != represented:
        raise ValueError("Subtitle segments must exist in exact Script presentation order")
    return tuple(SubtitleCue(index=i, start_ms=timing.start_ms, end_ms=timing.end_ms,
                             text=by_id[timing.segment_id].narration)
                 for i, timing in enumerate(plan.segments, 1))


def serialize_srt(cues: tuple[SubtitleCue, ...]) -> bytes:
    """UTF-8 without BOM, LF endings, sequential decimal indexes starting at 1.

    CRLF/CR become LF; empty embedded lines become a space-only line so they
    cannot terminate an SRT block. All wording and other whitespace are retained.
    """
    def timestamp(value: int) -> str:
        seconds, milliseconds = divmod(value, 1000)
        minutes, seconds = divmod(seconds, 60)
        hours, minutes = divmod(minutes, 60)
        return f"{hours:02d}:{minutes:02d}:{seconds:02d},{milliseconds:03d}"

    blocks = []
    end = 0
    for index, candidate in enumerate(cues, 1):
        cue = SubtitleCue.model_validate(candidate.model_dump())
        if cue.index != index or cue.start_ms < end:
            raise ValueError("SRT cues require sequential indexes and ordered nonoverlapping times")
        text = cue.text.replace("\r\n", "\n").replace("\r", "\n")
        text = "\n".join(line if line else " " for line in text.split("\n"))
        blocks.append(f"{index}\n{timestamp(cue.start_ms)} --> {timestamp(cue.end_ms)}\n{text}\n\n")
        end = cue.end_ms
    if not blocks:
        raise ValueError("SRT requires at least one cue")
    return "".join(blocks).encode("utf-8")
