"""Pure cross-authority manifest checks; no files, providers or semantic heuristics."""
from enum import StrEnum

from pydantic import ValidationError, computed_field

from history_studio.models import ArtifactReference, GenerationMethod, MediaPackage, MediaType, ScriptPackage, StoryboardPackage
from history_studio.models.base import Contract, Text


class MediaIntegrityIssueCode(StrEnum):
    INVALID_PACKAGE_STRUCTURE = "INVALID_PACKAGE_STRUCTURE"
    STORYBOARD_PROVENANCE_MISMATCH = "STORYBOARD_PROVENANCE_MISMATCH"
    SCRIPT_PROVENANCE_MISMATCH = "SCRIPT_PROVENANCE_MISMATCH"
    TITLE_MISMATCH = "TITLE_MISMATCH"
    SOURCE_SEGMENT_UNKNOWN = "SOURCE_SEGMENT_UNKNOWN"
    NARRATION_MISSING = "NARRATION_MISSING"
    NARRATION_EXTRA = "NARRATION_EXTRA"
    NARRATION_UNKNOWN = "NARRATION_UNKNOWN"
    NARRATION_ORDER_MISMATCH = "NARRATION_ORDER_MISMATCH"
    VISUAL_MISSING = "VISUAL_MISSING"
    VISUAL_UNKNOWN = "VISUAL_UNKNOWN"
    VISUAL_ORDER_MISMATCH = "VISUAL_ORDER_MISMATCH"
    SOURCE_SEGMENT_MISMATCH = "SOURCE_SEGMENT_MISMATCH"
    GENERATION_METHOD_MISMATCH = "GENERATION_METHOD_MISMATCH"
    FINAL_MEDIA_TYPE_MISMATCH = "FINAL_MEDIA_TYPE_MISMATCH"


class MediaIntegrityIssue(Contract):
    code: MediaIntegrityIssueCode
    message: Text
    segment_id: str | None = None
    shot_id: str | None = None


class MediaIntegrityReport(Contract):
    issues: tuple[MediaIntegrityIssue, ...] = ()

    @computed_field
    @property
    def is_valid(self) -> bool:
        return not self.issues


def validate_media_package(*, storyboard_input_ref: ArtifactReference, storyboard: StoryboardPackage,
                           script_input_ref: ArtifactReference, script: ScriptPackage,
                           package: MediaPackage) -> MediaIntegrityReport:
    """Authenticate explicit exact source references, complete coverage and order.

    The caller loads these authorities by the supplied immutable references.
    References cannot authenticate an arbitrary caller-asserted payload by content;
    workflow loading is the trusted boundary. Invalid authorities raise, malformed
    candidate manifests (including duplicates/invalid AUDIO records) are reported.
    Binary verification and measured audio duration checks belong to publication.
    """
    board_ref = ArtifactReference.model_validate(storyboard_input_ref.model_dump(mode="python", warnings=False))
    script_ref = ArtifactReference.model_validate(script_input_ref.model_dump(mode="python", warnings=False))
    storyboard = StoryboardPackage.model_validate(storyboard.model_dump(mode="python", warnings=False))
    script = ScriptPackage.model_validate(script.model_dump(mode="python", warnings=False))
    issues = []
    code = MediaIntegrityIssueCode

    def issue(kind, message, **location):
        issues.append(MediaIntegrityIssue(code=kind, message=message, **location))

    try:
        package = MediaPackage.model_validate(package.model_dump(mode="python", warnings=False))
    except (ValidationError, AttributeError, TypeError):
        issue(code.INVALID_PACKAGE_STRUCTURE, "Manifest violates durable contracts, including identity uniqueness and media compatibility")
        return MediaIntegrityReport(issues=issues)
    if board_ref.artifact_type != "storyboard" or package.storyboard_input_ref != board_ref:
        issue(code.STORYBOARD_PROVENANCE_MISMATCH, "Manifest must identify the exact supplied Storyboard reference")
    if (script_ref.artifact_type != "script" or storyboard.script_input_ref != script_ref
            or script_ref.project_id != board_ref.project_id
            or script.story_input_ref.project_id != script_ref.project_id):
        issue(code.SCRIPT_PROVENANCE_MISMATCH, "Storyboard must identify the exact supplied Script in the same project")
    if package.title != storyboard.title:
        issue(code.TITLE_MISMATCH, "Manifest title must match authoritative Storyboard title")
    segments = [segment for section in script.sections for segment in section.segments]
    segment_ids = {segment.segment_id for segment in segments}
    shots = [shot for section in storyboard.sections for shot in section.shots]
    required = {shot.source_segment_id for shot in shots}
    for shot in shots:
        if shot.source_segment_id not in segment_ids:
            issue(code.SOURCE_SEGMENT_UNKNOWN, "Storyboard segment is absent from exact Script",
                  segment_id=shot.source_segment_id, shot_id=shot.shot_id)
    expected_narration = [segment.segment_id for segment in segments if segment.segment_id in required]
    actual_narration = [asset.segment_id for asset in package.narration_assets]
    for asset in package.narration_assets:
        if asset.segment_id not in segment_ids:
            issue(code.NARRATION_UNKNOWN, "Narration segment is absent from exact Script", segment_id=asset.segment_id)
        elif asset.segment_id not in required:
            issue(code.NARRATION_EXTRA, "Narration segment is not represented by Storyboard", segment_id=asset.segment_id)
    for segment_id in expected_narration:
        if segment_id not in actual_narration:
            issue(code.NARRATION_MISSING, "Represented segment requires narration", segment_id=segment_id)
    if set(actual_narration) == set(expected_narration) and actual_narration != expected_narration:
        issue(code.NARRATION_ORDER_MISMATCH, "Narration must follow Script presentation order")
    by_shot = {shot.shot_id: shot for shot in shots}
    actual_shots = [asset.shot_id for asset in package.visual_assets]
    for asset in package.visual_assets:
        shot = by_shot.get(asset.shot_id)
        if shot is None:
            issue(code.VISUAL_UNKNOWN, "Visual shot is absent from exact Storyboard", shot_id=asset.shot_id)
            continue
        if asset.source_segment_id != shot.source_segment_id:
            issue(code.SOURCE_SEGMENT_MISMATCH, "Visual source segment must match approved shot", shot_id=asset.shot_id)
        if asset.generation_method != shot.generation_method:
            issue(code.GENERATION_METHOD_MISMATCH, "Visual method must match approved shot", shot_id=asset.shot_id)
        expected_type = MediaType.IMAGE if shot.generation_method == GenerationMethod.STATIC_IMAGE else MediaType.VIDEO
        if asset.asset.media_type != expected_type:
            issue(code.FINAL_MEDIA_TYPE_MISMATCH, "Final media type must match approved generation method", shot_id=asset.shot_id)
    for shot in shots:
        if shot.shot_id not in actual_shots:
            issue(code.VISUAL_MISSING, "Every Storyboard shot requires one final visual", shot_id=shot.shot_id)
    expected_shots = [shot.shot_id for shot in shots]
    if set(actual_shots) == set(expected_shots) and actual_shots != expected_shots:
        issue(code.VISUAL_ORDER_MISMATCH, "Visuals must follow Storyboard presentation order")
    return MediaIntegrityReport(issues=issues)
