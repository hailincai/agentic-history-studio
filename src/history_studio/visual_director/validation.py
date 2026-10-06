"""Pure Storyboard structural integrity checks; never historical prose validation."""
from enum import StrEnum

from pydantic import ValidationError, computed_field

from history_studio.models.artifact_reference import ArtifactReference
from history_studio.models.base import Contract, Text
from history_studio.models.storyboard_package import (
    StoryboardPackage, StoryboardSection, StoryboardShot, StoryboardShotKind,
)
from history_studio.models.visual_director_context import VisualDirectorContext


class StoryboardIntegrityIssueCode(StrEnum):
    INVALID_PACKAGE_STRUCTURE = "INVALID_PACKAGE_STRUCTURE"
    SCRIPT_PROVENANCE_MISMATCH = "SCRIPT_PROVENANCE_MISMATCH"
    TITLE_MISMATCH = "TITLE_MISMATCH"
    SECTION_MISSING = "SECTION_MISSING"
    SECTION_UNKNOWN = "SECTION_UNKNOWN"
    SECTION_ORDER_MISMATCH = "SECTION_ORDER_MISMATCH"
    SECTION_ID_DUPLICATE = "SECTION_ID_DUPLICATE"
    SECTION_IDENTITY_MISMATCH = "SECTION_IDENTITY_MISMATCH"
    SEGMENT_UNCOVERED = "SEGMENT_UNCOVERED"
    SOURCE_SEGMENT_UNKNOWN = "SOURCE_SEGMENT_UNKNOWN"
    SOURCE_SEGMENT_WRONG_SECTION = "SOURCE_SEGMENT_WRONG_SECTION"
    SHOT_KIND_MISMATCH = "SHOT_KIND_MISMATCH"
    SEGMENT_ORDER_REGRESSION = "SEGMENT_ORDER_REGRESSION"
    SHOT_ID_DUPLICATE = "SHOT_ID_DUPLICATE"


class StoryboardIntegrityIssue(Contract):
    code: StoryboardIntegrityIssueCode
    message: Text
    section_id: str | None = None
    shot_id: str | None = None
    source_segment_id: str | None = None


class StoryboardIntegrityReport(Contract):
    issues: tuple[StoryboardIntegrityIssue, ...] = ()

    @computed_field
    @property
    def is_valid(self) -> bool:
        return not self.issues


def _traversable(package: StoryboardPackage) -> bool:
    """Unsafe copies receive identity diagnostics only when traversal is safe."""
    return (isinstance(getattr(package, "script_input_ref", None), ArtifactReference)
            and isinstance(getattr(package, "title", None), str)
            and isinstance(getattr(package, "sections", None), (tuple, list))
            and all(isinstance(section, StoryboardSection)
                    and isinstance(getattr(section, "section_id", None), str)
                    and isinstance(getattr(section, "title", None), str)
                    and isinstance(getattr(section, "shots", None), (tuple, list))
                    and all(isinstance(shot, StoryboardShot)
                            and isinstance(getattr(shot, "shot_id", None), str)
                            and isinstance(getattr(shot, "source_segment_id", None), str)
                            and isinstance(getattr(shot, "kind", None), str)
                            for shot in section.shots)
                    for section in package.sections))


def validate_storyboard_integrity(context: VisualDirectorContext,
                                  package: StoryboardPackage) -> StoryboardIntegrityReport:
    """Re-authenticate exact lineage, canonical identity and complete visual coverage.

    Runtime supplies an approved authenticated context; invalid authority raises.
    Package-local shape errors are reported, followed by provenance/title, package
    presentation traversal, then missing coverage in context order. Traversable unsafe
    copies receive specific diagnostics as well as a shape issue. No repair or mutation
    occurs. Structural validity proves neither approval nor visual historical accuracy.
    """
    if not isinstance(context, VisualDirectorContext) or not isinstance(package, StoryboardPackage):
        raise TypeError("Validation requires VisualDirectorContext and StoryboardPackage")
    reference = ArtifactReference(project_id=context.script_input_ref.project_id,
        artifact_type=context.script_input_ref.artifact_type, version=context.script_input_ref.version)
    source = VisualDirectorContext.model_validate(context.model_dump(mode="json") | {
        "script_input_ref": reference.model_dump(mode="json")})
    issues = []
    code = StoryboardIntegrityIssueCode

    def issue(kind, message, **location):
        issues.append(StoryboardIntegrityIssue(code=kind, message=message, **location))

    try:
        # Python-shaped data preserves invalid raw types in deliberately unsafe copies.
        validated = StoryboardPackage.model_validate(package.model_dump(mode="python", warnings=False))
        ArtifactReference(project_id=package.script_input_ref.project_id,
            artifact_type=package.script_input_ref.artifact_type, version=package.script_input_ref.version)
    except (ValidationError, AttributeError, TypeError):
        issue(code.INVALID_PACKAGE_STRUCTURE, "Package violates durable contract structure")
        if not _traversable(package):
            return StoryboardIntegrityReport(issues=issues)
    else:
        package = validated  # Detached, normalized fields under existing Text semantics.
    if package.script_input_ref != source.script_input_ref:
        issue(code.SCRIPT_PROVENANCE_MISMATCH, "Exact Script input reference must match context")
    if package.title != source.title:
        issue(code.TITLE_MISMATCH, "Package title must match approved Script title")
    sections = {section.section_id: (index, section) for index, section in enumerate(source.sections)}
    segments = {segment.segment_id: (section.section_id, index, segment)
                for section in source.sections for index, segment in enumerate(section.segments)}
    seen_sections, seen_shots, covered = set(), set(), set()
    previous_section = -1
    for section in package.sections:
        loc = dict(section_id=section.section_id)
        duplicate = section.section_id in seen_sections
        if duplicate:
            issue(code.SECTION_ID_DUPLICATE, "Section ID is repeated", **loc)
        seen_sections.add(section.section_id)
        selected_section = sections.get(section.section_id)
        if selected_section is None:
            issue(code.SECTION_UNKNOWN, "Section is absent from approved Script", **loc)
        else:
            position, canonical = selected_section
            if not duplicate and position < previous_section:
                issue(code.SECTION_ORDER_MISMATCH, "Sections violate approved presentation order", **loc)
            previous_section = max(previous_section, position)
            if section.title != canonical.title:
                issue(code.SECTION_IDENTITY_MISMATCH, "Section title must match approved Script section", **loc)
        previous_segment = -1
        for shot in section.shots:
            shot_loc = loc | dict(shot_id=shot.shot_id, source_segment_id=shot.source_segment_id)
            if shot.shot_id in seen_shots:
                issue(code.SHOT_ID_DUPLICATE, "Shot ID is repeated globally", **shot_loc)
            seen_shots.add(shot.shot_id)
            selected = segments.get(shot.source_segment_id)
            if selected is None:
                issue(code.SOURCE_SEGMENT_UNKNOWN, "Source segment is absent from approved Script", **shot_loc)
                continue
            owner, position, segment = selected
            if owner != section.section_id:
                issue(code.SOURCE_SEGMENT_WRONG_SECTION, "Source segment belongs to another Script section", **shot_loc)
                continue
            covered.add(shot.source_segment_id)
            if shot.kind in tuple(StoryboardShotKind) and shot.kind != segment.kind:
                issue(code.SHOT_KIND_MISMATCH, "Shot kind must match approved Script segment kind", **shot_loc)
            if position < previous_segment:
                issue(code.SEGMENT_ORDER_REGRESSION, "Source segment positions must not decrease", **shot_loc)
            previous_segment = max(previous_segment, position)
    for section in source.sections:
        if section.section_id not in seen_sections:
            issue(code.SECTION_MISSING, "Approved Script section is missing", section_id=section.section_id)
        for segment in section.segments:
            if segment.segment_id not in covered:
                issue(code.SEGMENT_UNCOVERED, "Approved Script segment has no shot in its section",
                      section_id=section.section_id, source_segment_id=segment.segment_id)
    return StoryboardIntegrityReport(issues=issues)
