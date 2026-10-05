"""Pure structural grounding checks; no narration semantics or artifact discovery."""
from enum import StrEnum

from pydantic import ValidationError, computed_field

from history_studio.models.artifact_reference import ArtifactReference
from history_studio.models.base import Contract, Text
from history_studio.models.script_context import ScriptContext
from history_studio.models.script_package import (
    ScriptGrounding, ScriptPackage, ScriptSection, ScriptSegment, ScriptSegmentKind,
)


class ScriptGroundingIssueCode(StrEnum):
    INVALID_PACKAGE_STRUCTURE = "INVALID_PACKAGE_STRUCTURE"
    STORY_PROVENANCE_MISMATCH = "STORY_PROVENANCE_MISMATCH"
    UNKNOWN_SECTION = "UNKNOWN_SECTION"
    SECTION_ORDER_VIOLATION = "SECTION_ORDER_VIOLATION"
    DUPLICATE_SECTION = "DUPLICATE_SECTION"
    DUPLICATE_SEGMENT = "DUPLICATE_SEGMENT"
    MISSING_HISTORICAL_GROUNDING = "MISSING_HISTORICAL_GROUNDING"
    UNKNOWN_STORY_BEAT = "UNKNOWN_STORY_BEAT"
    BEAT_SECTION_MISMATCH = "BEAT_SECTION_MISMATCH"
    NON_HISTORICAL_BEAT = "NON_HISTORICAL_BEAT"
    EMPTY_FACT_SELECTION = "EMPTY_FACT_SELECTION"
    UNKNOWN_FACT = "UNKNOWN_FACT"
    FACT_BEAT_MISMATCH = "FACT_BEAT_MISMATCH"
    DUPLICATE_FACT = "DUPLICATE_FACT"
    STRUCTURAL_GROUNDING_PRESENT = "STRUCTURAL_GROUNDING_PRESENT"
    BEAT_ORDER_VIOLATION = "BEAT_ORDER_VIOLATION"


class ScriptGroundingValidationIssue(Contract):
    code: ScriptGroundingIssueCode
    message: Text
    section_id: str | None = None
    segment_id: str | None = None
    story_beat_id: str | None = None
    research_fact_id: str | None = None


class ScriptGroundingValidationReport(Contract):
    issues: tuple[ScriptGroundingValidationIssue, ...] = ()

    @computed_field
    @property
    def is_valid(self) -> bool:
        return not self.issues


def validate_script_grounding(context: ScriptContext, package: ScriptPackage) -> ScriptGroundingValidationReport:
    """Validate exact lineage and structural authorization without reconstructing output.

    Invalid context raises: it cannot be an authority. Package shape failures become
    report issues; unsafe model copies retaining traversable fields also receive specific
    grounding diagnostics. Issues follow shape errors, provenance, then presentation order.
    This report proves neither approval nor prose truth/qualification/date adequacy.
    """
    if not isinstance(context, ScriptContext) or not isinstance(package, ScriptPackage):
        raise TypeError("Validation requires ScriptContext and ScriptPackage")
    reference = ArtifactReference(project_id=context.story_input_ref.project_id,
        artifact_type=context.story_input_ref.artifact_type, version=context.story_input_ref.version)
    source = ScriptContext.model_validate(context.model_dump(mode="json") | {
        "story_input_ref": reference.model_dump(mode="json")})
    issues = []

    def issue(code, message, **location):
        issues.append(ScriptGroundingValidationIssue(code=code, message=message, **location))

    try:
        ScriptPackage.model_validate(package.model_dump(mode="python", warnings=False))
        ArtifactReference(project_id=package.story_input_ref.project_id,
            artifact_type=package.story_input_ref.artifact_type, version=package.story_input_ref.version)
    except (ValidationError, AttributeError, TypeError) as exc:
        issue(ScriptGroundingIssueCode.INVALID_PACKAGE_STRUCTURE, "Package violates durable contract structure")
        # Only typed, traversable values can receive additional identity diagnostics.
        if not isinstance(exc, ValidationError):
            return ScriptGroundingValidationReport(issues=issues)
        # Unsafe copies may replace nested models with arbitrary values. Report
        # malformed shape without attempting identity traversal through those values.
        if (not isinstance(package.story_input_ref, ArtifactReference)
                or not isinstance(package.sections, (tuple, list))
                or any(not isinstance(section, ScriptSection)
                       or not isinstance(section.segments, (tuple, list))
                       or any(not isinstance(segment, ScriptSegment)
                              or (segment.grounding is not None
                                  and not isinstance(segment.grounding, ScriptGrounding))
                              or (segment.grounding is not None
                                  and not isinstance(segment.grounding.research_fact_ids, (tuple, list)))
                              for segment in section.segments)
                       for section in package.sections)):
            return ScriptGroundingValidationReport(issues=issues)
    if package.story_input_ref != source.story_input_ref:
        issue(ScriptGroundingIssueCode.STORY_PROVENANCE_MISMATCH, "Exact Story input reference must match context")
    sections = {section.section_id: index for index, section in enumerate(source.sections)}
    beats = {beat.beat_id: (section.section_id, index, beat)
             for section in source.sections for index, beat in enumerate(section.beats)}
    facts = {fact.research_fact_id for section in source.sections for beat in section.beats for fact in beat.fact_refs}
    seen_sections, seen_segments = set(), set()
    previous_section = -1
    for section in package.sections:
        location = dict(section_id=section.section_id)
        if section.section_id in seen_sections:
            issue(ScriptGroundingIssueCode.DUPLICATE_SECTION, "Section ID is repeated", **location)
        seen_sections.add(section.section_id)
        position = sections.get(section.section_id)
        if position is None:
            issue(ScriptGroundingIssueCode.UNKNOWN_SECTION, "Section is absent from approved Story", **location)
        else:
            if position <= previous_section:
                issue(ScriptGroundingIssueCode.SECTION_ORDER_VIOLATION, "Sections violate approved presentation order", **location)
            previous_section = position
        previous_beat = -1
        for segment in section.segments:
            loc = location | dict(segment_id=segment.segment_id)
            if segment.segment_id in seen_segments:
                issue(ScriptGroundingIssueCode.DUPLICATE_SEGMENT, "Segment ID is repeated globally", **loc)
            seen_segments.add(segment.segment_id)
            if segment.kind == ScriptSegmentKind.STRUCTURAL:
                if segment.grounding is not None:
                    issue(ScriptGroundingIssueCode.STRUCTURAL_GROUNDING_PRESENT, "Structural segment carries grounding", **loc)
                continue
            if segment.kind != ScriptSegmentKind.HISTORICAL:
                continue  # Invalid enum is reported by durable shape validation.
            grounding = segment.grounding
            if grounding is None:
                issue(ScriptGroundingIssueCode.MISSING_HISTORICAL_GROUNDING, "Historical segment requires grounding", **loc)
                continue
            loc = loc | dict(story_beat_id=grounding.story_beat_id)
            selected = beats.get(grounding.story_beat_id)
            authorized = set()
            if selected is None:
                issue(ScriptGroundingIssueCode.UNKNOWN_STORY_BEAT, "Beat is absent from approved Story", **loc)
            else:
                section_id, beat_position, beat = selected
                authorized = {fact.research_fact_id for fact in beat.fact_refs}
                if section_id != section.section_id:
                    issue(ScriptGroundingIssueCode.BEAT_SECTION_MISMATCH, "Beat belongs to a different Story section", **loc)
                if beat.kind != "historical":
                    issue(ScriptGroundingIssueCode.NON_HISTORICAL_BEAT, "Historical segment targets a structural beat", **loc)
                elif section_id == section.section_id:
                    if beat_position < previous_beat:
                        issue(ScriptGroundingIssueCode.BEAT_ORDER_VIOLATION, "Historical beat positions must not decrease", **loc)
                    previous_beat = beat_position
            if not grounding.research_fact_ids:
                issue(ScriptGroundingIssueCode.EMPTY_FACT_SELECTION, "Historical grounding requires selected facts", **loc)
            seen_facts = set()
            for fact_id in grounding.research_fact_ids:
                fact_loc = loc | dict(research_fact_id=fact_id)
                if fact_id in seen_facts:
                    issue(ScriptGroundingIssueCode.DUPLICATE_FACT, "Selected fact ID is repeated", **fact_loc)
                seen_facts.add(fact_id)
                if fact_id not in facts:
                    issue(ScriptGroundingIssueCode.UNKNOWN_FACT, "Fact is absent from approved Story", **fact_loc)
                elif selected is not None and fact_id not in authorized:
                    issue(ScriptGroundingIssueCode.FACT_BEAT_MISMATCH, "Fact is not authorized by the selected beat", **fact_loc)
    return ScriptGroundingValidationReport(issues=issues)
