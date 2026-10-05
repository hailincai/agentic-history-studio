"""Exact workflow lineage, separate from artifact contents and execution position."""
from typing import Self

from pydantic import model_validator

from history_studio.models.artifact_reference import ArtifactReference
from history_studio.models.base import Contract


class WorkflowArtifactBindings(Contract):
    """Output and approved snapshots have distinct meanings; future gates populate neither automatically."""

    model_config = {"frozen": True}
    research: ArtifactReference | None = None
    verification: ArtifactReference | None = None
    approved_verification: ArtifactReference | None = None
    story: ArtifactReference | None = None
    approved_story: ArtifactReference | None = None
    script: ArtifactReference | None = None
    approved_script: ArtifactReference | None = None
    storyboard: ArtifactReference | None = None
    approved_storyboard: ArtifactReference | None = None

    @model_validator(mode="after")
    def typed_lineage(self) -> Self:
        projects = set()
        for field in type(self).model_fields:
            reference = getattr(self, field)
            if reference is not None:
                expected = field.removeprefix("approved_")
                if reference.artifact_type != expected:
                    raise ValueError(f"Workflow {field} reference must identify a {expected} artifact")
                projects.add(reference.project_id)
        if len(projects) > 1:
            raise ValueError("Workflow artifact references belong to different projects")
        return self

    def validate_project(self, project_id: str) -> None:
        # Reparse at the trusted boundary, including deliberately unvalidated model copies.
        validated = type(self).model_validate(self.model_dump(mode="json"))
        if any(reference is not None and reference.project_id != project_id
               for field in type(self).model_fields for reference in [getattr(validated, field)]):
            raise ValueError("Workflow artifact reference belongs to a different project")

    def with_research(self, reference: ArtifactReference) -> Self:
        """Replacing the root invalidates every dependent snapshot."""
        if reference == self.research:
            return type(self).model_validate(self.model_dump(mode="json"))
        return type(self)(research=reference)

    def with_verification(self, reference: ArtifactReference) -> Self:
        """Replacing a review target invalidates approval and later-stage lineage."""
        if reference == self.verification:
            return type(self).model_validate(self.model_dump(mode="json"))
        return type(self)(research=self.research, verification=reference)

    def with_story(self, reference: ArtifactReference) -> Self:
        """Replacing Story invalidates approval and all Story-derived outputs."""
        if reference == self.story:
            return type(self).model_validate(self.model_dump(mode="json"))
        return type(self)(research=self.research, verification=self.verification,
                          approved_verification=self.approved_verification, story=reference)
