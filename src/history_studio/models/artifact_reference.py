"""Runtime-owned identity of one immutable ArtifactStore snapshot."""
from pydantic import ConfigDict, Field

from .base import Contract, Identifier


class ArtifactReference(Contract):
    """Store version identity, distinct from a model's schema_version.

    Runtime supplies the project, stored artifact type, and selected persisted version.
    This reference does not infer identity from content or prove filesystem existence.
    """

    model_config = ConfigDict(frozen=True)

    project_id: Identifier
    artifact_type: Identifier
    version: int = Field(strict=True, gt=0)
