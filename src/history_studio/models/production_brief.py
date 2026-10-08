"""Runtime production guidance, separate from historical artifact authority."""
from decimal import Decimal
from pathlib import Path

from pydantic import ConfigDict, Field, field_validator

from .base import Contract, PositiveSeconds, Text
from .project import ProjectConfig
from .storyboard import GenerationMethod


class ProductionBrief(Contract):
    model_config = ConfigDict(frozen=True)

    topic: Text
    language: Text
    target_duration_seconds: PositiveSeconds
    allowed_generation_methods: tuple[GenerationMethod, ...] = Field(min_length=1)

    @field_validator("allowed_generation_methods")
    @classmethod
    def unique_methods(cls, value):
        return ProjectConfig.unique_methods(value)

    @classmethod
    def from_project(cls, project: ProjectConfig) -> "ProductionBrief":
        project = ProjectConfig.model_validate(project.model_dump())
        return cls(topic=project.topic, language=project.language,
                   target_duration_seconds=float(Decimal(str(project.target_duration_minutes)) * 60),
                   allowed_generation_methods=project.allowed_generation_methods)


def load_production_brief(project_dir: Path, *, project_id: str) -> ProductionBrief:
    """Read the exact project's persisted config; no fallback or artifact discovery."""
    project_dir = Path(project_dir)
    project = ProjectConfig.model_validate_json((project_dir / "project.json").read_text(encoding="utf-8"))
    if project.project_id != project_id or project_dir.name != project_id:
        raise ValueError("Production brief project must match the exact workflow project")
    return ProductionBrief.from_project(project)
