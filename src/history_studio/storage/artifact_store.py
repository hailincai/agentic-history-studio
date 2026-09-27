import os
import re
import tempfile
from pathlib import Path
from typing import TypeVar

from pydantic import BaseModel, TypeAdapter

from history_studio.models.base import Identifier

ModelT = TypeVar("ModelT", bound=BaseModel)


class ArtifactNotFoundError(FileNotFoundError):
    """The requested persisted version does not exist."""


def safe_component(value: str) -> str:
    TypeAdapter(Identifier).validate_python(value)
    reserved = {"CON", "PRN", "AUX", "NUL", *(f"COM{i}" for i in range(1, 10)),
                *(f"LPT{i}" for i in range(1, 10))}
    if value.upper() in reserved:
        raise ValueError("Reserved filesystem name")
    return value


def write_json(path: Path, model: BaseModel, *, replace: bool = False) -> None:
    """Publish complete UTF-8 JSON atomically on a local filesystem.

    Hard-link publication fails if the destination exists, unlike rename on POSIX.
    Unsupported filesystems fail safely. A crash can leave an unreferenced temp file.
    """
    path.parent.mkdir(parents=True, exist_ok=True)
    data = model.model_dump_json(indent=2).encode("utf-8")
    fd, temporary = tempfile.mkstemp(prefix=".pending-", suffix=".tmp", dir=path.parent)
    temporary_path = Path(temporary)
    try:
        with os.fdopen(fd, "wb") as stream:
            stream.write(data)
            stream.flush()
            os.fsync(stream.fileno())
        if replace:
            os.replace(temporary_path, path)
        else:
            os.link(temporary_path, path)
    finally:
        temporary_path.unlink(missing_ok=True)


class ArtifactStore:
    """Immutable versions under a project directory; readers revalidate contracts."""

    def __init__(self, project_dir: Path) -> None:
        self.project_dir = Path(project_dir).resolve()

    def _directory(self, artifact_type: str) -> Path:
        directory = self.project_dir / safe_component(artifact_type)
        if directory.resolve().parent != self.project_dir:
            raise ValueError("Artifact directory escapes project directory")
        return directory

    def list_versions(self, artifact_type: str) -> list[int]:
        directory = self._directory(artifact_type)
        pattern = re.compile(rf"{re.escape(artifact_type)}_v([1-9][0-9]*)\.json")
        return sorted(int(match.group(1)) for path in directory.glob("*.json")
                      if (match := pattern.fullmatch(path.name)) and path.is_file())

    def save(self, artifact_type: str, artifact: BaseModel) -> int:
        directory = self._directory(artifact_type)
        while True:
            versions = self.list_versions(artifact_type)
            version = versions[-1] + 1 if versions else 1
            try:
                write_json(directory / f"{artifact_type}_v{version}.json", artifact)
            except FileExistsError:
                continue  # Another local writer published this version first.
            return version

    def load(self, artifact_type: str, version: int, model_type: type[ModelT]) -> ModelT:
        if isinstance(version, bool) or not isinstance(version, int) or version < 1:
            raise ValueError("Version must be a positive integer")
        path = self._directory(artifact_type) / f"{artifact_type}_v{version}.json"
        if path.is_symlink():
            raise ValueError("Artifact files must not be symbolic links")
        try:
            return model_type.model_validate_json(path.read_text(encoding="utf-8"))
        except FileNotFoundError as exc:
            raise ArtifactNotFoundError(f"Missing {artifact_type} version {version}") from exc

    def load_latest(self, artifact_type: str, model_type: type[ModelT]) -> ModelT:
        versions = self.list_versions(artifact_type)
        if not versions:
            raise ArtifactNotFoundError(f"No {artifact_type} artifacts exist")
        return self.load(artifact_type, versions[-1], model_type)
