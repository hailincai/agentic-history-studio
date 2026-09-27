"""Immutable local persistence of the existing canonical SourceSpans representation."""
from pathlib import Path

from pydantic import RootModel

from history_studio.storage.artifact_store import safe_component, write_json
from .spans import SourceSpans, make_spans


class SourceStore:
    def __init__(self, directory: Path) -> None:
        self.directory = directory.resolve()

    def _path(self, source_id: str, source_version: str) -> Path:
        path = self.directory / safe_component(source_id) / (safe_component(source_version) + ".json")
        if not path.resolve().is_relative_to(self.directory) or path.is_symlink():
            raise ValueError("Source store path escapes its directory")
        return path

    def put(self, source: SourceSpans) -> None:
        path = self._path(source.source_id, source.source_version)
        if source != make_spans(source.source_id, source.text, source.truncated):
            raise ValueError("Invalid canonical source representation")
        if path.exists():
            if self.get(source.source_id, source.source_version) != source:
                raise ValueError("Canonical source version collision")
            return
        write_json(path, RootModel[SourceSpans](source))

    def get(self, source_id: str, source_version: str) -> SourceSpans:
        source = RootModel[SourceSpans].model_validate_json(
            self._path(source_id, source_version).read_text(encoding="utf-8")).root
        if (source.source_id != source_id or source.source_version != source_version
                or source != make_spans(source_id, source.text, source.truncated)):
            raise ValueError("Stored canonical source identity/content mismatch")
        return source
