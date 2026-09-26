from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

import pytest
from pydantic import ValidationError

from history_studio.models import ProjectConfig
from history_studio.storage import ArtifactNotFoundError, ArtifactStore
from history_studio.storage.artifact_store import write_json


def config(topic: str = "李白") -> ProjectConfig:
    return ProjectConfig(project_id="li_bai", topic=topic)


def test_versions_preserve_history_and_unicode(tmp_path: Path) -> None:
    store = ArtifactStore(tmp_path)
    assert store.list_versions("research") == []
    assert store.save("research", config()) == 1
    original = (tmp_path / "research/research_v1.json").read_bytes()
    assert "李白" in original.decode("utf-8")
    assert store.save("research", config("杜甫")) == 2
    assert store.list_versions("research") == [1, 2]
    assert (tmp_path / "research/research_v1.json").read_bytes() == original
    assert store.load("research", 1, ProjectConfig).topic == "李白"
    assert store.load_latest("research", ProjectConfig).topic == "杜甫"


def test_missing_and_unsafe_paths(tmp_path: Path) -> None:
    store = ArtifactStore(tmp_path)
    with pytest.raises(ArtifactNotFoundError):
        store.load_latest("facts", ProjectConfig)
    with pytest.raises(ArtifactNotFoundError):
        store.load("facts", 1, ProjectConfig)
    for artifact_type in ("../outside", "a/b", "CON", "/absolute"):
        with pytest.raises(ValueError):
            store.save(artifact_type, config())
    with pytest.raises(ValueError):
        store.load("facts", 0, ProjectConfig)


def test_atomic_publication_and_failed_write(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    store = ArtifactStore(tmp_path)
    store.save("facts", config())
    original = (tmp_path / "facts/facts_v1.json").read_bytes()
    def fail_link(*args: object) -> None:
        raise OSError("simulated publication failure")
    monkeypatch.setattr("history_studio.storage.artifact_store.os.link", fail_link)
    with pytest.raises(OSError):
        store.save("facts", config("other"))
    assert store.list_versions("facts") == [1]
    assert (tmp_path / "facts/facts_v1.json").read_bytes() == original
    assert not list((tmp_path / "facts").glob(".pending-*"))


def test_concurrent_writers_do_not_overwrite(tmp_path: Path) -> None:
    store = ArtifactStore(tmp_path)
    with ThreadPoolExecutor(max_workers=4) as executor:
        versions = list(executor.map(lambda i: store.save("facts", config(str(i))), range(12)))
    assert sorted(versions) == list(range(1, 13))
    assert {store.load("facts", v, ProjectConfig).topic for v in versions} == {str(i) for i in range(12)}


def test_corrupt_latest_is_not_silently_skipped(tmp_path: Path) -> None:
    store = ArtifactStore(tmp_path)
    store.save("facts", config())
    (tmp_path / "facts/facts_v2.json").write_text("{", encoding="utf-8")
    with pytest.raises(ValidationError):
        store.load_latest("facts", ProjectConfig)


def test_atomic_runtime_replacement_and_exclusive_default(tmp_path: Path) -> None:
    path = tmp_path / "state.json"
    write_json(path, config())
    with pytest.raises(FileExistsError):
        write_json(path, config("other"))
    assert ProjectConfig.model_validate_json(path.read_text(encoding="utf-8")).topic == "李白"
    write_json(path, config("other"), replace=True)
    assert ProjectConfig.model_validate_json(path.read_text(encoding="utf-8")).topic == "other"


def test_failed_flush_never_publishes(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    store = ArtifactStore(tmp_path)
    def fail_fsync(fd: int) -> None:
        raise OSError("simulated flush failure")
    monkeypatch.setattr("history_studio.storage.artifact_store.os.fsync", fail_fsync)
    with pytest.raises(OSError):
        store.save("facts", config())
    assert store.list_versions("facts") == []
    assert not list((tmp_path / "facts").glob(".pending-*"))
