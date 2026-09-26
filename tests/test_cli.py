import os
from pathlib import Path
import subprocess
import sys

from history_studio.cli import main
from history_studio.models import ProjectConfig
from history_studio.storage.artifact_store import write_json
from history_studio.workflow import ProjectState as S, RuntimeState


def create(root: Path) -> int:
    return main(["--projects-dir", str(root), "create", "--project-id", "li_bai",
                 "--topic", "李白", "--duration", "5", "--budget", "20"])


def test_create_status_review_resume(tmp_path: Path, capsys) -> None:
    assert create(tmp_path) == 0
    config = ProjectConfig.model_validate_json((tmp_path / "li_bai/project.json").read_text(encoding="utf-8"))
    assert config.topic == "李白"
    state_path = tmp_path / "li_bai/.runtime/state.json"
    assert RuntimeState.model_validate_json(state_path.read_text()).current_state == S.CREATED
    assert main(["--projects-dir", str(tmp_path), "status", "li_bai"]) == 0
    output = capsys.readouterr().out
    assert "李白" in output and "CREATED" in output and "None" in output
    assert main(["--projects-dir", str(tmp_path), "review", "li_bai", "facts"]) == 0
    assert "No facts artifact" in capsys.readouterr().out
    snapshot = state_path.read_bytes()
    assert main(["--projects-dir", str(tmp_path), "resume", "li_bai"]) == 0
    assert "Would resume from: CREATED" in capsys.readouterr().out
    assert state_path.read_bytes() == snapshot
    assert not (tmp_path / "li_bai/facts").exists()


def test_existing_project_and_missing_project(tmp_path: Path, capsys) -> None:
    assert create(tmp_path) == 0
    original = (tmp_path / "li_bai/project.json").read_bytes()
    assert create(tmp_path) == 1
    assert (tmp_path / "li_bai/project.json").read_bytes() == original
    assert main(["--projects-dir", str(tmp_path), "status", "missing"]) == 1
    assert "Error:" in capsys.readouterr().err


def test_failed_resume_is_read_only(tmp_path: Path, capsys) -> None:
    create(tmp_path)
    path = tmp_path / "li_bai/.runtime/state.json"
    write_json(path, RuntimeState(current_state=S.FAILED, last_successful_state=S.STORY_APPROVED,
                                 failed_state=S.SCRIPT_GENERATING,
                                 latest_error="interrupted"), replace=True)
    before = path.read_bytes()
    assert main(["--projects-dir", str(tmp_path), "resume", "li_bai"]) == 0
    output = capsys.readouterr().out
    assert "Would resume from: SCRIPT_GENERATING" in output
    assert "Last successful state: STORY_APPROVED" in output
    assert main(["--projects-dir", str(tmp_path), "status", "li_bai"]) == 0
    output = capsys.readouterr().out
    assert "Failed state: SCRIPT_GENERATING" in output
    assert "Last successful state: STORY_APPROVED" in output
    assert path.read_bytes() == before


def test_module_entrypoint(tmp_path: Path) -> None:
    env = os.environ | {"PYTHONPATH": os.pathsep.join([str(Path(__file__).resolve().parents[1] / "src"),
                                                os.environ.get("PYTHONPATH", "")]),
                        "PYTHONIOENCODING": "utf-8"}
    result = subprocess.run([sys.executable, "-m", "history_studio", "--projects-dir", str(tmp_path),
                             "create", "--project-id", "li_bai", "--topic", "李白"],
                            capture_output=True, text=True, encoding="utf-8", env=env, check=False)
    assert result.returncode == 0, result.stderr
    assert "李白" in result.stdout


def test_invalid_config_creates_no_project(tmp_path: Path) -> None:
    assert main(["--projects-dir", str(tmp_path), "create", "--project-id", "bad",
                 "--topic", "test", "--budget", "-1"]) == 1
    assert not (tmp_path / "bad").exists()


def test_status_latest_version_and_review(tmp_path: Path, capsys) -> None:
    from history_studio.storage import ArtifactStore
    create(tmp_path)
    store = ArtifactStore(tmp_path / "li_bai")
    config = ProjectConfig(project_id="li_bai", topic="topic")
    store.save("research", config)
    store.save("research", config)
    store.save("facts", config)
    assert main(["--projects-dir", str(tmp_path), "status", "li_bai"]) == 0
    assert "research: v2" in capsys.readouterr().out
    assert main(["--projects-dir", str(tmp_path), "review", "li_bai", "facts"]) == 0
    assert "facts_v1.json" in capsys.readouterr().out
    assert store.list_versions("approvals") == []
