from pathlib import Path
from types import SimpleNamespace

from history_studio.cli import main
from history_studio.models import ProjectConfig
from history_studio.research.openai_provider import RunConfiguration


def create(tmp_path: Path) -> None:
    assert main(["--projects-dir", str(tmp_path), "create", "--project-id", "test", "--topic", "李白",
                 "--research-scope", "李白出生至20岁以前的生平"]) == 0


def test_cli_scope_and_missing_api_key(tmp_path: Path, monkeypatch, capsys) -> None:
    create(tmp_path)
    project = ProjectConfig.model_validate_json((tmp_path / "test/project.json").read_text(encoding="utf-8"))
    assert project.research_scope == "李白出生至20岁以前的生平"
    monkeypatch.delenv("OPENAI_API_KEY", raising=False)
    assert main(["--projects-dir", str(tmp_path), "research", "test"]) == 1
    assert "OPENAI_API_KEY" in capsys.readouterr().err
    assert not (tmp_path / "test/research").exists()


def test_cli_persists_and_reuses_configuration(tmp_path: Path, monkeypatch) -> None:
    create(tmp_path)
    client = SimpleNamespace(close=lambda: None)
    monkeypatch.setattr("history_studio.research.openai_provider.create_client", lambda config: client)
    seen = []
    def fake_run(self, project, store, runtime_configuration=None):
        from history_studio.storage.artifact_store import write_json
        seen.append(runtime_configuration)
        write_json(store.project_dir / ".runtime/research_config.json", runtime_configuration, replace=True)
        return SimpleNamespace(progress=SimpleNamespace(status="LIMIT_REACHED"))
    monkeypatch.setattr("history_studio.research.agent.ResearchAgent.run", fake_run)
    config_path = tmp_path / "config.json"
    config = RunConfiguration()
    config.research.hard_budget_usd = 0.8
    config_path.write_text(config.model_dump_json(), encoding="utf-8")
    assert main(["--projects-dir", str(tmp_path), "research", "test", "--config", str(config_path)]) == 2
    assert main(["--projects-dir", str(tmp_path), "research", "test"]) == 2
    assert seen[0] == seen[1]


def test_config_error_does_not_echo_credentials(tmp_path: Path, capsys) -> None:
    create(tmp_path)
    config = tmp_path / "bad.json"
    config.write_text('{"api_key":"secret-do-not-log"}', encoding="utf-8")
    assert main(["--projects-dir", str(tmp_path), "research", "test", "--config", str(config)]) == 1
    output = capsys.readouterr()
    assert "secret-do-not-log" not in output.err + output.out
