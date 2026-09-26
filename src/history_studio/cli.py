import argparse
import sys
from pathlib import Path

from history_studio.models import ProjectConfig
from history_studio.storage.artifact_store import ArtifactStore, safe_component, write_json
from history_studio.workflow import ApprovalStage, ProjectState, ProjectStateMachine, RuntimeState


def project_path(root: Path, project_id: str) -> Path:
    root = root.resolve()
    path = root / safe_component(project_id)
    if path.resolve().parent != root:
        raise ValueError("Project directory escapes projects root")
    return path


def read_project(path: Path) -> tuple[ProjectConfig, RuntimeState]:
    config = ProjectConfig.model_validate_json((path / "project.json").read_text(encoding="utf-8"))
    if config.project_id != path.name:
        raise ValueError("Project configuration ID does not match its directory")
    state = RuntimeState.model_validate_json(
        (path / ".runtime" / "state.json").read_text(encoding="utf-8"))
    return config, state


def create_project(args: argparse.Namespace, path: Path) -> None:
    config = ProjectConfig(project_id=args.project_id, topic=args.topic, language=args.language,
                           target_duration_minutes=args.duration, budget_usd=args.budget,
                           output_width=args.width, output_height=args.height)
    # Exclusive directory creation prevents accidental reinitialization of an existing project.
    path.mkdir(parents=True, exist_ok=False)
    write_json(path / "project.json", config)
    write_json(path / ".runtime" / "state.json", RuntimeState())
    print(f"Created project {config.project_id}: {config.topic} [CREATED]")


def show_status(path: Path, config: ProjectConfig, state: RuntimeState) -> None:
    print(config.model_dump_json(indent=2))
    print(f"Workflow state: {state.current_state}")
    print(f"Last successful state: {state.last_successful_state}")
    if state.failed_state is not None:
        print(f"Failed state: {state.failed_state}")
    if state.latest_error:
        print(f"Latest error: {state.latest_error}")
    store = ArtifactStore(path)
    print("Latest artifact versions:")
    found = False
    for artifact_type in ("research", "facts", "story", "script", "storyboard", "approvals"):
        versions = store.list_versions(artifact_type)
        if versions:
            found = True
            print(f"  {artifact_type}: v{versions[-1]}")
    if not found:
        print("  None")


def review(path: Path, stage: str) -> None:
    store = ArtifactStore(path)
    versions = store.list_versions(stage)
    if not versions:
        print(f"No {stage} artifact is available for human review.")
    else:
        print(f"Review {stage} v{versions[-1]}: {path / stage / f'{stage}_v{versions[-1]}.json'}")
    print("An external human approval decision is required. This command records no decision.")


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Agentic History Studio — Phase 1 infrastructure")
    parser.add_argument("--projects-dir", type=Path, default=Path("projects"))
    commands = parser.add_subparsers(dest="command", required=True)
    create = commands.add_parser("create", help="Create project configuration and runtime state")
    create.add_argument("--project-id", required=True)
    create.add_argument("--topic", required=True)
    create.add_argument("--language", default="zh-CN")
    create.add_argument("--duration", type=float, default=5)
    create.add_argument("--budget", type=float, default=20)
    create.add_argument("--width", type=int, default=1280)
    create.add_argument("--height", type=int, default=720)
    for command in ("status", "review", "resume"):
        subparser = commands.add_parser(command)
        subparser.add_argument("project_id")
        if command == "review":
            subparser.add_argument("stage", choices=[stage.value for stage in ApprovalStage])
    args = parser.parse_args(argv)
    try:
        path = project_path(args.projects_dir, args.project_id)
        if args.command == "create":
            create_project(args, path)
        else:
            config, state = read_project(path)
            if args.command == "status":
                show_status(path, config, state)
            elif args.command == "review":
                review(path, args.stage)
            elif state.current_state == ProjectState.COMPLETE:
                print("Project is complete; there is no stage to resume.")
            else:
                print(f"Would resume from: {ProjectStateMachine(state).resume_state}")
                print(f"Last successful state: {state.last_successful_state}")
                if state.latest_error:
                    print(f"Latest error: {state.latest_error}")
                print("Phase 1 does not execute agents or change workflow state.")
    except (OSError, ValueError) as exc:
        print(f"Error: {exc}", file=sys.stderr)
        return 1
    return 0
