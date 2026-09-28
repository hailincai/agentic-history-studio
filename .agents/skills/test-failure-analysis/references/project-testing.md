# Project testing reference

Paths below are relative to the repository root. Recheck the relevant files when using this reference; implementation can evolve.

## Layout and execution

- `pyproject.toml` declares Python >=3.13, pytest as a development dependency, `testpaths = ["tests"]`, and `pythonpath = ["src"]`.
- Production code is under `src/history_studio/`; pytest modules are under `tests/test_*.py`.
- Run commands from the repository root. A focused PowerShell command is:

  ```powershell
  python -B -m pytest -p no:cacheprovider -q tests/test_research_agent.py::test_autonomous_dispatch_completion_and_persistence --basetemp="$PWD\.runtime\pytest-temp"
  ```

  Replace the node with the actual failing test; quote node IDs when shell syntax requires it. Pytest manages/clears its basetemp directory, so keep valuable artifacts out of that directory and avoid sharing it between concurrent runs.
- `full_regression.ps1` recursively removes `.runtime\pytest-temp`, recreates it, and runs `python -B -m pytest -p no:cacheprovider -q` with that project-local basetemp. With no test selector, it runs the full configured suite. Inspect its current contents before execution; do not invoke it for a focused-only request.
- `.gitignore` excludes `.runtime/`, `.pytest_cache/`, generated Python files, `.env`, and local `projects/` runtime artifacts. Ignored files may still hold important diagnostic evidence; do not clear them as a general troubleshooting step.

## Offline boundary

`tests/conftest.py::offline_network_guard` is an autouse fixture that blocks `socket.socket.connect` and `socket.create_connection`, even with credentials configured. Preserve this guard. Use existing fake providers/tools or mocked SDK transports for regression tests. The `research` CLI is a separate live/paid path, not an offline reproduction command. Do not read or print `.env` contents to diagnose ordinary test failures.

## Shared setup to inspect first

- `tests/test_research_agent.py`: `action()`, `update()`, `calls()`, `setup_run()`, `settings()`, `FakeProvider`, `FakeTools`, `SOURCE`, and `TEXT`; `state()` and `ledger()` load persisted runtime results. Other test modules import these helpers directly.
- `update()` builds a checkpoint proposal; its default COVERED goal includes completion criteria and a coverage assessment. Overrides such as `new_fact`, `gap_status`, and `quote` can alter validity. Inspect the resulting complete proposal, not just the overridden field.
- `tests/test_research_validation.py::malformed_fact()` starts from `update()` and mutates the first fact. Preserve the intended invalid field when repairing unrelated fixture drift.
- For specialized scenarios inspect `tests/test_goal_coverage_resume.py`, `tests/test_evidence_carry_forward.py`, `tests/test_research_spans.py`, `tests/test_research_retrieval.py`, and `tests/test_research_knowledge.py` and their helper imports.

## Authoritative validation boundaries

Use these files as navigation targets rather than duplicating their algorithms:

| Boundary | Authoritative files and diagnostic focus |
| --- | --- |
| Goal coverage and legacy readability | `src/history_studio/models/research_package.py`: canonical goals/assessments, `terminal_assessment_error()`, package reference validation. Legacy COVERED data without an assessment can remain readable; parsing alone does not establish current completion eligibility. |
| Completion and progress | `src/history_studio/research/progress.py`: `completion_status()`, `progress_marks()`, `progress_signals()`. Completion includes coverage checks as well as quantitative minimums; inspect structural progress instead of assuming a new checkpoint version proves progress. |
| Proposal versus persisted evidence | `src/history_studio/research/actions.py`, `research/agent.py`, `research/spans.py`, and `models/research.py`: follow selection parsing, accepted-evidence carry-forward, canonical resolution, and final provenance validation. |
| Canonical source integrity versus visibility | `src/history_studio/research/source_store.py` verifies stored representations against deterministic `make_spans()`. `research/retrieval.py` and `research/knowledge.py` select context; `research/agent.py` owns operation authorization. Stored content or retrieval visibility alone must not be mistaken for read authorization. |
| Resume and accumulated usage | `src/history_studio/research/agent.py` and `research/usage.py`: resume requires the usage ledger and matching run identity; prior accepted progress is seeded without granting fresh carry-forward credit. Trace these alongside the checkpoint. |
| Persistence and diagnostics | `src/history_studio/storage/artifact_store.py`, `research/diagnostics.py`, and `research/evidence_diagnostics.py`: inspect acceptance/persistence and sanitized rejection details before interpreting a final FAILED or LIMIT_REACHED assertion. |

In the table, shortened `research/` and `models/` paths are under `src/history_studio/`.

## Documentation drift

`README.md` is useful orientation, but currently describes RAG as future work while `research/source_store.py`, `research/retrieval.py`, and `research/knowledge.py` already implement source persistence and bounded retrieval/context projection. Verify claims about context retention, retrieval, and evidence lifecycle against current code and focused tests. If intended behavior remains ambiguous, report that ambiguity rather than silently treating either an old assertion or implementation as the specification.
