---
name: test-failure-analysis
description: Diagnose and resolve failing pytest tests in Agentic History Studio, including regressions, stale expectations, fixture problems, and environment failures. Use for reported test failures while respecting diagnostic-only or test-only scope.
---

# Test failure analysis

## Establish scope and reproduce

1. Establish the authorized scope from the current request and applicable instructions before changing files: diagnosis only, test migration, or implementation fix; permitted files and test scope; prohibited actions. Do not ask again for authorization already provided. If a necessary fix exceeds scope, report the boundary and stop before making that change.
2. Inspect `git status --short` and relevant diffs before diagnosis. Preserve existing work and distinguish it from changes made during this task.
3. Read [project-testing.md](references/project-testing.md) for repository test commands, shared helpers, and authoritative validation locations. Verify relevant details against current code.
4. Reproduce the smallest relevant failing test or parameterized case first, within the permitted test scope. Capture the exact command, failure, and expected versus actual behavior. If execution is prohibited or blocked, use supplied output and read-only tracing; explicitly report that reproduction was not performed.

## Diagnose before fixing

5. Inspect shared fixtures/helpers and their callers before editing individual tests. Expand the actual generated input, including defaults and parameter overrides; do not infer its validity from the helper name.
6. Trace the relevant execution path from input through parsing, validation, runtime handling, persistence/progress accounting, and assertion as applicable. Identify the first divergence. A final status mismatch can be a downstream symptom of an earlier rejected checkpoint.
7. Classify the failure with evidence; more than one category may apply:
   - **Production defect:** implementation violates the intended current contract.
   - **Stale/incorrect test expectation:** assertion contradicts intended current behavior.
   - **Fixture/test-data problem:** setup fails to represent the scenario being tested.
   - **Legacy compatibility issue:** historical data parsing or migration differs from current-write requirements.
   - **Persistence/checkpoint/resume issue:** accepted state, runtime history, or authorization is lost or incorrectly restored.
   - **Environment/tooling issue:** interpreter, dependencies, permissions, paths, or test infrastructure prevent a valid run.
8. Identify the root cause and supporting execution path before making a fix. Separate confirmed findings from hypotheses. If evidence is insufficient, continue focused investigation or report the missing evidence; do not guess a patch.

## Fix and verify within scope

9. Prefer the smallest correct fix. For stale shared data, update the helper only where its intended scenario requires it and check dependent callers. Preserve each test's original purpose and intentional malformed/legacy inputs.
10. Never weaken a domain invariant merely to make a test pass. Do not bypass provenance, authorization, completion, progress, or budget checks; disable the offline guard; or replace a meaningful assertion with a weaker one to hide a defect.
11. After a fix, rerun the focused failing tests and the smallest relevant neighboring tests within scope. Run broader regression testing when authorized and appropriate for the change's reach. Honor explicit focused-only or no-test restrictions; skill invocation does not authorize paid/live research, project resume, or commits.
12. Inspect the final diff for unrelated edits. Report verification limits honestly; a focused pass does not establish a full-suite pass.

## Final report

Include:

- Root cause, with relevant code/fixture locations and supporting evidence.
- Failure classification, distinguishing primary and contributing causes.
- Files changed, or none.
- Production code changed: **YES/NO**.
- Tests executed: exact commands and scope, or not run with reason.
- Test results: pass/fail counts and remaining failures grouped by cause when available.
- Remaining failures or risks, including unverified behavior and any scope blocker.
