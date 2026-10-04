---
name: test-failure-analysis
description: Diagnose and resolve failing pytest tests in Agentic History Studio, including regressions, stale expectations, fixture problems, and environment failures. Use for reported test failures while respecting diagnostic-only or test-only scope.
---

# Test failure analysis

## Establish scope and reproduce

1. Establish the authorized scope from the current request and applicable instructions before changing files: diagnosis only, test migration, or implementation fix; permitted files and test scope; prohibited actions. Do not ask again for authorization already provided. If a necessary fix exceeds scope, report the boundary and stop before making that change.
2. Inspect `git status --short` and relevant diffs before diagnosis. Preserve existing work and distinguish it from changes made during this task.
3. Read [project-testing.md](references/project-testing.md) for repository test commands, shared helpers, and authoritative validation locations. Verify relevant details against current code.
4. Use supplied failure evidence first; reproduce the smallest relevant failing test or parameterized case only when it adds decision-relevant information, within the permitted scope and diagnostic budget below. Capture the exact command, failure, and expected versus actual behavior. If execution is prohibited or blocked, use supplied output and read-only tracing; explicitly report that reproduction was not performed.

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
8. Establish enough evidence to classify the failure and recommend the smallest next action. Separate confirmed findings from hypotheses; a unique root cause is not required. Make a fix only when evidence supports that specific change and scope permits it. Otherwise investigate within the budget or report missing evidence; do not guess a patch.

## Fix and verify within scope

9. Prefer the smallest correct fix. For stale shared data, update the helper only where its intended scenario requires it and check dependent callers. Preserve each test's original purpose and intentional malformed/legacy inputs.
10. Never weaken a domain invariant merely to make a test pass. Do not bypass provenance, authorization, completion, progress, or budget checks; disable the offline guard; or replace a meaningful assertion with a weaker one to hide a defect.
11. After a fix, rerun the focused failing tests and the smallest relevant neighboring tests within scope. Run broader regression testing only within authorized scope and when appropriate for the change's reach; the full-suite policy below always applies. Honor explicit focused-only or no-test restrictions; skill invocation does not authorize paid/live research, project resume, or commits.
12. Inspect the final diff for unrelated edits. Report verification limits honestly; a focused pass does not establish a full-suite pass.

## Final report

Include:

- Confirmed or likely cause, with relevant code/fixture locations, supporting evidence, and confidence level.
- Failure classification, distinguishing primary and contributing causes.
- Files changed, or none.
- Production code changed: **YES/NO**.
- Tests executed: exact commands and scope, or not run with reason.
- Test results: pass/fail counts and remaining failures grouped by cause when available.
- Remaining failures or risks, including unverified behavior and any scope blocker.

## Diagnostic budget and termination

**Diagnosis optimizes for decision-relevant evidence, not certainty.**

Default engineering limits, unless the user explicitly changes the budget:

- Maximum **5 targeted test invocations** per diagnostic task, including reproduction.
- Maximum **3 narrowing/bisection attempts**, counted within those 5 invocations, not added to them.
- Never repeat an equivalent test command merely to seek confidence.
- Prefer static inspection once runtime evidence has sufficiently narrowed the problem.
- Treat expensive tests as consuming more budget: reduce the execution count as runtime increases. A 1-second test may support more narrowing within the limits; a 60-second group should strongly favor static inspection after one or two informative executions.

These limits are ceilings, not a requirement to run five tests. Track invocations, narrowing attempts, and elapsed execution time. Supplied results consume no new invocations and may already be sufficient. Budget exhaustion ends diagnosis; a new budget requires explicit user direction. Post-fix verification is separate from diagnostic execution, remains within authorized scope, and must not become a way to resume narrowing.

After each invocation ask: **"What new decision-relevant information would another execution provide?"** If there is no clear answer, STOP. Do not keep testing merely to prove a unique root cause.

Stop diagnosis immediately when any of these is true:

1. A minimal or sufficiently small reproducer identifies the likely contaminating test/module/state.
2. Evidence is sufficient to classify the failure and recommend the smallest next action, even without proving a unique root cause.
3. Remaining hypotheses cannot reasonably be distinguished within the diagnostic budget, or either execution limit is reached.
4. Further narrowing requires another broad or expensive test run.
5. Results indicate intermittent/environmental behavior rather than a deterministic defect.
6. The next meaningful step would exceed authorized scope.
7. Repeated execution produces no materially new information.

STOP means end diagnostic investigation and report findings; do not continue static tracing indefinitely. Proceed to a fix and focused verification only if the evidence already supports that change and the request authorizes it. Otherwise recommend the smallest next step without executing it.

## Full-suite policy

Do NOT run the full repository regression suite by default. Full regression is normally performed manually by the user at phase/milestone closure. Analyze supplied full-suite failures without rerunning them. Run the full suite only when the current request explicitly authorizes it; "diagnose this full-suite failure" is not authorization to rerun it. Explicit full-suite authorization does not remove diagnostic limits or stop conditions.

## Flaky and order-dependent failures

When the full suite fails but a targeted fresh-process run passes, first consider order dependency, leaked process/global state, environment interaction, or fixture isolation. Do not immediately change the failing test or production code. Inspect shared mutable state and fixture cleanup; use targeted prefix/module pairing or limited bisection only within the diagnostic budget and stop conditions. Pair suspected predecessors with the failing target in the same process to preserve the suspected interaction.

A sufficiently small polluting module/set is an acceptable diagnosis. Do not require one exact individual test when further narrowing has poor cost/benefit. Apply the intermittent/environmental stop condition when results support that classification; order dependency alone does not justify repeated runs.

## Report bounded uncertainty

When stopping without a unique root cause, include in the final report:

- Evidence established and hypotheses eliminated (or none); avoid claiming elimination beyond the observed scope.
- Leading remaining hypothesis/hypotheses and confidence level, with the reason for that confidence.
- Stopping reason and budget used.
- One specific smallest next diagnostic step and the information it would distinguish; identify any required authorization.
- Whether production changes are currently justified, and whether test changes are currently justified, separately.
- Whether the issue blocks the requested milestone; if its criteria are unknown, state that the impact is undetermined.

**"Unable to prove the unique root cause within budget" is NOT equivalent to "diagnosis failed."** A successful diagnosis may conclude: likely cross-suite state contamination; production fix not justified yet; one specific next experiment recommended. Report bounded uncertainty without presenting it as failure.
