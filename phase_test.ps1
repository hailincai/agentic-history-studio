python -m pytest -q --tb=short *> full_regression_2026-10-09.log

$testExitCode = $LASTEXITCODE

Get-Content full_regression_2026-10-09.log -Tail 60

Write-Host "Pytest exit code: $testExitCode"