# smoke-test.ps1

$ErrorActionPreference = "Stop"

# Force UTF-8 for console / child-process communication
[Console]::InputEncoding  = [System.Text.UTF8Encoding]::new($false)
[Console]::OutputEncoding = [System.Text.UTF8Encoding]::new($false)
$OutputEncoding = [System.Text.UTF8Encoding]::new($false)

$projectId = "li_bai_early_life"
$topic = "李白"
$scope = "李白出生至20岁以前的生平"

Write-Host "=== Clean old project ==="
Remove-Item -Recurse -Force ".\projects\$projectId" -ErrorAction SilentlyContinue

Write-Host "Project exists after cleanup:"
Test-Path ".\projects\$projectId"

Write-Host "=== Create project ==="
python -X utf8 -m history_studio create `
    --project-id $projectId `
    --topic $topic `
    --research-scope $scope `
    --language "zh-CN" `
    --duration 5 `
    --budget 20

Write-Host "=== Project status ==="
python -X utf8 -m history_studio status $projectId

Write-Host "=== Smoke configuration ==="
Get-Content ".\examples\research-smoke.json" -Encoding UTF8

Write-Host "=== Start research smoke ==="
python -X utf8 -m history_studio research $projectId `
    --config ".\examples\research-smoke.json"