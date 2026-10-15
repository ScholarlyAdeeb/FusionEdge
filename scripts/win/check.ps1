# Wait for a run to finish (metrics.json + benchmark.json written, no run_experiment.py
# alive), then run the post-run audit and the acceptance gate. Output goes to
# results\checks\<run>_audit.txt and results\checks\<run>_check.txt (+ .json, _samples.png).
#
#   check.ps1 -Run baseline_simplecnn_v4
param([Parameter(Mandatory = $true)][string]$Run)
$ErrorActionPreference = 'Continue'
$root = (Resolve-Path (Join-Path $PSScriptRoot '..\..')).Path
Set-Location $root
$runDir = "experiments\$Run"
$out = Join-Path $root 'results\checks'
New-Item -ItemType Directory -Force $out | Out-Null

Write-Host "waiting for $runDir to finish (metrics.json, benchmark.json, no trainer process)..."
while ($true) {
    $alive = @(Get-CimInstance Win32_Process | Where-Object { $_.Name -like 'python*.exe' -and $_.CommandLine -match 'run_experiment\.py' })
    if ((Test-Path "$runDir\metrics.json") -and (Test-Path "$runDir\benchmark.json") -and $alive.Count -eq 0) { break }
    Start-Sleep -Seconds 15
}
Write-Host "== audit_run =="
& python scripts\audit_run.py $runDir 2>&1 | ForEach-Object { "$_" } | Tee-Object -FilePath (Join-Path $out "${Run}_audit.txt")
Write-Host "== check_run =="
& python scripts\check_run.py $runDir 2>&1 | ForEach-Object { "$_" } | Tee-Object -FilePath (Join-Path $out "${Run}_check.txt")
Write-Host "done: results\checks\${Run}_*.txt"
