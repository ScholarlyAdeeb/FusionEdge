# FusionEdge launcher for Windows: duplicate-guard -> tests -> training window + dashboard.
#
#   launch.ps1 -Config configs\baseline.yaml -Run baseline_simplecnn_v4 [-StopRunning] [-SkipTests] [-NoTrain]
#
# * Refuses to start training while another run_experiment.py python process is alive
#   (no duplicate GPU jobs). Only python*.exe counts: the idle `cmd /k` window left
#   open after a finished run also contains 'run_experiment.py' in its command line. -StopRunning kills those processes first (process tree), and records
#   exactly which ones in the report.
# * Runs the full pytest suite and only launches training if it passes.
# * Training opens in its own console window with a live progress bar
#   (--progress always); the dashboard (scripts\monitor.py) opens in a second window
#   and in the browser.
# Every step is written to results\recovery\launch_<timestamp>.txt.
param(
    [string]$Config = "configs\baseline.yaml",
    [string]$Run = "",
    [switch]$StopRunning,
    [switch]$SkipTests,
    [switch]$NoTrain
)
$ErrorActionPreference = 'Continue'
$root = (Resolve-Path (Join-Path $PSScriptRoot '..\..')).Path
Set-Location $root
$logDir = Join-Path $root 'results\recovery'
New-Item -ItemType Directory -Force $logDir | Out-Null
$stamp = Get-Date -Format 'yyyyMMdd_HHmmss'
$report = Join-Path $logDir "launch_$stamp.txt"
function Say([string]$m) { Write-Host $m; Add-Content -Path $report -Value $m -Encoding utf8 }

Say "FusionEdge launch $stamp"
Say "root=$root config=$Config run=$Run stop=$StopRunning skipTests=$SkipTests noTrain=$NoTrain"
$py = Get-Command python -ErrorAction SilentlyContinue
if (-not $py) { Say "ERROR: python not on PATH"; exit 2 }
Say ("python: " + $py.Source + "  " + (& python --version 2>&1))

function Get-Runs { Get-CimInstance Win32_Process | Where-Object { $_.Name -like 'python*.exe' -and $_.CommandLine -match 'run_experiment\.py' } }

# ---- 1. duplicate guard ---------------------------------------------------
$procs = @(Get-Runs)
foreach ($p in $procs) { Say "RUNNING pid=$($p.ProcessId) started=$($p.CreationDate) cmd=$($p.CommandLine)" }
if ($procs.Count -eq 0) { Say "no run_experiment.py process running" }
if ($procs.Count -gt 0 -and $StopRunning) {
    foreach ($p in $procs) {
        Say ("stopping pid $($p.ProcessId): " + ((taskkill /T /F /PID $p.ProcessId 2>&1) -join ' | '))
    }
    Start-Sleep -Seconds 5
    $procs = @(Get-Runs)
}
if ($procs.Count -gt 0 -and -not $NoTrain) {
    Say "ABORT: a training process is still running; not launching a duplicate."
    exit 3
}

# ---- 2. tests -------------------------------------------------------------
if (-not $SkipTests) {
    $testOut = Join-Path $logDir "pytest_$stamp.txt"
    Say "== pytest (output: $testOut) =="
    & python -m pytest -q -p no:cacheprovider 2>&1 | ForEach-Object { "$_" } | Tee-Object -FilePath $testOut
    $code = $LASTEXITCODE
    Say "pytest exit code: $code"
    Get-Content $testOut -Tail 3 | ForEach-Object { Say "  $_" }
    if ($code -ne 0) { Say "ABORT: tests failed, not training."; exit 4 }
}

# ---- 3. training window + dashboard ---------------------------------------
if ($NoTrain) { Say "NoTrain: done."; exit 0 }
if (-not $Run) { Say "ERROR: -Run <run_name> is required to launch training"; exit 2 }
if (Test-Path (Join-Path $root "experiments\$Run")) {
    Say "ABORT: experiments\$Run already exists; pick a new run name (never overwrite a run)."
    exit 5
}
Say "launching training window: $Run"
Start-Process cmd -WorkingDirectory $root -ArgumentList '/k', "title FusionEdge TRAINING - $Run && python scripts\run_experiment.py --config $Config --progress always"
Start-Sleep -Seconds 25
Say "launching dashboard window"
Start-Process cmd -WorkingDirectory $root -ArgumentList '/k', "title FusionEdge DASHBOARD - $Run && python scripts\monitor.py --run $Run"
Say "launched."
exit 0
