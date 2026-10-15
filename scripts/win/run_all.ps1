# Unattended FusionEdge pipeline (docs/PROTOCOL.md):
#   duplicate guard -> pytest -> baseline v5 -> GATE -> FusionEdge -> 7 ablations -> tables
#
#   powershell -NoProfile -ExecutionPolicy Bypass -File scripts\win\run_all.ps1
#
# * The baseline must pass audit_run.py AND check_run.py, or the pipeline STOPS before
#   FusionEdge (no FusionEdge result is produced against an unaccepted baseline).
# * Each run is trained in THIS window with a live progress bar; a dashboard window
#   (scripts\monitor.py --runs) shows every run's curves in the browser.
# * Re-running is safe: finished runs (metrics.json present) are skipped, and a run
#   interrupted mid-training is resumed from its own last.pt (scheduler + RNG restored).
# * Everything is logged to results\pipeline\run_all_<timestamp>.txt; per-run checks
#   go to results\checks\.
$ErrorActionPreference = 'Continue'
$root = (Resolve-Path (Join-Path $PSScriptRoot '..\..')).Path
Set-Location $root
$host.UI.RawUI.WindowTitle = "FusionEdge PIPELINE"
$logDir = Join-Path $root 'results\pipeline'
$chkDir = Join-Path $root 'results\checks'
New-Item -ItemType Directory -Force $logDir, $chkDir | Out-Null
$stamp = Get-Date -Format 'yyyyMMdd_HHmmss'
$log = Join-Path $logDir "run_all_$stamp.txt"
function Say([string]$m) { $line = "$(Get-Date -Format 'HH:mm:ss')  $m"; Write-Host $line -ForegroundColor Cyan; Add-Content -Path $log -Value $line -Encoding utf8 }

# baseline first (gate), then the main model, then the ablations
$plan = @(
    @{ cfg = 'configs\baseline.yaml';                  run = 'baseline_simplecnn_v6'   ; gate = $true },
    @{ cfg = 'configs\fusionedge.yaml';                run = 'fusionedge_main' },
    @{ cfg = 'configs\ablation_no_cross.yaml';         run = 'ablation_no_cross' },
    @{ cfg = 'configs\ablation_no_edge.yaml';          run = 'ablation_no_edge' },
    @{ cfg = 'configs\ablation_avg_fusion.yaml';       run = 'ablation_avg_fusion' },
    @{ cfg = 'configs\ablation_no_se.yaml';            run = 'ablation_no_se' },
    @{ cfg = 'configs\ablation_dense_conv.yaml';       run = 'ablation_dense_conv' },
    @{ cfg = 'configs\ablation_unshared_stem.yaml';    run = 'ablation_unshared_stem' },
    @{ cfg = 'configs\ablation_rgb_head.yaml';         run = 'ablation_direct_y_head' }
)

Say "FusionEdge pipeline $stamp  root=$root"
Say ("python: " + (Get-Command python).Source + "  " + (& python --version 2>&1))

# ---- duplicate guard ---------------------------------------------------------
$alive = @(Get-CimInstance Win32_Process | Where-Object { $_.Name -like 'python*.exe' -and $_.CommandLine -match 'run_experiment\.py' })
if ($alive.Count -gt 0) {
    foreach ($p in $alive) { Say "ABORT: training already running pid=$($p.ProcessId) $($p.CommandLine)" }
    Read-Host "press Enter to close"; exit 3
}

# ---- tests --------------------------------------------------------------------
$testOut = Join-Path $logDir "pytest_$stamp.txt"
Say "pytest -> $testOut"
& python -m pytest -q -p no:cacheprovider 2>&1 | ForEach-Object { "$_" } | Tee-Object -FilePath $testOut
if ($LASTEXITCODE -ne 0) { Say "ABORT: tests failed"; Read-Host "press Enter to close"; exit 4 }
Say ("tests: " + ((Get-Content $testOut | Select-String 'passed|failed' | Select-Object -Last 1).Line))

# ---- dashboard ----------------------------------------------------------------
Start-Process cmd -WorkingDirectory $root -ArgumentList '/k', 'title FusionEdge DASHBOARD && python scripts\monitor.py --runs'
Say "dashboard window started (sweep overview)"

function Invoke-Run($item) {
    $run = $item.run; $dir = "experiments\$run"
    if (Test-Path "$dir\metrics.json") { Say "[$run] already complete, skipping training"; return $true }
    $resume = @()
    if ((Test-Path $dir) -and (Test-Path "$dir\checkpoints\last.pt")) {
        Say "[$run] found interrupted run -> resuming from its last.pt"
        $resume = @('--resume', "$dir\checkpoints\last.pt")
    } elseif (Test-Path $dir) {
        Say "[$run] ABORT-RUN: $dir exists without a checkpoint; not overwriting it"; return $false
    }
    Say "[$run] training ($($item.cfg))"
    $t0 = Get-Date
    # Out-Host: keep the trainer's stdout out of this function's return value
    & python scripts\run_experiment.py --config $item.cfg --progress always @resume | Out-Host
    $code = $LASTEXITCODE
    Say ("[$run] run_experiment exit $code after {0:N1} min" -f ((Get-Date) - $t0).TotalMinutes)
    return ($code -eq 0 -and (Test-Path "$dir\metrics.json"))
}

function Invoke-Checks($run) {
    $dir = "experiments\$run"
    & python scripts\audit_run.py $dir 2>&1 | ForEach-Object { "$_" } | Tee-Object -FilePath (Join-Path $chkDir "${run}_audit.txt") | Out-Null
    $a = $LASTEXITCODE
    & python scripts\check_run.py $dir 2>&1 | ForEach-Object { "$_" } | Tee-Object -FilePath (Join-Path $chkDir "${run}_check.txt") | Out-Null
    $c = $LASTEXITCODE
    Say "[$run] audit=$(if ($a -eq 0) {'PASS'} else {'FAIL'}) check=$(if ($c -eq 0) {'PASS'} else {'FAIL'})"
    Get-Content (Join-Path $chkDir "${run}_check.txt") | Select-String 'FAIL\]' | ForEach-Object { Say "    $($_.Line.Trim())" }
    return ($a -eq 0 -and $c -eq 0)
}

$summary = @()
foreach ($item in $plan) {
    $ok = Invoke-Run $item
    if (-not $ok) {
        $summary += "$($item.run): TRAIN/EVAL FAILED"
        if ($item.gate) { Say "STOP: baseline did not complete; FusionEdge not started."; break }
        continue
    }
    $passed = Invoke-Checks $item.run
    $summary += "$($item.run): checks $(if ($passed) {'PASS'} else {'FAIL'})"
    if ($item.gate -and -not $passed) {
        Say "STOP: baseline failed its acceptance gate; FusionEdge not started (protocol)."
        break
    }
}

# ---- tables (generated from artefacts only) ----------------------------------
& python scripts\make_figures.py --protocol-only --out results\tables.md 2>&1 | ForEach-Object { "$_" } | Tee-Object -FilePath (Join-Path $logDir "make_figures_$stamp.txt")
Say "tables -> results\tables.md (exit $LASTEXITCODE)"

Say "==== SUMMARY ===="
foreach ($s in $summary) { Say "  $s" }
Say "pipeline finished. Log: $log"
Read-Host "press Enter to close"
