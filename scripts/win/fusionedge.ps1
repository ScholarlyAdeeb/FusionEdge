# FusionEdge main run ONLY, under docs\PROTOCOL.md with the same gates as baseline v6:
# tests (incl. GPU) -> pre-flight -> full run -> audit + gate. Stops at the first
# failure. Never starts any ablation.
#
#   powershell -NoProfile -ExecutionPolicy Bypass -File scripts\win\fusionedge.ps1
#
# Log: results\fusionedge\fusionedge_<timestamp>.txt (every step, exit codes, durations).
$ErrorActionPreference = 'Continue'
$root = (Resolve-Path (Join-Path $PSScriptRoot '..\..')).Path
Set-Location $root
$host.UI.RawUI.WindowTitle = "FusionEdge - main run"
$Run = 'fusionedge_main'
$Cfg = 'configs\fusionedge.yaml'
$logDir = Join-Path $root 'results\fusionedge'
$chk = Join-Path $root 'results\checks'
New-Item -ItemType Directory -Force $logDir, $chk | Out-Null
$stamp = Get-Date -Format 'yyyyMMdd_HHmmss'
$log = Join-Path $logDir "fusionedge_$stamp.txt"
function Say([string]$m) { $l = "$(Get-Date -Format 'HH:mm:ss')  $m"; Write-Host $l -ForegroundColor Cyan; Add-Content -Path $log -Value $l -Encoding utf8 }
function Stop-Here([string]$why) { Say "STOP: $why"; Say "nothing else was started. log: $log"; Read-Host "press Enter to close"; exit 1 }

Say "fusionedge pipeline $stamp  root=$root"
Say ("python: " + (Get-Command python).Source + "  " + (& python --version 2>&1))

# 0. guards: no concurrent training, never overwrite a run
$alive = @(Get-CimInstance Win32_Process | Where-Object { $_.Name -like 'python*.exe' -and $_.CommandLine -match 'run_experiment\.py' })
if ($alive.Count -gt 0) { Stop-Here "training already running: pid $($alive[0].ProcessId)" }
if (Test-Path "experiments\$Run") { Stop-Here "experiments\$Run already exists; refusing to overwrite" }
$gpu = & python -c "import torch; print(torch.cuda.is_available(), torch.cuda.get_device_name(0) if torch.cuda.is_available() else '-')" 2>&1
Say "cuda: $gpu"
if ("$gpu" -notmatch '^True') { Stop-Here "CUDA not available" }

# 1. full test suite; GPU tests must RUN, not skip
$t = Join-Path $logDir "pytest_$stamp.txt"
Say "1/4 pytest -> $t"
& python -m pytest -q -rs -p no:cacheprovider 2>&1 | ForEach-Object { "$_" } | Tee-Object -FilePath $t | Out-Host
$code = $LASTEXITCODE
$summary = (Get-Content $t | Select-String '\d+ (passed|failed)' | Select-Object -Last 1).Line
Say "    pytest exit $code : $summary"
if ($code -ne 0) { Stop-Here "tests failed" }
if ($summary -match 'skipped') { Get-Content $t | Select-String '^SKIPPED' | ForEach-Object { Say "    $($_.Line)" }; Stop-Here "some tests were skipped; GPU tests must run on this machine" }

# 2. pre-flight on THIS model's config (loss precision, smoke training, checkpoints, resume, test exclusion)
$p = Join-Path $logDir "preflight_$stamp.txt"
Say "2/4 pre-flight -> $p"
& python scripts\preflight_v6.py --config $Cfg 2>&1 | ForEach-Object { "$_" } | Tee-Object -FilePath $p | Out-Host
$code = $LASTEXITCODE
Get-Content $p | Select-String -CaseSensitive 'FAIL\]|PREFLIGHT|info:' | ForEach-Object { Say "    $($_.Line.Trim())" }
if ($code -ne 0) { Stop-Here "pre-flight failed" }

# 3. the full run (documented protocol: configs\default.yaml + configs\fusionedge.yaml)
Start-Process cmd -WorkingDirectory $root -ArgumentList '/k', "title FusionEdge DASHBOARD - $Run && timeout /t 20 >nul && python scripts\monitor.py --run $Run"
Say "3/4 training $Run (dashboard window opening)"
$t0 = Get-Date
& python scripts\run_experiment.py --config $Cfg --progress always | Out-Host
$code = $LASTEXITCODE
Say ("    run_experiment exit $code after {0:N1} min" -f ((Get-Date) - $t0).TotalMinutes)
if ($code -ne 0 -or -not (Test-Path "experiments\$Run\metrics.json")) { Stop-Here "training/evaluation did not complete" }

# 4. audit + acceptance gate
Say "4/4 audit + gate"
& python scripts\audit_run.py "experiments\$Run" 2>&1 | ForEach-Object { "$_" } | Tee-Object -FilePath (Join-Path $chk "${Run}_audit.txt") | Out-Host
$a = $LASTEXITCODE
& python scripts\check_run.py "experiments\$Run" 2>&1 | ForEach-Object { "$_" } | Tee-Object -FilePath (Join-Path $chk "${Run}_check.txt") | Out-Host
$c = $LASTEXITCODE
Say "    audit $(if ($a -eq 0) {'PASS'} else {'FAIL'}), gate $(if ($c -eq 0) {'PASS'} else {'FAIL'})"
Get-Content (Join-Path $chk "${Run}_check.txt") | Select-String 'FAIL\]' | ForEach-Object { Say "    $($_.Line.Trim())" }
if ($a -ne 0 -or $c -ne 0) { Stop-Here "FusionEdge failed audit or gate; diagnose before anything else" }
Say "FusionEdge complete and accepted by the automated gate. Visual review pending. Nothing else started."
Read-Host "press Enter to close"
