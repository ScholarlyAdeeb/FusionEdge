# Seed repeats (docs\PROTOCOL.md, "Seed repeats"): four protocol runs, ONE AT A
# TIME, each through scripts\win\protocol_run.ps1 (tests -> pre-flight -> run ->
# audit + gate). Interleaved so an interruption still leaves matched seed pairs.
# Stops at the first run that fails; never retries, never overwrites.
#
#   powershell -NoProfile -ExecutionPolicy Bypass -File scripts\win\seed_repeats.ps1
$ErrorActionPreference = 'Continue'
$root = (Resolve-Path (Join-Path $PSScriptRoot '..\..')).Path
Set-Location $root
$runs = @(
    @('baseline_simplecnn_inject_s1235', 'configs\baseline_inject_s1235.yaml'),
    @('fusionedge_main_s1235',           'configs\fusionedge_s1235.yaml'),
    @('baseline_simplecnn_inject_s1236', 'configs\baseline_inject_s1236.yaml'),
    @('fusionedge_main_s1236',           'configs\fusionedge_s1236.yaml')
)
New-Item -ItemType Directory -Force 'results\seed_repeats' | Out-Null
$log = "results\seed_repeats\seed_repeats_$(Get-Date -Format 'yyyyMMdd_HHmmss').txt"
function Say([string]$m) { $l = "$(Get-Date -Format 'HH:mm:ss')  $m"; Write-Host $l -ForegroundColor Yellow; Add-Content -Path $log -Value $l -Encoding utf8 }

$i = 0
foreach ($r in $runs) {
    $i++
    if (Test-Path "experiments\$($r[0])") { Say "[$i/4] $($r[0]) already exists; stopping (never overwrite)"; exit 1 }
    Say "[$i/4] start $($r[0]) ($($r[1]))"
    $t0 = Get-Date
    & powershell -NoProfile -NonInteractive -ExecutionPolicy Bypass -File scripts\win\protocol_run.ps1 -Run $r[0] -Cfg $r[1]
    $code = $LASTEXITCODE
    $ok = (Test-Path "results\checks\$($r[0]).json") -and ((Get-Content "results\checks\$($r[0]).json" -Raw | ConvertFrom-Json).passed)
    Say ("[$i/4] $($r[0]) exit $code, gate passed: $ok, {0:N1} min" -f ((Get-Date) - $t0).TotalMinutes)
    if ($code -ne 0 -or -not $ok) { Say "STOP: $($r[0]) did not pass; remaining runs not started"; exit 1 }
}
Say "all 4 seed repeats complete and accepted"
