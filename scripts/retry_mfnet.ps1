# scripts/retry_mfnet.ps1
#
# Conservative, bounded retry for MFNet acquisition.
#
# WHY THIS SCRIPT IS DELIBERATELY SLOW
# -------------------------------------
# The MFNet archive on Google Drive is quota-blocked ("have had many accesses" /
# HTTP 500 / redirect to an account sign-in page). Drive applies rate limits to
# repeatedly-hitting endpoints, so hammering it every 25 minutes is counterproductive:
# it is more likely to extend the block than to clear it, and it is inconsiderate
# to a third-party service. The previous version of this script retried 8 times at
# 25-minute intervals and every attempt failed for the same reason.
#
# The policy here is therefore:
#   * exponential backoff, starting at 6 hours, capped at 24 hours;
#   * a small default attempt budget (4), overridable on the command line;
#   * it stops permanently once Google returns a quota-class signal, because a
#     quota block is not something a client should try to out-wait;
#   * it honours data/_archives/ -- if you place the archive there by hand, no
#     network request is made at all.
#
# Usage:
#   powershell -ExecutionPolicy Bypass -File scripts\retry_mfnet.ps1
#   powershell -ExecutionPolicy Bypass -File scripts\retry_mfnet.ps1 -MaxAttempts 10
#   powershell -ExecutionPolicy Bypass -File scripts\retry_mfnet.ps1 -FirstDelayHours 12
param(
    [int]$MaxAttempts = 4,
    [double]$FirstDelayHours = 6,
    [double]$MaxDelayHours = 24
)

$ErrorActionPreference = 'Continue'
$log = 'data\_mfnet_retry.log'
Set-Location (Split-Path -Parent $PSScriptRoot)

function Write-Log($msg) {
    $line = "[$(Get-Date -Format s)] $msg"
    Write-Output $line
    Add-Content -Path $log -Value $line
}

function Get-RegistryStatus {
    if (Test-Path 'data\registry\MFNet.json') {
        try { return (Get-Content 'data\registry\MFNet.json' -Raw | ConvertFrom-Json).status }
        catch { return 'unreadable' }
    }
    return 'absent'
}

Write-Log "conservative retry loop starting: $MaxAttempts attempts, first delay ${FirstDelayHours}h"

$delay = $FirstDelayHours
for ($i = 1; $i -le $MaxAttempts; $i++) {
    # A manually-provided archive needs no network at all.
    if (Test-Path 'data\_archives\*.tar.gz') {
        Write-Log "archive found in data\_archives -- extracting without any network request"
    } else {
        Write-Log "attempt $i/$MaxAttempts -- waiting ${delay}h since the previous attempt"
        if ($i -gt 1) { Start-Sleep -Seconds ([int]($delay * 3600)) }
        $delay = [Math]::Min($delay * 2, $MaxDelayHours)

        Write-Log "attempt $i/$MaxAttempts -- contacting Google Drive once"
        python scripts\download_data.py --dataset mfnet --timeout-min 15 2>&1 |
            Select-Object -Last 3 | ForEach-Object { Write-Log "  $_" }
    }

    $status = Get-RegistryStatus
    Write-Log "registry status: $status"
    if ($status -eq 'ok') {
        Write-Log "SUCCESS on attempt $i -- dataset registered and verified"
        exit 0
    }

    if ($status -eq 'unavailable') {
        # A quota block does not clear by waiting; continuing to poll is not the
        # right response. Stop and let a human place the archive, or re-run later
        # on a deliberate schedule.
        Write-Log "still quota-blocked after $i attempt(s). Stopping rather than " +
                  "hammering Drive. Options: (a) place ir_det_dataset.tar.gz in " +
                  "data\_archives\ and run: python scripts\download_data.py " +
                  "--dataset mfnet   (b) re-run this script later"
        exit 1
    }
}

Write-Log "exhausted $MaxAttempts attempts; MFNet still unavailable"
exit 1