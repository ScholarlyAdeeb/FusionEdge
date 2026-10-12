# scripts/retry_mfnet.ps1
#
# Bounded retry loop for MFNet acquisition. Drive's per-file quota on the MFNet
# archive is transient (it is a widely-mirrored file, so it is frequently over
# quota), so a later attempt can succeed where this one failed. This loop makes
# at most $MaxAttempts attempts, $SleepMinutes apart, and exits as soon as the
# registry reports status=ok. It never fabricates data: on persistent failure it
# exits non-zero and real-data training stays BLOCKED.
param(
    [int]$MaxAttempts = 8,
    [int]$SleepMinutes = 25
)

$ErrorActionPreference = 'Continue'
$log = 'data\_mfnet_retry.log'
for ($i = 1; $i -le $MaxAttempts; $i++) {
    "[$(Get-Date -Format s)] attempt $i/$MaxAttempts" | Tee-Object -FilePath $log -Append
    python scripts\download_data.py --dataset mfnet --timeout-min 10 2>&1 |
        Tee-Object -FilePath $log -Append | Select-Object -Last 4

    $status = (python -c "import json;print(json.load(open('data/registry/MFNet.json'))['status'])" 2>$null)
    "[$(Get-Date -Format s)] registry status: $status" | Tee-Object -FilePath $log -Append
    if ($status -eq 'ok') {
        "[$(Get-Date -Format s)] SUCCESS on attempt $i" | Tee-Object -FilePath $log -Append
        exit 0
    }
    if ($i -lt $MaxAttempts) {
        "[$(Get-Date -Format s)] sleeping $SleepMinutes min" | Tee-Object -FilePath $log -Append
        Start-Sleep -Seconds ($SleepMinutes * 60)
    }
}
"[$(Get-Date -Format s)] EXHAUSTED $MaxAttempts attempts; MFNet still unavailable" |
    Tee-Object -FilePath $log -Append
exit 1