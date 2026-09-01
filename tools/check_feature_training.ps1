param(
    [string]$RunDir = ""
)

$projectRoot = Split-Path -Parent $PSScriptRoot
if ([string]::IsNullOrWhiteSpace($RunDir)) {
    $latestRun = Get-ChildItem -LiteralPath (Join-Path $projectRoot "train_log") -Directory -ErrorAction SilentlyContinue |
        Where-Object { $_.Name -ne "smoke_20260803" } |
        Sort-Object LastWriteTime -Descending |
        Select-Object -First 1
    if ($null -eq $latestRun) {
        throw "No feature training run was found."
    }
    $RunDir = $latestRun.FullName
}

$RunDir = [System.IO.Path]::GetFullPath($RunDir)
Write-Output "RUN_DIR=$RunDir"

foreach ($statusName in @("launcher_status.json", "status.json", "final_result.json")) {
    $statusPath = Join-Path $RunDir $statusName
    if (Test-Path -LiteralPath $statusPath) {
        Write-Output "=== $statusName ==="
        Get-Content -LiteralPath $statusPath -Raw
    }
}

$metricsPath = Join-Path $RunDir "metrics.jsonl"
if (Test-Path -LiteralPath $metricsPath) {
    Write-Output "=== latest metrics ==="
    Get-Content -LiteralPath $metricsPath -Tail 1
}

Write-Output "=== checkpoints ==="
Get-ChildItem -LiteralPath $RunDir -File -Filter "*.pt" -ErrorAction SilentlyContinue |
    Select-Object Name, Length, LastWriteTime

$logPath = Join-Path $RunDir "train.log"
if (Test-Path -LiteralPath $logPath) {
    Write-Output "=== train log tail ==="
    Get-Content -LiteralPath $logPath -Tail 30
} else {
    $cacheLog = Join-Path $RunDir "cache_build.log"
    if (Test-Path -LiteralPath $cacheLog) {
        Write-Output "=== cache log tail ==="
        Get-Content -LiteralPath $cacheLog -Tail 20
    }
}

Write-Output "=== GPU ==="
& nvidia-smi --query-gpu=name,memory.total,memory.used,memory.free,utilization.gpu --format=csv,noheader,nounits
