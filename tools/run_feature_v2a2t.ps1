param(
    [ValidateSet("CMUMOSI", "CMUMOSEI", "CHSIMS")]
    [string]$Dataset = "CMUMOSEI",
    [string]$DatasetRoot = "",
    [string]$Python = "python",
    [string]$RunName = "",
    [int]$Seed = 42,
    [string]$Resume = "",
    [int]$WaitForProcessId = 0,
    [switch]$EvalTestEveryEpoch
)

$ErrorActionPreference = "Stop"
$projectRoot = Split-Path -Parent $PSScriptRoot
$datasetSlug = $Dataset.ToLowerInvariant()
if ([string]::IsNullOrWhiteSpace($DatasetRoot)) {
    if ($Dataset -eq "CMUMOSI") {
        $DatasetRoot = Join-Path $projectRoot "datasets\CMUMOSI"
    }
    elseif ($Dataset -eq "CHSIMS") {
        $DatasetRoot = Join-Path $projectRoot "datasets\CHSIMS"
    }
    else {
        $DatasetRoot = Join-Path $projectRoot "datasets\CMUMOSEI"
    }
}
if ([string]::IsNullOrWhiteSpace($RunName)) {
    $RunName = "${datasetSlug}_full_seed${Seed}_$(Get-Date -Format 'yyyyMMdd_HHmmss')"
}
$cachePath = Join-Path $projectRoot "cache\ebmc_${datasetSlug}_full.pt"
$runRoot = Join-Path $projectRoot "train_log\$RunName"
$launcherStatus = Join-Path $runRoot "launcher_status.json"
$cacheLog = Join-Path $runRoot "cache_build.log"
$trainLog = Join-Path $runRoot "train.log"

if ([string]::IsNullOrWhiteSpace($Resume)) {
    $existingTrainingArtifacts = @(
        (Join-Path $runRoot "metrics.jsonl"),
        (Join-Path $runRoot "best.pt"),
        (Join-Path $runRoot "last.pt"),
        (Join-Path $runRoot "final_result.json")
    ) | Where-Object { Test-Path -LiteralPath $_ }
    if ($existingTrainingArtifacts.Count -gt 0) {
        throw "Run directory already contains training artifacts: $runRoot. Use a new RunName, or pass -Resume with its last.pt."
    }
}
elseif (-not (Test-Path -LiteralPath $Resume)) {
    throw "Resume checkpoint not found: $Resume"
}

New-Item -ItemType Directory -Force -Path $runRoot | Out-Null
Set-Location $projectRoot

function Write-LauncherStatus {
    param(
        [string]$Status,
        [string]$Stage,
        [string]$Message = ""
    )
    [ordered]@{
        status = $Status
        stage = $Stage
        message = $Message
        pid = $PID
        run_name = $RunName
        run_root = $runRoot
        cache_path = $cachePath
        updated_at = (Get-Date).ToString("o")
    } | ConvertTo-Json | Set-Content -LiteralPath $launcherStatus -Encoding UTF8
}

function Invoke-PythonLogged {
    param(
        [string[]]$Arguments,
        [string]$LogPath
    )
    $previousErrorAction = $ErrorActionPreference
    $ErrorActionPreference = "Continue"
    & $Python @Arguments 2>&1 | Tee-Object -FilePath $LogPath
    $exitCode = $LASTEXITCODE
    $ErrorActionPreference = $previousErrorAction
    if ($exitCode -ne 0) {
        throw "Python command failed with exit code $exitCode"
    }
}

try {
    if ($WaitForProcessId -gt 0) {
        while (Get-Process -Id $WaitForProcessId -ErrorAction SilentlyContinue) {
            Write-LauncherStatus -Status "queued" -Stage "waiting" -Message "Waiting for process $WaitForProcessId to exit."
            Start-Sleep -Seconds 60
        }
    }

    if (-not (Test-Path -LiteralPath $cachePath)) {
        Write-LauncherStatus -Status "running" -Stage "cache" -Message "Building the full EBMC tensor cache."
        Invoke-PythonLogged -Arguments @(
            ".\tools\build_ebmc_feature_cache.py",
            "--dataset", $Dataset,
            "--dataset-root", $DatasetRoot,
            "--output", $cachePath
        ) -LogPath $cacheLog
    }

    Write-LauncherStatus -Status "running" -Stage "training" -Message "Training feature-domain V2A2T."
    if ($Dataset -eq "CMUMOSI") {
        $epochs = "150"
        $batchSize = "64"
        $hiddenDim = "256"
        $numExperts = "4"
        $expansion = "2"
        $dropout = "0.35"
        $learningRate = "0.0002"
        $weightDecay = "0.0005"
        $warmupRatio = "0.10"
        $patience = "25"
    }
    elseif ($Dataset -eq "CHSIMS") {
        $epochs = "60"
        $batchSize = "24"
        $hiddenDim = "64"
        $numExperts = "2"
        $expertDim = "32"
        $expansion = "2"
        $dropout = "0.35"
        $learningRate = "0.0002"
        $weightDecay = "0.002"
        $warmupRatio = "0.10"
        $patience = "10"
        $visualDepth = "2"
        $audioDepth = "3"
        $textDepth = "3"
        $auxiliaryWeight = "0.03"
        $correlationWeight = "0.05"
        $routerWeight = "0.0"
        $conditionScale = "0.03"
        $monitor = "Acc_3"
        $gradClip = "0.5"
    }
    else {
        $epochs = "60"
        $batchSize = "256"
        $hiddenDim = "512"
        $numExperts = "8"
        $expansion = "4"
        $dropout = "0.2"
        $learningRate = "0.0003"
        $weightDecay = "0.0001"
        $warmupRatio = "0.08"
        $patience = "12"
    }
    if ($Dataset -ne "CHSIMS") {
        $expertDim = "0"
        $visualDepth = "3"
        $audioDepth = "4"
        $textDepth = "4"
        $auxiliaryWeight = "0.1"
        $correlationWeight = "0.05"
        $routerWeight = "0.01"
        $conditionScale = "0.01"
        $monitor = "MAE"
        $gradClip = "1.0"
    }

    $trainArguments = @(
        ".\main_feature_v2a2t.py",
        "--dataset-name", $Dataset,
        "--cache-path", $cachePath,
        "--output-dir", $runRoot,
        "--epochs", $epochs,
        "--batch-size", $batchSize,
        "--num-workers", "0",
        "--seed", "$Seed",
        "--hidden-dim", $hiddenDim,
        "--visual-depth", $visualDepth,
        "--audio-depth", $audioDepth,
        "--text-depth", $textDepth,
        "--num-experts", $numExperts,
        "--expert-dim", $expertDim,
        "--expansion", $expansion,
        "--dropout", $dropout,
        "--learning-rate", $learningRate,
        "--weight-decay", $weightDecay,
        "--warmup-ratio", $warmupRatio,
        "--auxiliary-weight", $auxiliaryWeight,
        "--correlation-weight", $correlationWeight,
        "--router-weight", $routerWeight,
        "--grad-clip", $gradClip,
        "--audio-condition-scale-init", $conditionScale,
        "--text-condition-scale-init", $conditionScale,
        "--monitor", $monitor,
        "--patience", $patience,
        "--precision", "fp16"
    )
    if (-not [string]::IsNullOrWhiteSpace($Resume)) {
        $trainArguments += @("--resume", $Resume)
    }
    if ($EvalTestEveryEpoch) {
        $trainArguments += "--eval-test-every-epoch"
    }
    elseif ($Dataset -eq "CHSIMS") {
        $trainArguments += "--eval-test-every-epoch"
    }
    Invoke-PythonLogged -Arguments $trainArguments -LogPath $trainLog
    Write-LauncherStatus -Status "completed" -Stage "training" -Message "Feature-domain V2A2T training completed."
}
catch {
    Write-LauncherStatus -Status "failed" -Stage "pipeline" -Message $_.Exception.Message
    Write-Error $_
    exit 1
}
