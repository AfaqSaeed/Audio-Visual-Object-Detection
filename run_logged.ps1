param(
    [Parameter(Mandatory = $true, Position = 0)]
    [string]$Name,

    [Parameter(Mandatory = $true, Position = 1)]
    [string]$CommandLine
)

$ErrorActionPreference = "Stop"

$repoRoot = Split-Path -Parent $MyInvocation.MyCommand.Path
$logsDir = Join-Path $repoRoot "logs"
if (-not (Test-Path -LiteralPath $logsDir)) {
    New-Item -ItemType Directory -Path $logsDir | Out-Null
}

$timestamp = Get-Date -Format "yyyyMMdd_HHmmss"
$safeName = ($Name -replace '[^A-Za-z0-9._-]', '_')
$sessionLog = Join-Path $logsDir "${timestamp}_${safeName}.log"
$metaLog = Join-Path $logsDir "${timestamp}_${safeName}.meta.txt"

$startTime = Get-Date

@(
    "name: $Name"
    "start_time: $($startTime.ToString('o'))"
    "working_directory: $PWD"
    "command: $CommandLine"
) | Set-Content -Path $metaLog

Write-Host "Logging command output to: $sessionLog"
Write-Host "Logging metadata to: $metaLog"
Write-Host "Running: $CommandLine"

$oldEap = $ErrorActionPreference
$ErrorActionPreference = "Continue"

try {
    Invoke-Expression $CommandLine 2>&1 | Tee-Object -FilePath $sessionLog
    $exitCode = $LASTEXITCODE
    if ($null -eq $exitCode) {
        $exitCode = 0
    }
}
finally {
    $ErrorActionPreference = $oldEap
}

$endTime = Get-Date
$duration = $endTime - $startTime

Add-Content -Path $metaLog -Value @(
    "end_time: $($endTime.ToString('o'))"
    "duration_seconds: $([math]::Round($duration.TotalSeconds, 3))"
    "exit_code: $exitCode"
    "output_log: $sessionLog"
)

if ($exitCode -ne 0) {
    Write-Error "Command failed with exit code $exitCode. See $sessionLog"
    exit $exitCode
}

Write-Host "Command completed successfully."
Write-Host "Exit code: $exitCode"
