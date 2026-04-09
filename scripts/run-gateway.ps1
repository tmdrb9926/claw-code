param(
    [Parameter(Position=0)]
    [string]$Action = "serve",
    [Parameter(Position=1, ValueFromRemainingArguments)]
    [string[]]$Remaining
)

$ScriptDir = Split-Path -Parent $MyInvocation.MyCommand.Path
$RootDir = Split-Path -Parent $ScriptDir
$Gateway = Join-Path $RootDir "rust\target\release\claw-gateway.exe"
$Port = if ($env:GATEWAY_PORT) { $env:GATEWAY_PORT } else { "8443" }

if (-not (Test-Path $Gateway)) {
    Write-Host "claw-gateway.exe not found. Build first:" -ForegroundColor Red
    Write-Host "  .\scripts\build.ps1"
    exit 1
}

if ($Action -eq "setup") {
    Write-Host "========================================="
    Write-Host "  Gateway Setup"
    Write-Host "========================================="
    & $Gateway key create --name "default"
    Write-Host ""
    Write-Host "Done! Start gateway:  .\scripts\run-gateway.ps1"
    exit 0
}

if ($Action -eq "key") {
    & $Gateway key $Remaining
    exit 0
}

if ($Action -eq "serve") {
    $ollamaOk = $false
    try {
        $null = Invoke-RestMethod "http://127.0.0.1:11434/api/tags" -TimeoutSec 3
        $ollamaOk = $true
    }
    catch {}

    if ($ollamaOk) {
        Write-Host "Ollama: running" -ForegroundColor Green
    }
    else {
        Write-Host "WARNING: Ollama not responding" -ForegroundColor Yellow
    }

    Write-Host "========================================="
    Write-Host "  Claw Gateway (port: $Port)"
    Write-Host "========================================="
    & $Gateway serve --port $Port
    exit 0
}

Write-Host "Usage:"
Write-Host "  .\scripts\run-gateway.ps1              # start gateway"
Write-Host "  .\scripts\run-gateway.ps1 setup        # initial setup"
Write-Host "  .\scripts\run-gateway.ps1 key create --name 'name'"
Write-Host "  .\scripts\run-gateway.ps1 key list"
