# Claw Code 빌드
$ErrorActionPreference = "Stop"
$RootDir = Split-Path -Parent (Split-Path -Parent $PSScriptRoot)
if (-not $RootDir) { $RootDir = Split-Path -Parent (Split-Path -Parent $MyInvocation.MyCommand.Path) }
$RustDir = Join-Path $RootDir "rust"

Write-Host "========================================="
Write-Host "  Claw Code 빌드"
Write-Host "========================================="

Set-Location $RustDir

Write-Host "[1/2] cargo build --release..."
cargo build --release --workspace
if ($LASTEXITCODE -ne 0) { exit $LASTEXITCODE }

Write-Host "[2/2] 빌드 완료!" -ForegroundColor Green
Write-Host ""
Write-Host "바이너리 위치:"
Write-Host "  claw:         $RustDir\target\release\claw.exe"
Write-Host "  claw-gateway: $RustDir\target\release\claw-gateway.exe"
Write-Host ""
Write-Host "실행:"
Write-Host "  .\scripts\run-claw.ps1"
Write-Host "  .\scripts\run-gateway.ps1"
