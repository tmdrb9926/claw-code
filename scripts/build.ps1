$ScriptDir = Split-Path -Parent $MyInvocation.MyCommand.Path
$RootDir = Split-Path -Parent $ScriptDir
$RustDir = Join-Path $RootDir "rust"
$env:PATH = "$env:USERPROFILE\.cargo\bin;$env:PATH"

Write-Host "========================================="
Write-Host "  Claw Code Build"
Write-Host "========================================="

Set-Location $RustDir

Write-Host "[1/2] cargo build --release..."
cargo build --release --workspace
if ($LASTEXITCODE -ne 0) { exit $LASTEXITCODE }

Write-Host "[2/2] Done!" -ForegroundColor Green
Write-Host ""
Write-Host "Binaries:"
Write-Host "  claw:         $RustDir\target\release\claw.exe"
Write-Host "  claw-gateway: $RustDir\target\release\claw-gateway.exe"
Write-Host ""
Write-Host "Run:"
Write-Host "  .\scripts\run-claw.ps1"
Write-Host "  .\scripts\run-gateway.ps1"
