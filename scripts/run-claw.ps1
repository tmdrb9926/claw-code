param(
    [Parameter(Position=0, ValueFromRemainingArguments)]
    [string[]]$Prompt
)

$ScriptDir = Split-Path -Parent $MyInvocation.MyCommand.Path
$RootDir = Split-Path -Parent $ScriptDir
$Claw = Join-Path $RootDir "rust\target\release\claw.exe"

$env:OLLAMA_BASE_URL = if ($env:OLLAMA_BASE_URL) { $env:OLLAMA_BASE_URL } else { "http://localhost:11434" }
$env:OLLAMA_MODEL = if ($env:OLLAMA_MODEL) { $env:OLLAMA_MODEL } else { "gemma4:31b" }
$env:OLLAMA_NUM_CTX = if ($env:OLLAMA_NUM_CTX) { $env:OLLAMA_NUM_CTX } else { "32768" }
$env:OLLAMA_KEEP_ALIVE = if ($env:OLLAMA_KEEP_ALIVE) { $env:OLLAMA_KEEP_ALIVE } else { "-1m" }

if (-not (Test-Path $Claw)) {
    Write-Host "claw.exe not found: $Claw" -ForegroundColor Red
    Write-Host "Build first:  .\scripts\build.ps1"
    exit 1
}

Write-Host "Ollama: $env:OLLAMA_BASE_URL"
$ollamaOk = $false
try {
    $null = Invoke-RestMethod "$env:OLLAMA_BASE_URL/api/tags" -TimeoutSec 3
    $ollamaOk = $true
}
catch {}

if (-not $ollamaOk) {
    Write-Host "Ollama not running! Start with: ollama serve" -ForegroundColor Red
    exit 1
}

$models = Invoke-RestMethod "$env:OLLAMA_BASE_URL/api/tags"
$found = $models.models | Where-Object { $_.name -like "*$($env:OLLAMA_MODEL)*" }
if (-not $found) {
    Write-Host "Model '$env:OLLAMA_MODEL' not found." -ForegroundColor Yellow
    Write-Host "Available:"
    $models.models | ForEach-Object { Write-Host "  - $($_.name)" }
    Write-Host ""
    Write-Host "Download: ollama pull $env:OLLAMA_MODEL"
    Write-Host "Or set:   `$env:OLLAMA_MODEL='qwen3-30b'"
    exit 1
}

Write-Host "Model: $env:OLLAMA_MODEL" -ForegroundColor Green
Write-Host "Context: $env:OLLAMA_NUM_CTX tokens"
Write-Host "========================================="

if ($Prompt -and $Prompt.Count -gt 0) {
    $text = $Prompt -join " "
    & $Claw --model $env:OLLAMA_MODEL prompt $text
}
else {
    & $Claw --model $env:OLLAMA_MODEL
}
