# Claw Code 로컬 LLM 에이전트 실행
$ErrorActionPreference = "Stop"
$RootDir = Split-Path -Parent (Split-Path -Parent $PSScriptRoot)
if (-not $RootDir) { $RootDir = Split-Path -Parent (Split-Path -Parent $MyInvocation.MyCommand.Path) }
$Claw = Join-Path $RootDir "rust\target\release\claw.exe"

# 설정
$env:OLLAMA_BASE_URL = if ($env:OLLAMA_BASE_URL) { $env:OLLAMA_BASE_URL } else { "http://localhost:11434" }
$env:OLLAMA_MODEL = if ($env:OLLAMA_MODEL) { $env:OLLAMA_MODEL } else { "gemma4:31b" }
$env:OLLAMA_NUM_CTX = if ($env:OLLAMA_NUM_CTX) { $env:OLLAMA_NUM_CTX } else { "32768" }
$env:OLLAMA_KEEP_ALIVE = if ($env:OLLAMA_KEEP_ALIVE) { $env:OLLAMA_KEEP_ALIVE } else { "-1m" }

# 바이너리 확인
if (-not (Test-Path $Claw)) {
    Write-Host "claw.exe가 없습니다. 먼저 빌드하세요:" -ForegroundColor Red
    Write-Host "  .\scripts\build.ps1"
    exit 1
}

# Ollama 확인
Write-Host "Ollama 서버 확인 중... ($env:OLLAMA_BASE_URL)"
try {
    $null = Invoke-RestMethod "$env:OLLAMA_BASE_URL/api/tags" -TimeoutSec 3
} catch {
    Write-Host "Ollama 서버가 실행 중이 아닙니다!" -ForegroundColor Red
    Write-Host "  ollama serve  로 먼저 시작하세요."
    exit 1
}

# 모델 확인
$models = Invoke-RestMethod "$env:OLLAMA_BASE_URL/api/tags"
$found = $models.models | Where-Object { $_.name -like "*$($env:OLLAMA_MODEL)*" }
if (-not $found) {
    Write-Host "모델 '$env:OLLAMA_MODEL'이 없습니다." -ForegroundColor Yellow
    Write-Host "사용 가능한 모델:"
    $models.models | ForEach-Object { Write-Host "  - $($_.name)  ($($_.details.parameter_size))" }
    Write-Host ""
    Write-Host "모델 다운로드: ollama pull gemma4:31b"
    exit 1
}

Write-Host "모델: $env:OLLAMA_MODEL" -ForegroundColor Green
Write-Host "컨텍스트: $env:OLLAMA_NUM_CTX 토큰"
Write-Host "========================================="

# 실행
if ($args.Count -gt 0) {
    $prompt = $args -join " "
    & $Claw --model $env:OLLAMA_MODEL prompt $prompt
} else {
    & $Claw --model $env:OLLAMA_MODEL
}
