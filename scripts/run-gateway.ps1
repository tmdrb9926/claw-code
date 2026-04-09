# Claw Gateway 실행
$ErrorActionPreference = "Stop"
$RootDir = Split-Path -Parent (Split-Path -Parent $PSScriptRoot)
if (-not $RootDir) { $RootDir = Split-Path -Parent (Split-Path -Parent $MyInvocation.MyCommand.Path) }
$Gateway = Join-Path $RootDir "rust\target\release\claw-gateway.exe"
$Port = if ($env:GATEWAY_PORT) { $env:GATEWAY_PORT } else { "8443" }

# 바이너리 확인
if (-not (Test-Path $Gateway)) {
    Write-Host "claw-gateway.exe가 없습니다. 먼저 빌드하세요:" -ForegroundColor Red
    Write-Host "  .\scripts\build.ps1"
    exit 1
}

$action = if ($args.Count -gt 0) { $args[0] } else { "serve" }

switch ($action) {
    "setup" {
        Write-Host "========================================="
        Write-Host "  Gateway 초기 설정"
        Write-Host "========================================="
        Write-Host ""
        Write-Host "[1/2] API 키 생성 중..."
        & $Gateway key create --name "default"
        Write-Host ""
        Write-Host "[2/2] 설정 완료!"
        Write-Host ""
        Write-Host "게이트웨이 시작:"
        Write-Host "  .\scripts\run-gateway.ps1"
        Write-Host ""
        Write-Host "외부에서 접속:"
        Write-Host "  curl http://<서버IP>:$Port/api/tags -H 'Authorization: Bearer <키>'"
    }
    "key" {
        $remaining = $args[1..($args.Count-1)]
        & $Gateway key @remaining
    }
    "serve" {
        # Ollama 확인
        try {
            $null = Invoke-RestMethod "http://127.0.0.1:11434/api/tags" -TimeoutSec 3
            Write-Host "Ollama: 실행 중" -ForegroundColor Green
        } catch {
            Write-Host "WARNING: Ollama 서버가 응답하지 않습니다." -ForegroundColor Yellow
        }

        Write-Host "========================================="
        Write-Host "  Claw Gateway 시작 (포트: $Port)"
        Write-Host "========================================="
        & $Gateway serve --port $Port
    }
    default {
        Write-Host "사용법:"
        Write-Host "  .\scripts\run-gateway.ps1              # 게이트웨이 시작"
        Write-Host "  .\scripts\run-gateway.ps1 setup        # 초기 설정 (키 생성)"
        Write-Host "  .\scripts\run-gateway.ps1 key create --name '이름'"
        Write-Host "  .\scripts\run-gateway.ps1 key list"
    }
}
