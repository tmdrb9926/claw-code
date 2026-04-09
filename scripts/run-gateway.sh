#!/bin/bash
# Claw Gateway (API 게이트웨이) 실행 스크립트
set -e

SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
ROOT_DIR="$(dirname "$SCRIPT_DIR")"
GATEWAY="$ROOT_DIR/rust/target/release/claw-gateway"

export PATH="$HOME/.cargo/bin:$PATH"

# ---- 설정 ----
PORT="${GATEWAY_PORT:-8443}"
OLLAMA_URL="${OLLAMA_BASE_URL:-http://127.0.0.1:11434}"

# ---- 바이너리 체크 ----
if [ ! -f "$GATEWAY" ]; then
    echo "claw-gateway 바이너리가 없습니다. 먼저 빌드하세요:"
    echo "  ./scripts/build.sh"
    exit 1
fi

# ---- 키 관리 ----
case "${1:-}" in
    key)
        shift
        "$GATEWAY" key "$@"
        exit 0
        ;;
    setup)
        echo "========================================="
        echo "  Gateway 초기 설정"
        echo "========================================="
        echo ""
        echo "[1/2] API 키 생성 중..."
        KEY=$("$GATEWAY" key create --name "default" 2>&1 | grep -o 'claw-sk-[a-zA-Z0-9]*' || true)
        if [ -z "$KEY" ]; then
            echo "  키 생성 실패. 수동으로 생성하세요:"
            echo "    $GATEWAY key create --name \"default\""
            exit 1
        fi
        echo "  API 키: $KEY"
        echo ""
        echo "[2/2] 설정 완료!"
        echo ""
        echo "게이트웨이 시작:"
        echo "  ./scripts/run-gateway.sh"
        echo ""
        echo "외부에서 접속:"
        echo "  curl http://<서버IP>:$PORT/api/tags -H 'Authorization: Bearer $KEY'"
        exit 0
        ;;
    ""|serve)
        # 아래에서 서버 시작
        ;;
    *)
        echo "사용법:"
        echo "  ./scripts/run-gateway.sh              # 게이트웨이 시작"
        echo "  ./scripts/run-gateway.sh setup         # 초기 설정 (키 생성)"
        echo "  ./scripts/run-gateway.sh key create --name \"이름\"  # 키 생성"
        echo "  ./scripts/run-gateway.sh key list      # 키 목록"
        echo "  ./scripts/run-gateway.sh key revoke <id>  # 키 폐기"
        exit 0
        ;;
esac

# ---- Ollama 상태 확인 ----
echo "Ollama 서버 확인 중... ($OLLAMA_URL)"
if ! curl -s "$OLLAMA_URL/api/tags" > /dev/null 2>&1; then
    echo "WARNING: Ollama 서버가 응답하지 않습니다."
    echo "  게이트웨이는 시작하지만 요청이 실패할 수 있습니다."
fi

# ---- 키 존재 확인 ----
KEY_COUNT=$("$GATEWAY" key list 2>&1 | grep -c "claw-sk" || true)
if [ "$KEY_COUNT" -eq 0 ]; then
    echo ""
    echo "API 키가 없습니다! 먼저 설정하세요:"
    echo "  ./scripts/run-gateway.sh setup"
    exit 1
fi

echo "========================================="
echo "  Claw Gateway 시작"
echo "  포트: $PORT"
echo "  Ollama: $OLLAMA_URL"
echo "  API 키: ${KEY_COUNT}개 등록됨"
echo "========================================="

"$GATEWAY" serve --port "$PORT"
