#!/bin/bash
# Claw Code 로컬 LLM 에이전트 실행 스크립트
set -e

SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
ROOT_DIR="$(dirname "$SCRIPT_DIR")"
CLAW="$ROOT_DIR/rust/target/release/claw"

export PATH="$HOME/.cargo/bin:$PATH"

# ---- 설정 ----
# Ollama 서버 주소 (기본: localhost)
export OLLAMA_BASE_URL="${OLLAMA_BASE_URL:-http://localhost:11434}"
# 기본 모델 (gemma4:31b 또는 다른 Ollama 모델)
export OLLAMA_MODEL="${OLLAMA_MODEL:-gemma4:31b}"
# 컨텍스트 윈도우 (기본: 32K)
export OLLAMA_NUM_CTX="${OLLAMA_NUM_CTX:-32768}"
# 모델 메모리 유지 (-1m = 영구)
export OLLAMA_KEEP_ALIVE="${OLLAMA_KEEP_ALIVE:--1m}"

# ---- 바이너리 체크 ----
if [ ! -f "$CLAW" ]; then
    echo "claw 바이너리가 없습니다. 먼저 빌드하세요:"
    echo "  ./scripts/build.sh"
    exit 1
fi

# ---- Ollama 상태 확인 ----
echo "Ollama 서버 확인 중... ($OLLAMA_BASE_URL)"
if ! curl -s "$OLLAMA_BASE_URL/api/tags" > /dev/null 2>&1; then
    echo "Ollama 서버가 실행 중이 아닙니다!"
    echo "  ollama serve  로 먼저 시작하세요."
    exit 1
fi

# 모델 확인
MODEL_EXISTS=$(curl -s "$OLLAMA_BASE_URL/api/tags" | grep -c "$OLLAMA_MODEL" || true)
if [ "$MODEL_EXISTS" -eq 0 ]; then
    echo "모델 '$OLLAMA_MODEL'이 없습니다."
    echo "사용 가능한 모델:"
    curl -s "$OLLAMA_BASE_URL/api/tags" | python3 -c "
import sys, json
data = json.load(sys.stdin)
for m in data.get('models', []):
    print(f\"  - {m['name']}  ({m['details'].get('parameter_size', 'unknown')})\" )
" 2>/dev/null || curl -s "$OLLAMA_BASE_URL/api/tags"
    echo ""
    echo "모델을 다운로드하려면: ollama pull gemma4:31b"
    echo "또는 환경변수를 변경하세요: export OLLAMA_MODEL=<모델명>"
    exit 1
fi

echo "모델: $OLLAMA_MODEL"
echo "컨텍스트: $OLLAMA_NUM_CTX 토큰"
echo "========================================="

# ---- 실행 ----
if [ $# -gt 0 ]; then
    # 인자가 있으면 prompt 모드
    "$CLAW" --model "$OLLAMA_MODEL" prompt "$*"
else
    # 인자 없으면 인터랙티브 모드
    "$CLAW" --model "$OLLAMA_MODEL"
fi
