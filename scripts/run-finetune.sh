#!/bin/bash
# Gemma 4 파인튜닝 실행 스크립트
set -e

SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
ROOT_DIR="$(dirname "$SCRIPT_DIR")"

export PATH="$HOME/.cargo/bin:$PATH"

echo "========================================="
echo "  Gemma 4 파인튜닝 환경 설정"
echo "========================================="

# ---- Python 환경 확인 ----
if ! command -v python3 &>/dev/null && ! command -v python &>/dev/null; then
    echo "Python이 설치되어 있지 않습니다."
    echo "  winget install Python.Python.3.13"
    exit 1
fi

PYTHON=$(command -v python3 || command -v python)
echo "Python: $($PYTHON --version)"

# ---- 가상환경 생성 ----
VENV_DIR="$HOME/.claw/finetune/venv"

case "${1:-}" in
    setup)
        echo ""
        echo "[1/3] 가상환경 생성..."
        if [ ! -d "$VENV_DIR" ]; then
            $PYTHON -m venv "$VENV_DIR"
            echo "  생성 완료: $VENV_DIR"
        else
            echo "  이미 존재: $VENV_DIR"
        fi

        echo "[2/3] Unsloth 설치..."
        source "$VENV_DIR/Scripts/activate" 2>/dev/null || source "$VENV_DIR/bin/activate"
        pip install --upgrade pip
        pip install unsloth --quiet
        echo "  Unsloth 설치 완료"

        echo "[3/3] GPU 확인..."
        $PYTHON -c "
import torch
if torch.cuda.is_available():
    gpu = torch.cuda.get_device_name(0)
    mem = torch.cuda.get_device_properties(0).total_memory / 1024**3
    print(f'  GPU: {gpu} ({mem:.0f}GB)')
else:
    print('  WARNING: CUDA를 사용할 수 없습니다!')
"
        echo ""
        echo "파인튜닝 환경 설정 완료!"
        echo "학습 데이터를 준비한 후 다음 명령으로 실행하세요:"
        echo "  ./scripts/run-finetune.sh train"
        ;;

    train)
        echo ""
        if [ ! -d "$VENV_DIR" ]; then
            echo "환경이 설정되지 않았습니다. 먼저 실행하세요:"
            echo "  ./scripts/run-finetune.sh setup"
            exit 1
        fi

        source "$VENV_DIR/Scripts/activate" 2>/dev/null || source "$VENV_DIR/bin/activate"

        # 학습 데이터 확인
        TRAIN_DATA="$HOME/.claw/finetune/data/train.jsonl"
        if [ ! -f "$TRAIN_DATA" ]; then
            echo "학습 데이터가 없습니다: $TRAIN_DATA"
            echo ""
            echo "Claw Code를 사용하면서 자동으로 수집되거나,"
            echo "수동으로 JSONL 파일을 생성할 수 있습니다."
            echo ""
            echo "형식:"
            echo '  {"conversations": [{"role": "user", "content": "..."}, {"role": "assistant", "content": "..."}]}'
            exit 1
        fi

        SAMPLE_COUNT=$(wc -l < "$TRAIN_DATA")
        echo "학습 데이터: $TRAIN_DATA ($SAMPLE_COUNT 샘플)"
        echo ""

        # 학습 스크립트 확인/생성
        TRAIN_SCRIPT="$HOME/.claw/finetune/scripts/finetune.py"
        if [ ! -f "$TRAIN_SCRIPT" ]; then
            echo "학습 스크립트가 없습니다."
            echo "Claw Code에서 자동 생성됩니다:"
            echo "  claw finetune run"
            exit 1
        fi

        echo "학습 시작..."
        $PYTHON "$TRAIN_SCRIPT"
        echo ""
        echo "학습 완료! GGUF 파일을 확인하세요:"
        ls -la "$HOME/.claw/finetune/models/"*.gguf 2>/dev/null || echo "  (GGUF 파일을 찾을 수 없습니다)"
        ;;

    status)
        echo ""
        echo "=== 파인튜닝 상태 ==="
        echo ""

        # 세션 로그
        LOG_DIR="$HOME/.claw/logs"
        if [ -d "$LOG_DIR" ]; then
            LOG_COUNT=$(ls "$LOG_DIR"/*.jsonl 2>/dev/null | wc -l || echo 0)
            echo "세션 로그: ${LOG_COUNT}개"
        else
            echo "세션 로그: 없음"
        fi

        # 학습 데이터
        TRAIN_DATA="$HOME/.claw/finetune/data/train.jsonl"
        if [ -f "$TRAIN_DATA" ]; then
            SAMPLE_COUNT=$(wc -l < "$TRAIN_DATA")
            echo "학습 데이터: ${SAMPLE_COUNT} 샘플"
        else
            echo "학습 데이터: 없음"
        fi

        # 모델 버전
        MODEL_DIR="$HOME/.claw/finetune/models"
        if [ -d "$MODEL_DIR" ]; then
            echo "파인튜닝된 모델:"
            ls -lh "$MODEL_DIR"/*.gguf 2>/dev/null | awk '{print "  " $NF " (" $5 ")"}' || echo "  없음"
        else
            echo "파인튜닝된 모델: 없음"
        fi

        # Unsloth 환경
        if [ -d "$VENV_DIR" ]; then
            echo "Unsloth 환경: 설정됨"
        else
            echo "Unsloth 환경: 미설정 (./scripts/run-finetune.sh setup)"
        fi
        ;;

    *)
        echo "사용법:"
        echo "  ./scripts/run-finetune.sh setup    # Unsloth 환경 설정"
        echo "  ./scripts/run-finetune.sh train    # 파인튜닝 실행"
        echo "  ./scripts/run-finetune.sh status   # 현재 상태 확인"
        ;;
esac
