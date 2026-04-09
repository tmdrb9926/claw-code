#!/bin/bash
# Claw Code + Ollama 로컬 에이전트 빌드 스크립트
set -e

SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
ROOT_DIR="$(dirname "$SCRIPT_DIR")"
RUST_DIR="$ROOT_DIR/rust"

export PATH="$HOME/.cargo/bin:$PATH"

echo "========================================="
echo "  Claw Code 빌드"
echo "========================================="

cd "$RUST_DIR"

echo "[1/3] cargo check..."
cargo check --workspace

echo "[2/3] cargo build --release..."
cargo build --release --workspace

echo "[3/3] 빌드 완료!"
echo ""
echo "바이너리 위치:"
echo "  claw:         $RUST_DIR/target/release/claw"
echo "  claw-gateway: $RUST_DIR/target/release/claw-gateway"
echo ""
echo "다음 명령으로 실행하세요:"
echo "  ./scripts/run-claw.sh"
echo "  ./scripts/run-gateway.sh"
