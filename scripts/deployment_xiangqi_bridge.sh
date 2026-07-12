#!/usr/bin/env bash
# Launcher for 象眸 SF ↔ 相弈象棋 HTTP bridge (Chess98-compatible :9494).
set -euo pipefail
DEPLOY="$(cd "$(dirname "$0")/.." && pwd)"
PYTHON="${PYTHON:-python3}"
export PYTHONPATH="$DEPLOY/lib:${PYTHONPATH:-}"

args=("$@")
has_book=0
has_nnue=0
for a in "${args[@]}"; do
  case "$a" in
    --book|--book=*) has_book=1 ;;
    --nnue|--nnue=*) has_nnue=1 ;;
    --no-book) has_book=1 ;;
    --no-nnue-default) has_nnue=1 ;;
  esac
done
if [[ $has_book -eq 0 && -f "$DEPLOY/db/BOOK.DAT" ]]; then
  args=(--book "$DEPLOY/db/BOOK.DAT" "${args[@]}")
fi
if [[ $has_nnue -eq 0 && -f "$DEPLOY/model/quantized.xqnnue.bin" ]]; then
  args=(--nnue "$DEPLOY/model/quantized.xqnnue.bin" "${args[@]}")
fi

exec "$PYTHON" -m mycchess_sf.xiangqi_auto_bridge "${args[@]}"
