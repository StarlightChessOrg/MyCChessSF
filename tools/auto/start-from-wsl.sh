#!/usr/bin/env bash
# 从 WSL 一键转交 Windows PowerShell 启动 Selenium（不要在 WSL 里直接 npm start）
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
WIN_DIR="$(wslpath -w "$SCRIPT_DIR")"
LEVEL="${OPPONENT_LEVEL:-9}"
HOST="${BRIDGE_HOST:-127.0.0.1}"
PORT="${BRIDGE_PORT:-9494}"
EXTRA=()

if [[ "${KILL_EDGE:-0}" == "1" ]]; then
  EXTRA+=(-KillEdge)
fi

exec powershell.exe -NoProfile -ExecutionPolicy Bypass \
  -File "${WIN_DIR}\\start-windows.ps1" \
  -OpponentLevel "$LEVEL" \
  -BridgeHost "$HOST" \
  -BridgePort "$PORT" \
  "${EXTRA[@]}"
