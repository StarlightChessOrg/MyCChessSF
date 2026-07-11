#!/usr/bin/env bash
# Launcher for MyCChessSF web play (human vs XQWL wizard).
set -euo pipefail
DEPLOY="$(cd "$(dirname "$0")/.." && pwd)"
PYTHON="${PYTHON:-python3}"
export PYTHONPATH="$DEPLOY/lib:${PYTHONPATH:-}"

args=("$@")
has_book=0
for a in "${args[@]}"; do
  case "$a" in
    --book|--book=*) has_book=1 ;;
  esac
done
if [[ $has_book -eq 0 ]]; then
  args=(--book "$DEPLOY/db/BOOK.DAT" "${args[@]}")
fi

exec "$PYTHON" -m mycchess_sf.play_web "${args[@]}"
