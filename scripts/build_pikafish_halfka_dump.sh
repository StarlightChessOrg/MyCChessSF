#!/usr/bin/env bash
# Build Pikafish halfka_dump reference tool for WSL parity tests.
set -euo pipefail

ROOT="$(cd "$(dirname "$0")/.." && pwd)"
PIKAFISH_DIR="${PIKAFISH_DIR:-$ROOT/third_party/Pikafish}"
SRC="$PIKAFISH_DIR/src"
DUMP="$ROOT/third_party/halfka_dump"
TABLES="$ROOT/third_party/halfka_tables"

if [[ ! -d "$SRC" ]]; then
  echo "[pikafish] missing $SRC — download Pikafish into third_party/Pikafish first." >&2
  exit 1
fi

CXXFLAGS=(-std=c++17 -O2 -DNDEBUG -DIS_64BIT -I"$SRC")

echo "[pikafish] building halfka_tables ..."
g++ "${CXXFLAGS[@]}" "$ROOT/scripts/pikafish_halfka_tables.cpp" -o "$TABLES"

echo "[pikafish] building halfka_dump (standalone) ..."
g++ "${CXXFLAGS[@]}" "$ROOT/scripts/pikafish_halfka_dump.cpp" -o "$DUMP"

echo "[pikafish] built $TABLES and $DUMP"
