#!/usr/bin/env bash
# Build xqwlight_core.so on Linux and install the Python package (run from repo root)
set -euo pipefail
ROOT="$(cd "$(dirname "$0")/.." && pwd)"
cd "$ROOT/cpp"
mkdir -p build
cd build
cmake .. -DPython_EXECUTABLE="$(command -v python3)"
cmake --build . --config Release -j"$(nproc 2>/dev/null || echo 2)"
SO="$(find . -maxdepth 1 -name 'xqwlight_core*.so' | head -1)"
if [[ -z "$SO" ]]; then
  echo "xqwlight_core*.so not found" >&2
  exit 1
fi
cp -f "$SO" "$ROOT/"
echo "Copied $(basename "$SO") -> $ROOT/"
cd "$ROOT"
pip install -e .
echo "Done. Run: mycchess-play-web --host 0.0.0.0 --port 8080"
