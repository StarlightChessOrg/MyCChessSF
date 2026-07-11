#!/usr/bin/env bash
# Build xqwlight_core.so on Linux and install the Python package (run from repo root)
set -euo pipefail
ROOT="$(cd "$(dirname "$0")/.." && pwd)"
PYTHON="${PYTHON:-$(command -v python3)}"
# C++ optimization flag for xqwlight_core (override: CXXOPT=-O2 bash scripts/build_linux.sh)
CXXOPT="${CXXOPT:--O3}"

echo "[build] Python: $PYTHON"
echo "[build] CXXOPT: $CXXOPT"
"$PYTHON" -m pip install -q -U pip
"$PYTHON" -m pip install -q pybind11 numpy pillow
# pybind11 CMake config must match the same interpreter used for the extension
if ! "$PYTHON" -c "import pybind11; print('pybind11', pybind11.__version__, pybind11.get_cmake_dir())"; then
  echo "[build] Failed to import pybind11 after pip install" >&2
  exit 1
fi

cd "$ROOT/cpp"
rm -rf build
mkdir -p build
cd build
cmake .. -DPython_EXECUTABLE="$PYTHON" -DCMAKE_CXX_FLAGS="$CXXOPT"
cmake --build . -j"$(nproc 2>/dev/null || echo 2)"
SO="$(find . -maxdepth 1 -name 'xqwlight_core*.so' | head -1)"
GEN="$(find . -maxdepth 1 -name 'xqwl_gen_nnue' -type f | head -1)"
if [[ -z "$SO" ]]; then
  echo "[build] xqwlight_core*.so not found" >&2
  exit 1
fi
cp -f "$SO" "$ROOT/"
echo "[build] Copied $(basename "$SO") -> $ROOT/"
if [[ -n "$GEN" ]]; then
  cp -f "$GEN" "$ROOT/"
  echo "[build] Copied $(basename "$GEN") -> $ROOT/"
fi
cd "$ROOT"
if [[ ! -f mycchess_sf/static/xqwl/board.png ]]; then
  echo "[build] Fetching XQWL UI assets..."
  "$PYTHON" scripts/fetch_xqwl_assets.py
fi
"$PYTHON" -m pip install -q -e .
echo "[build] Done. Run: mycchess-play-web --host 0.0.0.0 --port 5151"
if [[ -n "$GEN" ]]; then
  echo "[build] NNUE data gen: ./xqwl_gen_nnue --output-dir nnue_data"
fi
