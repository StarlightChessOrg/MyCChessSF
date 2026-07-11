#!/usr/bin/env bash
# Build xqwlight_core.so on Linux, install Python package, and pack deployment/
set -euo pipefail
ROOT="$(cd "$(dirname "$0")/.." && pwd)"
PYTHON="${PYTHON:-$(command -v python3)}"
CXXOPT="${CXXOPT:--O3}"
DEPLOY="$ROOT/deployment"
DEPLOY_BIN="$DEPLOY/bin"
DEPLOY_LIB="$DEPLOY/lib"
DEPLOY_DB="$DEPLOY/db"

echo "[build] Python: $PYTHON"
echo "[build] CXXOPT: $CXXOPT"
"$PYTHON" -m pip install -q -U pip
"$PYTHON" -m pip install -q pybind11 numpy pillow py7zr
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
SO_NAME="$(basename "$SO")"
GEN_NAME=""
if [[ -n "$GEN" ]]; then
  GEN_NAME="$(basename "$GEN")"
fi
if [[ -z "$SO" ]]; then
  echo "[build] xqwlight_core*.so not found" >&2
  exit 1
fi

cd "$ROOT"
cp -f "cpp/build/$SO_NAME" "$ROOT/"
echo "[build] Copied $SO_NAME -> $ROOT/"
if [[ -n "$GEN_NAME" ]]; then
  cp -f "cpp/build/$GEN_NAME" "$ROOT/"
  echo "[build] Copied $GEN_NAME -> $ROOT/"
fi

if [[ ! -f mycchess_sf/static/xqwl/board.png ]]; then
  echo "[build] Fetching XQWL UI assets..."
  "$PYTHON" scripts/fetch_xqwl_assets.py
fi

echo "[build] Packing deployment/ ..."
rm -rf "$DEPLOY"
mkdir -p "$DEPLOY_BIN" "$DEPLOY_LIB" "$DEPLOY_DB"
cp -f "$ROOT/$SO_NAME" "$DEPLOY_LIB/"
if [[ -n "$GEN_NAME" ]]; then
  cp -f "$ROOT/$GEN_NAME" "$DEPLOY_BIN/"
  chmod +x "$DEPLOY_BIN/$GEN_NAME"
fi
if [[ ! -f data/compressed_files/book.7z ]]; then
  echo "[build] data/compressed_files/book.7z not found" >&2
  exit 1
fi
"$PYTHON" scripts/extract_book.py --archive data/compressed_files/book.7z --output-dir "$DEPLOY_DB"
cp -f scripts/deployment_README.md "$DEPLOY/README.md"

"$PYTHON" -m pip install -q -e .
echo "[build] Done."
echo "[build] Deployment: $DEPLOY"
echo "[build]   lib/  -> $SO_NAME (Python 扩展)"
if [[ -n "$GEN_NAME" ]]; then
  echo "[build]   bin/  -> $GEN_NAME"
fi
echo "[build]   db/   -> BOOK.DAT"
echo "[build] Run web: PYTHONPATH=$DEPLOY_LIB mycchess-play-web --book $DEPLOY_DB/BOOK.DAT"
if [[ -n "$GEN_NAME" ]]; then
  echo "[build] NNUE data gen: $DEPLOY_BIN/xqwl_gen_nnue --output-dir nnue_data"
fi
