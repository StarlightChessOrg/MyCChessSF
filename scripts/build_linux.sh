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
DEPLOY_MODEL="$DEPLOY/model"
NNUE_SRC="$ROOT/data/nnue_model"

# WSL + DrvFS (/mnt/c, …): Windows host clock can be ahead of Linux, so CMake-
# generated Makefiles look "in the future" and gmake warns about clock skew.
sync_build_timestamps() {
  local dir="${1:-.}"
  [[ -d "$dir" ]] || return 0
  find "$dir" -type f -exec touch {} + 2>/dev/null || true
}

# Windows checkouts may store CRLF; strip before deployment launchers run under WSL.
strip_crlf() {
  local f
  for f in "$@"; do
    [[ -f "$f" ]] || continue
    sed -i 's/\r$//' "$f" 2>/dev/null || sed -i '' 's/\r$//' "$f"
  done
}

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
case "$ROOT" in
  /mnt/*)
    echo "[build] DrvFS mount detected; syncing build timestamps (WSL clock skew)..."
    sync_build_timestamps .
    ;;
esac
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
mkdir -p "$DEPLOY_BIN" "$DEPLOY_LIB" "$DEPLOY_DB" "$DEPLOY_MODEL"
cp -f "$ROOT/$SO_NAME" "$DEPLOY_LIB/"
rm -rf "$DEPLOY_LIB/mycchess_sf"
cp -a "$ROOT/mycchess_sf" "$DEPLOY_LIB/mycchess_sf"
find "$DEPLOY_LIB/mycchess_sf" -type d -name __pycache__ -exec rm -rf {} + 2>/dev/null || true
cp -f scripts/deployment_play_web.sh "$DEPLOY_BIN/mycchess-play-web"
strip_crlf "$DEPLOY_BIN/mycchess-play-web"
chmod +x "$DEPLOY_BIN/mycchess-play-web"
cp -f scripts/deployment_requirements.txt "$DEPLOY/requirements.txt"
if [[ -n "$GEN_NAME" ]]; then
  cp -f "$ROOT/$GEN_NAME" "$DEPLOY_BIN/"
  chmod +x "$DEPLOY_BIN/$GEN_NAME"
fi
if [[ ! -f data/compressed_files/book.7z ]]; then
  echo "[build] data/compressed_files/book.7z not found" >&2
  exit 1
fi
"$PYTHON" scripts/extract_book.py --archive data/compressed_files/book.7z --output-dir "$DEPLOY_DB"
if compgen -G "$NNUE_SRC/*.xqnnue.bin" > /dev/null; then
  cp -f "$NNUE_SRC"/*.xqnnue.bin "$DEPLOY_MODEL/"
  echo "[build]   model/ -> $(basename -a "$NNUE_SRC"/*.xqnnue.bin | tr '\n' ' ')"
else
  echo "[build]   model/ -> (empty; add data/nnue_model/*.xqnnue.bin for NNUE eval)"
fi
cp -f scripts/deployment_README.md "$DEPLOY/README.md"

"$PYTHON" -m pip install -q -e .
echo "[build] Done."
echo "[build] Deployment: $DEPLOY"
echo "[build]   lib/  -> $SO_NAME + mycchess_sf/ (网页对弈 Python 包)"
if [[ -n "$GEN_NAME" ]]; then
  echo "[build]   bin/  -> $GEN_NAME, mycchess-play-web"
else
  echo "[build]   bin/  -> mycchess-play-web"
fi
echo "[build]   db/   -> BOOK.DAT"
if compgen -G "$DEPLOY_MODEL/*.xqnnue.bin" > /dev/null; then
  echo "[build]   model/ -> NNUE weights"
fi
echo "[build] Web play: pip install -r $DEPLOY/requirements.txt && $DEPLOY_BIN/mycchess-play-web --host 0.0.0.0 --port 5151"
if [[ -n "$GEN_NAME" ]]; then
  echo "[build] NNUE data gen: $DEPLOY_BIN/xqwl_gen_nnue  (loads $DEPLOY_MODEL/quantized.xqnnue.bin)"
fi
