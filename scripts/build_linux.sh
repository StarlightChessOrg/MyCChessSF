#!/usr/bin/env bash
# Build xqwlight_core.so on Linux and install the Python package (run from repo root)
set -euo pipefail
ROOT="$(cd "$(dirname "$0")/.." && pwd)"
PYTHON="${PYTHON:-$(command -v python3)}"

echo "[build] Python: $PYTHON"
"$PYTHON" -m pip install -q -U pip
"$PYTHON" -m pip install -q pybind11 numpy
# pybind11 CMake config must match the same interpreter used for the extension
if ! "$PYTHON" -c "import pybind11; print('pybind11', pybind11.__version__, pybind11.get_cmake_dir())"; then
  echo "[build] Failed to import pybind11 after pip install" >&2
  exit 1
fi

cd "$ROOT/cpp"
rm -rf build
mkdir -p build
cd build
cmake .. -DPython_EXECUTABLE="$PYTHON"
cmake --build . -j"$(nproc 2>/dev/null || echo 2)"
SO="$(find . -maxdepth 1 -name 'xqwlight_core*.so' | head -1)"
if [[ -z "$SO" ]]; then
  echo "[build] xqwlight_core*.so not found" >&2
  exit 1
fi
cp -f "$SO" "$ROOT/"
echo "[build] Copied $(basename "$SO") -> $ROOT/"
cd "$ROOT"
"$PYTHON" -m pip install -q -e .
echo "[build] Done. Run: mycchess-play-web --host 0.0.0.0 --port 5151"
