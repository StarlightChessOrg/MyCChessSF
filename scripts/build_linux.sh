#!/usr/bin/env bash
# 在 Linux 上编译 xqwlight_core.so 并安装 Python 包（需在仓库根目录执行）
set -euo pipefail
ROOT="$(cd "$(dirname "$0")/.." && pwd)"
cd "$ROOT/cpp"
mkdir -p build
cd build
cmake .. -DPython_EXECUTABLE="$(command -v python3)"
cmake --build . --config Release -j"$(nproc 2>/dev/null || echo 2)"
SO="$(find . -maxdepth 1 -name 'xqwlight_core*.so' | head -1)"
if [[ -z "$SO" ]]; then
  echo "未找到 xqwlight_core*.so" >&2
  exit 1
fi
cp -f "$SO" "$ROOT/"
echo "已复制 $(basename "$SO") -> $ROOT/"
cd "$ROOT"
pip install -e .
echo "完成。运行: mycchess-play-web --host 0.0.0.0 --port 8080"
