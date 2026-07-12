#!/usr/bin/env bash
# Augment workspace merged.txt with PST/in_check and install under MyCChessSF/nnue_data/.
if [ -z "${BASH_VERSION:-}" ]; then
  exec /usr/bin/env bash "$0" "$@"
fi
set -euo pipefail

ROOT="$(cd "$(dirname "$0")/.." && pwd)"
cd "$ROOT"

if [[ ! -f scripts/build_linux.sh ]]; then
  echo "Run from MyCChessSF repo root." >&2
  exit 1
fi

if ! python3 -c "import xqwlight_core" 2>/dev/null; then
  if ls "$ROOT"/cpp/build/xqwlight_core*.so >/dev/null 2>&1; then
    cp -f "$ROOT"/cpp/build/xqwlight_core*.so "$ROOT/"
  else
    echo "[augment] building xqwlight_core ..."
    PYTHON="${PYTHON:-python3}" bash scripts/build_linux.sh || {
      echo "[augment] build failed; try: python3 -m pip install --break-system-packages pybind11" >&2
      exit 1
    }
  fi
fi

INPUT="${1:?usage: $0 INPUT [OUTPUT]}"
OUTPUT="${2:-$ROOT/nnue_data/merged.txt}"

mkdir -p "$(dirname "$OUTPUT")"

echo "[augment] input=$INPUT"
echo "[augment] output=$OUTPUT"

python3 scripts/augment_nnue_data_pst.py --input "$INPUT" --output "$OUTPUT" --workers auto

echo "[augment] verify head:"
head -n 3 "$OUTPUT"
