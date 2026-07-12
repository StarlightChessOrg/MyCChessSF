#!/usr/bin/env bash
# Build deployment/ and pack dist/MyCChessSF-<version>-linux-x86_64-py<py>.tar.gz + .sha256
set -euo pipefail
ROOT="$(cd "$(dirname "$0")/.." && pwd)"
VERSION="${1:-0.1.0}"
PYTHON="${PYTHON:-$(command -v python3)}"
PYTAG="$("$PYTHON" -c 'import sys; print(f"{sys.version_info.major}.{sys.version_info.minor}")')"
DIST="$ROOT/dist"
NAME="MyCChessSF-v${VERSION}-linux-x86_64-py${PYTAG}"
ARCHIVE="$DIST/${NAME}.tar.gz"
CHECKSUM="$ARCHIVE.sha256"

if [[ "${SKIP_BUILD:-0}" != "1" ]]; then
  echo "[release] Building deployment/ ..."
  (cd "$ROOT" && PYTHON="$PYTHON" bash scripts/build_linux.sh)
else
  echo "[release] SKIP_BUILD=1, using existing deployment/"
fi

if [[ ! -d "$ROOT/deployment/lib" ]]; then
  echo "[release] deployment/ missing; run build_linux.sh first" >&2
  exit 1
fi

mkdir -p "$DIST"
rm -f "$ARCHIVE" "$CHECKSUM"
tar -C "$ROOT" -czf "$ARCHIVE" deployment
sha256sum "$ARCHIVE" > "$CHECKSUM"

echo "[release] Archive: $ARCHIVE"
echo "[release] Size:    $(du -h "$ARCHIVE" | cut -f1)"
echo "[release] SHA256:  $(cut -d' ' -f1 "$CHECKSUM")"
echo "[release] Python:  $("$PYTHON" --version 2>&1)"
echo "[release] .so:     $(basename "$ROOT"/deployment/lib/xqwlight_core*.so)"
