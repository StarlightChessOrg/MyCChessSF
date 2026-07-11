#!/usr/bin/env python3
"""Fetch XQWL Win32 RES assets from xqbase/xqwlight and prepare web copies."""
from __future__ import annotations

import shutil
import urllib.request
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
STATIC = ROOT / "mycchess_sf" / "static" / "xqwl"
DATA = ROOT / "data"
BASE = "https://raw.githubusercontent.com/xqbase/xqwlight/master/Win32"

BMPS = [
    "BOARD.BMP",
    "SELECTED.BMP",
    "RK.BMP",
    "RA.BMP",
    "RB.BMP",
    "RN.BMP",
    "RR.BMP",
    "RC.BMP",
    "RP.BMP",
    "BK.BMP",
    "BA.BMP",
    "BB.BMP",
    "BN.BMP",
    "BR.BMP",
    "BC.BMP",
    "BP.BMP",
]
WAVS = [
    "CLICK.WAV",
    "ILLEGAL.WAV",
    "MOVE.WAV",
    "MOVE2.WAV",
    "CAPTURE.WAV",
    "CAPTURE2.WAV",
    "CHECK.WAV",
    "CHECK2.WAV",
    "WIN.WAV",
    "DRAW.WAV",
    "LOSS.WAV",
]
MASK_RGB = (0, 255, 0)


def _download(url: str, dest: Path) -> None:
    dest.parent.mkdir(parents=True, exist_ok=True)
    if dest.exists() and dest.stat().st_size > 0:
        return
    print(f"[fetch] {dest.name}")
    urllib.request.urlretrieve(url, dest)


def _bmp_to_png(src: Path, dest: Path, *, chroma: bool) -> None:
    from PIL import Image

    img = Image.open(src).convert("RGBA")
    if chroma:
        px = img.load()
        w, h = img.size
        for y in range(h):
            for x in range(w):
                r, g, b, a = px[x, y]
                if (r, g, b) == MASK_RGB:
                    px[x, y] = (r, g, b, 0)
    dest.parent.mkdir(parents=True, exist_ok=True)
    img.save(dest, optimize=True)


def main() -> None:
    tmp = ROOT / "scripts" / "_xqwl_res_cache"
    tmp.mkdir(parents=True, exist_ok=True)
    STATIC.mkdir(parents=True, exist_ok=True)
    DATA.mkdir(parents=True, exist_ok=True)

    for name in BMPS:
        _download(f"{BASE}/RES/{name}", tmp / name)
    for name in WAVS:
        _download(f"{BASE}/RES/{name}", tmp / name)
    _download(f"{BASE}/BOOK.DAT", DATA / "BOOK.DAT")

    for name in BMPS:
        stem = name[:-4].lower()
        chroma = name != "BOARD.BMP"
        _bmp_to_png(tmp / name, STATIC / f"{stem}.png", chroma=chroma)

    for name in WAVS:
        shutil.copy2(tmp / name, STATIC / name.lower())

    print(f"[fetch] assets -> {STATIC}")
    print(f"[fetch] BOOK.DAT -> {DATA / 'BOOK.DAT'}")


if __name__ == "__main__":
    main()
