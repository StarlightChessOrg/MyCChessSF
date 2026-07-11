"""XQWL06 Win32 UI constants and asset paths (ported for web)."""
from __future__ import annotations

from pathlib import Path

# Win32/XQWL06.CPP
SQUARE_SIZE = 56
BOARD_EDGE = 8
BOARD_WIDTH = BOARD_EDGE + SQUARE_SIZE * 9 + BOARD_EDGE
BOARD_HEIGHT = BOARD_EDGE + SQUARE_SIZE * 10 + BOARD_EDGE

STATIC_DIR = Path(__file__).resolve().parent / "static" / "xqwl"

PIECE_SPRITE: dict[str, str] = {
    "K": "rk",
    "A": "ra",
    "B": "rb",
    "N": "rn",
    "R": "rr",
    "C": "rc",
    "P": "rp",
    "k": "bk",
    "a": "ba",
    "b": "bb",
    "n": "bn",
    "r": "br",
    "c": "bc",
    "p": "bp",
}

SOUND_NAMES = (
    "click",
    "illegal",
    "move",
    "move2",
    "capture",
    "capture2",
    "check",
    "check2",
    "win",
    "draw",
    "loss",
)


def assets_available() -> bool:
    return (STATIC_DIR / "board.png").is_file()
