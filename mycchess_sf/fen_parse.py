"""Chinese chess FEN parsing (no external chess library)."""
from __future__ import annotations

import numpy as np

FULL_INIT_FEN = "rnbakabnr/9/1c5c1/p1p1p1p1p/9/9/P1P1P1P1P/1C5C1/9/RNBAKABNR w - - 0 1"


def parse_fen_board(fen: str) -> tuple[np.ndarray, bool]:
    """
    Return ``(boardarr, red_to_move)``.
    ``boardarr`` is shape ``(10, 9)``, ``dtype='<U1'``; row 0 is black's back rank.
    """

    parts = fen.strip().split()
    rows = parts[0].split("/")
    if len(rows) != 10:
        raise ValueError(f"Expected 10 FEN ranks, got {len(rows)}: {fen!r}")
    board = np.full((10, 9), "", dtype="<U1")
    for y, row in enumerate(rows):
        x = 0
        for ch in row:
            if ch.isdigit():
                x += int(ch)
            else:
                if x >= 9:
                    raise ValueError(f"FEN rank overflow: {row!r}")
                board[y, x] = ch
                x += 1
        if x != 9:
            raise ValueError(f"FEN rank width != 9: {row!r}")
    stm = parts[1] if len(parts) > 1 else "w"
    red_to_move = stm != "b"
    return board, red_to_move
