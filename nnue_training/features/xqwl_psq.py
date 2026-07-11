"""XQWL-PSQ sparse features (no king bucketing)."""
from __future__ import annotations

import sys
from pathlib import Path

import numpy as np

_ROOT = Path(__file__).resolve().parents[2]
if str(_ROOT) not in sys.path:
    sys.path.insert(0, str(_ROOT))

from mycchess_sf.fen_parse import parse_fen_board

N_SQUARES = 90
N_KINDS = 7
N_FEATURES = N_KINDS * N_SQUARES * 2  # friendly + enemy

PIECE_KIND: dict[str, int] = {
    "K": 0,
    "A": 1,
    "B": 2,
    "N": 3,
    "R": 4,
    "C": 5,
    "P": 6,
    "k": 0,
    "a": 1,
    "b": 2,
    "n": 3,
    "r": 4,
    "c": 5,
    "p": 6,
}

FRIENDLY_OFFSET = 0
ENEMY_OFFSET = N_KINDS * N_SQUARES


def _square_index(y: int, x: int) -> int:
    return y * 9 + x


def _feature_index(kind: int, sq: int, *, friendly: bool) -> int:
    base = FRIENDLY_OFFSET if friendly else ENEMY_OFFSET
    return base + kind * N_SQUARES + sq


def fen_to_feature_indices(fen: str) -> np.ndarray:
    """
    Return active feature indices for side-to-move perspective.

    Encoding: 7 piece kinds × 90 squares for friendly pieces, same for enemy.
    When black to move the board is vertically flipped so friendly is always
    at the bottom from the mover's view.
    """
    board, red_to_move = parse_fen_board(fen)
    indices: list[int] = []

    for y in range(10):
        for x in range(9):
            ch = board[y, x]
            if not ch:
                continue
            if red_to_move:
                view_y = y
                friendly = ch.isupper()
            else:
                view_y = 9 - y
                friendly = ch.islower()
            kind = PIECE_KIND[ch]
            sq = _square_index(view_y, x)
            indices.append(_feature_index(kind, sq, friendly=friendly))

    return np.array(indices, dtype=np.int64)
