"""Rule backend: ``xqwlight_core.Position`` plus FEN-derived board view."""
from __future__ import annotations

import numpy as np

from mycchess_sf.fen_parse import parse_fen_board
from mycchess_sf.iccs_util import parse_move_squares

# RepValue drawish branch magnitude; BAN_VALUE separates drawish vs ban-loss
REP_RULE_VALUE_DRAWISH_ABS = 50


def _require_xqwlight():
    try:
        from xqwlight_core import Position as _P  # noqa: F401

        return _P
    except ImportError as e:
        raise ImportError(
            "MyCChessSF requires a built ``xqwlight_core`` extension (see README / CMake)."
        ) from e


class XqwlGameState:
    """Wraps ``xqwlight_core.Position``; ``board_view`` has red on the bottom row."""

    __slots__ = ("pos", "_last_move_iccs")

    def __init__(self) -> None:
        P = _require_xqwlight()
        self.pos = P()
        self._last_move_iccs: str | None = None

    def reset(self) -> None:
        self.pos.reset()
        self._last_move_iccs = None

    @property
    def last_move_iccs(self) -> str | None:
        return self._last_move_iccs

    @property
    def red_to_move(self) -> bool:
        return int(self.pos.side_to_move()) == 0

    def get_side(self) -> str:
        return "red" if self.red_to_move else "black"

    def legal_moves_iccs_str(self) -> list[str]:
        return list(self.pos.legal_moves_iccs())

    def make_move_iccs(self, mv: str) -> bool:
        ok = bool(self.pos.make_move_iccs(mv))
        if ok:
            self._last_move_iccs = mv
        return ok

    def fen(self) -> str:
        return self.pos.fen()

    def board_view(self) -> np.ndarray:
        board, _ = parse_fen_board(self.pos.fen())
        # FEN row 0 = black (top), row 9 = red (bottom); no extra flip.
        return np.array(board, dtype="<U1", copy=True)

    def terminal(self) -> tuple[bool, str]:
        k = int(self.pos.terminal_kind())
        if k == 0:
            return False, ""
        if k == 1:
            return True, "checkmate"
        if k == 2:
            return True, "repetition_rule"
        if k == 3:
            return True, "move_limit_draw"
        return True, "unknown"

    def rep_value_if_any(self) -> int:
        return int(self.pos.rep_value_if_any())

    def copy(self) -> XqwlGameState:
        o = object.__new__(XqwlGameState)
        o.pos = self.pos.copy()
        o._last_move_iccs = self._last_move_iccs
        return o

    def raw_position(self):
        """Underlying ``xqwlight_core.Position`` for ``Engine.search_best_iccs``."""
        return self.pos
