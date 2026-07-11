"""唯一规则后端：``xqwlight_core.Position`` + 从 FEN 派生的观测字段。"""
from __future__ import annotations

import numpy as np

from mycchess_sf.fen_parse import parse_fen_board
from mycchess_sf.iccs_util import parse_move_squares

REP_RULE_VALUE_DRAWISH_ABS = 50


def _require_xqwlight():
    try:
        from xqwlight_core import Position as _P  # noqa: F401

        return _P
    except ImportError as e:
        raise ImportError(
            "MyCChessSF 需要已编译的 ``xqwlight_core`` 扩展（见 README 的 CMake 说明）。"
            "规则与搜索均来自象棋小巫师 XQWL06。"
        ) from e


class XqwlGameState:
    """封装 ``xqwlight_core.Position``；``board_view`` 红方在下。"""

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
        flipped = np.flip(np.asarray(board), axis=0)
        return np.array(flipped, dtype="<U1", copy=True)

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
        """底层 ``xqwlight_core.Position``，供 ``Engine.search_best_iccs`` 使用。"""
        return self.pos
