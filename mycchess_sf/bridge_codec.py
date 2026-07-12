"""Chess98 / 相弈 auto-play move encoding (4-digit bridge ↔ ICCS)."""
from __future__ import annotations

from mycchess_sf.iccs_util import parse_move_squares


def iccs_to_move_id(iccs: str) -> int:
    x1, y1, x2, y2 = parse_move_squares(iccs)
    return x1 * 1000 + y1 * 100 + x2 * 10 + y2


def move_id_to_iccs(move_id: int) -> str:
    mid = int(move_id)
    x1 = mid // 1000
    y1 = (mid // 100) % 10
    x2 = (mid // 10) % 10
    y2 = mid % 10
    return f"{x1}{y1}-{x2}{y2}"


def bridge4_to_iccs(code: str) -> str:
    """Decode z.js ``playermove`` (y1x1y2x2) or 5-digit Chess98 move id."""
    digits = "".join(c for c in code.strip() if c.isdigit())
    if not digits:
        raise ValueError(f"Invalid bridge move: {code!r}")
    if len(digits) >= 5:
        return move_id_to_iccs(int(digits[-5:]))
    if len(digits) == 4:
        y1, x1, y2, x2 = (int(digits[i]) for i in range(4))
        return f"{x1}{y1}-{x2}{y2}"
    raise ValueError(f"Invalid bridge move length: {code!r}")


def iccs_to_bridge4(iccs: str) -> str:
    x1, y1, x2, y2 = parse_move_squares(iccs)
    return f"{y1}{x1}{y2}{x2}"


def iccs_to_computer_token(iccs: str) -> str:
    """Token for ``GET /computer`` (4-digit y1x1y2x2, z.js compatible)."""
    return iccs_to_bridge4(iccs)
