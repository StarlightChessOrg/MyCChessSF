"""Human-readable move / think-log descriptions."""
from __future__ import annotations

from mycchess_sf.iccs_util import parse_move_squares

_PIECE_CN: dict[str, str] = {
    "K": "帅",
    "A": "仕",
    "B": "相",
    "N": "马",
    "R": "车",
    "C": "炮",
    "P": "兵",
    "k": "将",
    "a": "士",
    "b": "象",
    "n": "马",
    "r": "车",
    "c": "炮",
    "p": "卒",
}


def piece_cn(ch: str | None) -> str:
    if not ch:
        return "?"
    return _PIECE_CN.get(ch, "?")


def _format_vl(score: int) -> str:
    s = int(score)
    return f"+{s}" if s > 0 else str(s)


def think_log_entry(
    *,
    depth: int,
    elapsed_ms: float,
    piece_ch: str,
    iccs: str,
    from_book: bool = False,
    score: int = 0,
) -> dict[str, object]:
    x1, y1, x2, y2 = parse_move_squares(iccs)
    return {
        "depth": int(depth),
        "elapsed_ms": round(float(elapsed_ms)),
        "score": int(score),
        "piece": piece_cn(piece_ch),
        "x1": x1,
        "y1": y1,
        "x2": x2,
        "y2": y2,
        "iccs": iccs,
        "from_book": bool(from_book),
        "text": format_think_log_text(
            depth=depth,
            elapsed_ms=elapsed_ms,
            piece=piece_cn(piece_ch),
            x1=x1,
            y1=y1,
            x2=x2,
            y2=y2,
            from_book=from_book,
            score=score,
        ),
    }


def format_think_log_text(
    *,
    depth: int,
    elapsed_ms: float,
    piece: str,
    x1: int,
    y1: int,
    x2: int,
    y2: int,
    from_book: bool,
    score: int = 0,
) -> str:
    ms = int(round(elapsed_ms))
    vl = _format_vl(score)
    src = f"({x1},{y1})"
    dst = f"({x2},{y2})"
    if from_book:
        return f"开局库 · {ms}ms · vl {vl} · {piece} · {src} → {dst}"
    return f"深度 {depth} · {ms}ms · vl {vl} · {piece} · {src} → {dst}"
