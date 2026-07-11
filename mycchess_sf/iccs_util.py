"""ICCS move string parsing (e.g. ``7747-7767``)."""


def iccs_y_to_board_view_row(y_iccs: int) -> int:
    """ICCS / engine y (0=black top .. 9=red bottom) equals ``board_view`` row."""
    return int(y_iccs)


def parse_move_squares(move: str) -> tuple[int, int, int, int]:
    if len(move) < 5 or move[2] != "-":
        raise ValueError(f"Invalid move: {move!r}")
    x1, y1, x2, y2 = int(move[0]), int(move[1]), int(move[3]), int(move[4])
    return x1, y1, x2, y2
