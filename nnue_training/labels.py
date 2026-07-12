"""XQWL label constants and mate-zone helpers (aligned with cpp/include/xqwl_portable_prefix.h)."""
from __future__ import annotations

MATE_VALUE = 10_000
WIN_VALUE = MATE_VALUE - 200  # 9800
DEFAULT_MATE_THRESHOLD = WIN_VALUE


def is_mate_label(vl: float, *, threshold: float = DEFAULT_MATE_THRESHOLD) -> bool:
    """True when |vl| is in the mate / win band (search returns ±9998-style scores here)."""
    return abs(vl) >= threshold
