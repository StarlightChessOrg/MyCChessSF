#!/usr/bin/env python3
"""Compare C++ HalfKAv2 feature indices (via xqwlight_core) vs Python."""
from __future__ import annotations

import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "nnue_training"))

import xqwlight_core as xc
from features.half_ka_v2_hm import fen_to_feature_indices
from mycchess_sf.fen_parse import FULL_INIT_FEN


def cpp_indices(fen: str) -> list[int]:
    pos = xc.Position()
    if not pos.set_fen(fen):
        raise ValueError(f"bad fen: {fen}")
    return sorted(int(x) for x in pos.halfka_feature_indices())


def py_indices(fen: str) -> list[int]:
    return sorted(int(x) for x in fen_to_feature_indices(fen))


def main() -> None:
    fens = [FULL_INIT_FEN]

    failed = 0
    for fen in fens:
        py = py_indices(fen)
        cpp = cpp_indices(fen)
        if py != cpp:
            failed += 1
            print("FAIL", fen[:60])
            print("  py only", set(py) - set(cpp))
            print("  cpp only", set(cpp) - set(py))
        else:
            print(f"OK {len(py)} indices")

    if failed:
        sys.exit(1)
    print("PASS: C++ HalfKA indices match Python")


if __name__ == "__main__":
    main()
