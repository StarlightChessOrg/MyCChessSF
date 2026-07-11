#!/usr/bin/env python3
"""Scan nnue_data worker files for malformed / glued lines."""
from __future__ import annotations

import glob
import re
import sys
from collections import Counter
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from mycchess_sf.fen_parse import parse_fen_board

FEN_TAIL = re.compile(r" (w|b) - - \d+ \d+$")


def classify_line(raw: str) -> str | None:
    raw = raw.rstrip("\n\r")
    if not raw.strip():
        return None

    tab_count = raw.count("\t")
    if tab_count == 0:
        return "no_tab"
    if tab_count > 1:
        return "glued_two_records"

    fen, vl_str = raw.split("\t", 1)
    tail_count = len(FEN_TAIL.findall(raw))
    if tail_count >= 2:
        return "multiple_fen_tails_in_line"

    rank_slashes = fen.split(" ", 1)[0].count("/")
    if rank_slashes != 9:
        return "bad_rank_count"

    try:
        float(vl_str)
    except ValueError:
        if " w - - " in vl_str or " b - - " in vl_str:
            return "glued_score_and_next_fen"
        return "bad_score"

    try:
        parse_fen_board(fen)
    except ValueError:
        return "invalid_fen"

    return None


def main() -> None:
    data_dir = ROOT / "nnue_data"
    if len(sys.argv) > 1:
        data_dir = Path(sys.argv[1])
    files = sorted(glob.glob(str(data_dir / "worker_*/chunk_*.txt")))
    if not files:
        print(f"No worker_*/chunk_*.txt under {data_dir}")
        sys.exit(1)

    stats: Counter[str] = Counter()
    examples: dict[str, list[str]] = {}
    total = 0

    for fp in files:
        path = Path(fp)
        with open(path, encoding="utf-8", errors="replace") as f:
            for lineno, line in enumerate(f, 1):
                total += 1
                kind = classify_line(line)
                if kind:
                    stats[kind] += 1
                    if len(examples.get(kind, [])) < 2:
                        examples.setdefault(kind, []).append(
                            f"{path.name}:{lineno}: {line.rstrip()[:180]}"
                        )

    print(f"Scanned {len(files)} files, {total:,} lines")
    bad = sum(stats.values())
    print(f"Bad lines: {bad:,} ({100 * bad / total:.3f}%)")
    for kind, n in stats.most_common():
        print(f"  {kind}: {n}")
        for ex in examples.get(kind, []):
            print(f"    e.g. {ex}")


if __name__ == "__main__":
    main()
