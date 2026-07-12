#!/usr/bin/env python3
"""Ablation: opening move ranking — PST vs NNUE static eval vs 1-ply search."""
from __future__ import annotations

import argparse
import sys
from dataclasses import dataclass
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))


@dataclass
class MoveScore:
    iccs: str
    pst_after: int
    nnue_after: int
    search_score: int
    search_depth: int


def _load_engine(nnue_path: Path):
    import xqwlight_core as xc

    engine = xc.Engine()
    if not engine.load_nnue(str(nnue_path)):
        raise RuntimeError(f"load_nnue failed: {nnue_path}")
    return engine


def _rank_moves(
    pos,
    engine,
    *,
    time_ms: int,
    top_n: int,
) -> list[MoveScore]:
    moves = list(pos.legal_moves_iccs())
    rows: list[MoveScore] = []

    pst_root = int(pos.evaluate())
    nnue_root = int(engine.evaluate_nnue_raw(pos))

    for mv in moves:
        trial = pos.copy()
        if not trial.make_move_iccs(mv):
            continue
        pst = int(trial.evaluate())
        nnue = int(engine.evaluate_nnue_raw(trial))
        detail = engine.search_best_detail(trial, time_ms, False)
        rows.append(
            MoveScore(
                iccs=mv,
                pst_after=pst,
                nnue_after=nnue,
                search_score=int(detail.get("score", 0)),
                search_depth=int(detail.get("depth", 0)),
            )
        )

    rows.sort(key=lambda r: r.nnue_after, reverse=True)
    return rows[:top_n], pst_root, nnue_root, len(moves)


def _print_table(title: str, rows: list[MoveScore], *, sort_key: str) -> None:
    ordered = sorted(rows, key=lambda r: getattr(r, sort_key), reverse=True)
    print(f"\n=== {title} (top {len(ordered)}) ===")
    print(f"{'move':<10} {'pst':>8} {'nnue':>8} {'search':>8} {'depth':>5}")
    for r in ordered:
        print(
            f"{r.iccs:<10} {r.pst_after:>8} {r.nnue_after:>8} "
            f"{r.search_score:>8} {r.search_depth:>5}"
        )


def main() -> None:
    parser = argparse.ArgumentParser(description="Opening step-1 NNUE ablation")
    parser.add_argument(
        "--nnue",
        type=Path,
        default=ROOT / "data" / "nnue_model" / "quantized.xqnnue.bin",
    )
    parser.add_argument("--time-ms", type=int, default=100)
    parser.add_argument("--top", type=int, default=15)
    parser.add_argument(
        "--fen",
        type=str,
        default="",
        help="Optional FEN; default standard start (red to move).",
    )
    args = parser.parse_args()

    import xqwlight_core as xc

    if not args.nnue.is_file():
        print(f"NNUE not found: {args.nnue}")
        sys.exit(1)

    engine = _load_engine(args.nnue)
    pos = xc.Position()
    if args.fen:
        if not pos.set_fen(args.fen):
            print(f"Invalid FEN: {args.fen}")
            sys.exit(1)

    fen = pos.fen()
    print(f"[pos] {fen}")
    print(f"[nnue] {args.nnue}")
    print(f"[simd] {engine.nnue_simd_backend()}")
    print(f"[drift] incremental verify={engine.verify_nnue_incremental(pos)}")

    best_nnue = engine.search_best_detail(pos, args.time_ms, False)
    engine.clear_nnue()
    best_pst = engine.search_best_detail(pos, args.time_ms, False)
    engine.load_nnue(str(args.nnue))

    print(f"\n[search best @ {args.time_ms}ms, no book]")
    print(
        f"  NNUE on : iccs={best_nnue.get('iccs')!r} "
        f"score={best_nnue.get('score')} depth={best_nnue.get('depth')}"
    )
    print(
        f"  NNUE off: iccs={best_pst.get('iccs')!r} "
        f"score={best_pst.get('score')} depth={best_pst.get('depth')}"
    )

    top_nnue, pst_root, nnue_root, n_moves = _rank_moves(
        pos, engine, time_ms=args.time_ms, top_n=max(args.top, 20)
    )
    print(f"\n[root] legal_moves={n_moves}  PST={pst_root}  NNUE={nnue_root}")

    all_rows, _, _, _ = _rank_moves(pos, engine, time_ms=args.time_ms, top_n=9999)
    _print_table("sorted by NNUE static (after move)", all_rows[: args.top], sort_key="nnue_after")
    _print_table("sorted by PST static (after move)", all_rows[: args.top], sort_key="pst_after")

    nnue_best = max(all_rows, key=lambda r: r.nnue_after)
    pst_best = max(all_rows, key=lambda r: r.pst_after)
    print(f"\n[static argmax] NNUE -> {nnue_best.iccs} ({nnue_best.nnue_after})")
    print(f"[static argmax] PST  -> {pst_best.iccs} ({pst_best.pst_after})")

    if nnue_best.iccs != pst_best.iccs:
        print("[verdict] NNUE and PST disagree on best 1-ply static move.")
    if best_nnue.get("iccs") == nnue_best.iccs:
        print("[verdict] NNUE search picks same move as NNUE static argmax (shallow search).")


if __name__ == "__main__":
    main()
