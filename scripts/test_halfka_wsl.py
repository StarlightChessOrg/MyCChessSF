#!/usr/bin/env python3
"""Validate HalfKAv2_hm Python port against Pikafish reference dump (WSL)."""
from __future__ import annotations

import argparse
import random
import re
import subprocess
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
TRAIN = ROOT / "nnue_training"
if str(TRAIN) not in sys.path:
    sys.path.insert(0, str(TRAIN))

from features.half_ka_v2_hm import (  # noqa: E402
    N_FEATURES,
    PS_NB,
    fen_to_feature_indices_both_perspectives,
)
from mycchess_sf.fen_parse import FULL_INIT_FEN  # noqa: E402

DUMP = ROOT / "third_party" / "halfka_dump"
BUILD_SH = ROOT / "scripts" / "build_pikafish_halfka_dump.sh"

PERSP_RE = re.compile(
    r"perspective=(?P<persp>[wb])\n"
    r"bucket=(?P<bucket>\d+) mirror=(?P<mirror>\d+) attack_bucket=(?P<attack>\d+) "
    r"mid_mirror=(?P<mid>\d+)\n"
    r"king=(?P<king>\d+) oking=(?P<oking>\d+)\n"
    r"indices:(?P<indices>[^\n]*)\n",
    re.MULTILINE,
)


def ensure_dump() -> Path:
    if DUMP.is_file():
        return DUMP
    print(f"[halfka] building reference dump via {BUILD_SH} ...")
    subprocess.run(["bash", str(BUILD_SH)], check=True, cwd=ROOT)
    if not DUMP.is_file():
        raise FileNotFoundError(f"halfka_dump not found at {DUMP}")
    return DUMP


def pikafish_dump(fen: str, dump_bin: Path) -> dict[str, dict]:
    proc = subprocess.run(
        [str(dump_bin)],
        input=fen + "\n",
        capture_output=True,
        text=True,
        check=True,
    )
    block = proc.stdout
    if "ERROR" in block:
        raise RuntimeError(block.strip())
    out: dict[str, dict] = {}
    for m in PERSP_RE.finditer(block):
        persp = m.group("persp")
        indices = [int(x) for x in m.group("indices").split()] if m.group("indices").strip() else []
        out[persp] = {
            "bucket": int(m.group("bucket")),
            "mirror": int(m.group("mirror")),
            "attack_bucket": int(m.group("attack")),
            "mid_mirror": int(m.group("mid")),
            "king": int(m.group("king")),
            "oking": int(m.group("oking")),
            "indices": indices,
        }
    return out


def load_random_fens(path: Path, n: int, seed: int) -> list[str]:
    rng = random.Random(seed)
    if not path.is_file():
        return []
    lines: list[str] = []
    with open(path, encoding="utf-8", errors="replace") as f:
        for line in f:
            parts = line.strip().split("\t")
            if len(parts) == 2:
                lines.append(parts[0])
    if not lines:
        return []
    rng.shuffle(lines)
    return lines[:n]


def compare_fen(fen: str, dump_bin: Path) -> tuple[bool, str]:
    ref = pikafish_dump(fen, dump_bin)
    stm_py, other_py = fen_to_feature_indices_both_perspectives(fen)
    stm_side = "w" if fen.split()[1] == "w" else "b"
    other_side = "b" if stm_side == "w" else "w"

    stm_ref = ref.get(stm_side)
    other_ref = ref.get(other_side)
    if stm_ref is None or other_ref is None:
        return False, "missing perspective in Pikafish dump"

    stm_py_list = sorted(int(x) for x in stm_py)
    other_py_list = sorted(int(x) for x in other_py)
    stm_ref_list = sorted(stm_ref["indices"])
    other_ref_list = sorted(other_ref["indices"])
    ok = True
    msgs: list[str] = []

    if stm_py_list != stm_ref_list:
        ok = False
        msgs.append(
            f"STM mismatch: py={len(stm_py_list)} ref={len(stm_ref_list)} "
            f"delta={set(stm_py_list) ^ set(stm_ref_list)}"
        )
    if other_py_list != other_ref_list:
        ok = False
        msgs.append(
            f"other mismatch: py={len(other_py_list)} ref={len(other_ref_list)} "
            f"delta={set(other_py_list) ^ set(other_ref_list)}"
        )
    return ok, "; ".join(msgs) if msgs else "ok"


def self_tests() -> None:
    stm, other = fen_to_feature_indices_both_perspectives(FULL_INIT_FEN)
    assert len(stm) == 32
    assert len(other) == 32
    assert stm.max() < N_FEATURES
    assert other.max() < N_FEATURES
    assert len(set(stm)) == len(stm)
    assert PS_NB == 689


def main() -> None:
    parser = argparse.ArgumentParser(description="HalfKAv2_hm parity tests (Python vs Pikafish)")
    parser.add_argument("--skip-pikafish", action="store_true", help="Only run Python self-tests")
    parser.add_argument("--samples", type=int, default=200)
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()

    print("[halfka] self-tests ...")
    self_tests()
    print(f"  startpos stm/other = 32 features each, N_FEATURES={N_FEATURES}")

    if args.skip_pikafish:
        print("PASS (self-tests only)")
        return

    dump_bin = ensure_dump()
    fens = [FULL_INIT_FEN]
    data_path = ROOT.parent / "nnue_data" / "merged.txt"
    if not data_path.is_file():
        data_path = ROOT / "nnue_data" / "worker_0" / "chunk_0.txt"
    fens.extend(load_random_fens(data_path, args.samples, args.seed))

    failed = 0
    checked = 0
    for fen in fens:
        ok, msg = compare_fen(fen, dump_bin)
        checked += 1
        if not ok:
            failed += 1
            print(f"FAIL {fen[:72]}... :: {msg}")

    print(f"\n[halfka] checked={checked} failed={failed}")
    if failed:
        sys.exit(1)
    print("PASS: Python HalfKAv2_hm matches Pikafish reference")


if __name__ == "__main__":
    main()
