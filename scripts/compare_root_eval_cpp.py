#!/usr/bin/env python3
from __future__ import annotations

import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

import xqwlight_core as xc

bin_path = ROOT / "data/nnue_model/quantized.xqnnue.bin"
eng = xc.Engine()
eng.load_nnue(str(bin_path))
pos = xc.Position()
fen = pos.fen()
print("fen:", fen)
print("PST:", pos.evaluate())
print("cpp NNUE:", eng.evaluate_nnue_raw(pos))

moves = ["22-21", "17-14", "77-73", "79-67", "87-85", "72-82", "69-47", "49-48"]
print("\nmove       PST   NNUE")
for mv in moves:
    t = pos.copy()
    if not t.make_move_iccs(mv):
        print(mv, "illegal")
        continue
    print(f"{mv:<10} {t.evaluate():>4} {eng.evaluate_nnue_raw(t):>5}")

best = eng.search_best_detail(pos, 100, False)
eng.clear_nnue()
best_pst = eng.search_best_detail(pos, 100, False)
print("\nsearch NNUE on:", best)
print("search PST only:", best_pst)
