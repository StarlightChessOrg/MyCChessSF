#!/usr/bin/env python3
from __future__ import annotations

import sys
from pathlib import Path

import torch

ROOT = Path(__file__).resolve().parents[1]
for p in (ROOT, ROOT / "nnue_training", ROOT / "nnue_qINT8"):
    sys.path.insert(0, str(p))

import xqwlight_core as xc
from features.xqwl_psq import fen_to_feature_indices
from int8_nnue import QuantizedNNUE
from model.nnue import NNUE
from mycchess_sf.fen_parse import FULL_INIT_FEN

ckpt_path = ROOT / "nnue_training/checkpoints/full_gpu/best.pt"
bin_path = ROOT / "data/nnue_model/quantized.xqnnue.bin"
pt_path = ROOT / "data/nnue_model/quantized.xqint8.pt"
data_path = ROOT / "nnue_data/merged.txt"

ckpt = torch.load(ckpt_path, map_location="cpu", weights_only=False)
cfg = ckpt["config"]["model"]
float_m = NNUE(l1=cfg["l1"], l2=cfg["l2"], l3=cfg["l3"])
float_m.load_state_dict(ckpt["model_state_dict"])
float_m.eval()
mean, std = float(ckpt["label_mean"]), float(ckpt["label_std"])
q = QuantizedNNUE.load(str(pt_path))

eng = xc.Engine()
eng.load_nnue(str(bin_path))
pos = xc.Position()
fen = pos.fen()
idx = torch.tensor(fen_to_feature_indices(fen), dtype=torch.long)
off = torch.tensor([0], dtype=torch.long)
with torch.no_grad():
    fvl = float(float_m(idx, off).item()) * std + mean
    qvl = float(q.forward_vl_int8(idx, off).item())

print("checkpoint label_mean/std:", mean, std)
print("fen:", fen)
print("PST:", pos.evaluate())
print("float NNUE:", round(fvl, 2))
print("py int8:", round(qvl, 2))
print("cpp int8:", eng.evaluate_nnue_raw(pos))

key = fen.split(" w ")[0] + " w"
vals: list[float] = []
if data_path.is_file():
    with open(data_path, encoding="utf-8") as fp:
        for line in fp:
            if line.startswith(key):
                vals.append(float(line.strip().split("\t")[1]))
    print(f"merged.txt rows with same board+side: {len(vals):,}")
    if vals:
        import statistics as st

        print(f"  vl min={min(vals):.0f} max={max(vals):.0f} mean={st.mean(vals):.2f} median={st.median(vals):.2f}")

moves = ["22-21", "17-14", "77-73", "79-67", "87-85", "72-82", "69-47", "49-48"]
print("\nmove       PST   NNUE")
for mv in moves:
    t = pos.copy()
    if not t.make_move_iccs(mv):
        print(mv, "illegal")
        continue
    print(f"{mv:<10} {t.evaluate():>4} {eng.evaluate_nnue_raw(t):>5}")
