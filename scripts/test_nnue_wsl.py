#!/usr/bin/env python3
"""Compare C++ NNUE eval (xqwlight_core) vs Python QuantizedNNUE."""
from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import torch

ROOT = Path(__file__).resolve().parents[1]
TRAIN = ROOT / "nnue_training"
QINT8 = ROOT / "nnue_qINT8"

for p in (ROOT, TRAIN, QINT8):
    if str(p) not in sys.path:
        sys.path.insert(0, str(p))

import xqwlight_core as xc
from data.dataset import _parse_line
from features.xqwl_psq import fen_to_feature_indices
from int8_nnue import QuantizedNNUE


def py_nnue_vl(model: QuantizedNNUE, fen: str) -> float:
    idx = fen_to_feature_indices(fen)
    indices = torch.tensor(idx, dtype=torch.long)
    offsets = torch.tensor([0], dtype=torch.long)
    return float(model.forward_vl(indices, offsets).item())


def load_samples(path: Path, n: int) -> list[tuple[str, float]]:
    out: list[tuple[str, float]] = []
    with open(path, encoding="utf-8", errors="replace") as f:
        for line in f:
            s = _parse_line(line)
            if s is not None:
                out.append((s.fen, s.vl))
                if len(out) >= n:
                    break
    return out


def main() -> None:
    bin_path = QINT8 / "output" / "quantized.xqnnue.bin"
    pt_path = QINT8 / "output" / "quantized.xqint8.pt"
    data_path = ROOT / "nnue_data" / "worker_0.txt"

    py_model = QuantizedNNUE.load(str(pt_path))
    engine = xc.Engine()
    if not engine.load_nnue(str(bin_path)):
        print("load_nnue failed")
        sys.exit(1)

    pos = xc.Position()
    py_vl = py_nnue_vl(py_model, pos.fen())
    cpp_vl = engine.evaluate_nnue_raw(pos)
    print("[startpos]")
    print(f"  PST={pos.evaluate()}  C++ raw={cpp_vl}  Py NNUE={py_vl:.2f}  |diff|={abs(cpp_vl - py_vl):.2f}")
    inc_drift = engine.verify_nnue_incremental(pos)
    print(f"  incremental drift (root+1ply): {inc_drift}")

    samples = load_samples(data_path, 500)
    diffs: list[float] = []
    max_diff = 0.0
    worst = ""
    skipped = 0
    for fen, _vl in samples:
        if not pos.set_fen(fen):
            skipped += 1
            continue
        cpp = engine.evaluate_nnue_raw(pos)
        py = py_nnue_vl(py_model, fen)
        d = abs(cpp - py)
        diffs.append(d)
        if d > max_diff:
            max_diff = d
            worst = fen

    diffs_arr = np.array(diffs, dtype=np.float64)
    print(f"\n[compare] n={len(diffs)} skipped={skipped}")
    print(f"  |diff| mean={diffs_arr.mean():.3f}  max={diffs_arr.max():.3f}  p99={np.percentile(diffs_arr, 99):.3f}")
    print(f"  within 1.0: {(diffs_arr <= 1.0).mean() * 100:.1f}%")
    print(f"  within 5.0: {(diffs_arr <= 5.0).mean() * 100:.1f}%")
    if worst:
        print(f"  worst fen: {worst[:60]}...")

    detail = engine.search_best_detail(pos, 200, False)
    print(f"\n[search+nnue] iccs={detail['iccs']!r} depth={detail['depth']} score={detail['score']}")

    engine.clear_nnue()
    detail2 = engine.search_best_detail(pos, 200, False)
    print(f"[search+pst] iccs={detail2['iccs']!r} depth={detail2['depth']} score={detail2['score']}")

    ok = diffs_arr.max() <= 5.0 and abs(cpp_vl - py_vl) <= 1.0 and inc_drift == 0
    print(f"\n{'PASS' if ok else 'CHECK'}: C++ NNUE matches Python reference")
    sys.exit(0 if ok else 1)


if __name__ == "__main__":
    main()
