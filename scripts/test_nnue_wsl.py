#!/usr/bin/env python3
"""Compare C++ INT8 NNUE eval vs Python QuantizedNNUE (float + int8 paths)."""
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


def py_indices(fen: str) -> tuple[torch.Tensor, torch.Tensor]:
    idx = fen_to_feature_indices(fen)
    return torch.tensor(idx, dtype=torch.long), torch.tensor([0], dtype=torch.long)


def py_nnue_vl_float(model: QuantizedNNUE, fen: str) -> float:
    indices, offsets = py_indices(fen)
    return float(model.forward_vl(indices, offsets).item())


def py_nnue_vl_int8(model: QuantizedNNUE, fen: str) -> float:
    indices, offsets = py_indices(fen)
    return float(model.forward_vl_int8(indices, offsets).item())


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

    if not pt_path.is_file():
        print(f"Missing quantized checkpoint: {pt_path}")
        print("Run: cd nnue_qINT8 && python3 quantize.py")
        sys.exit(1)
    if not bin_path.is_file():
        print(f"Missing binary: {bin_path}")
        sys.exit(1)

    py_model = QuantizedNNUE.load(str(pt_path))
    if py_model.fc0.input_scale <= 0.0:
        print("ERROR: FC input_scale unset in checkpoint; re-run quantize.py")
        sys.exit(1)

    engine = xc.Engine()
    if not engine.load_nnue(str(bin_path)):
        print("load_nnue failed")
        sys.exit(1)

    print(f"[simd] C++ backend={engine.nnue_simd_backend()}")
    print(
        f"[scales] fc0={py_model.fc0.input_scale:.6g}  "
        f"fc1={py_model.fc1.input_scale:.6g}  fc2={py_model.fc2.input_scale:.6g}"
    )

    pos = xc.Position()
    fen0 = pos.fen()
    py_f = py_nnue_vl_float(py_model, fen0)
    py_i = py_nnue_vl_int8(py_model, fen0)
    cpp_vl = engine.evaluate_nnue_raw(pos)
    inc_drift = engine.verify_nnue_incremental(pos)
    print("[startpos]")
    print(f"  PST={pos.evaluate()}  C++={cpp_vl}  PyFloat={py_f:.2f}  PyInt8={py_i:.2f}")
    print(f"  |C++-PyFloat|={abs(cpp_vl - py_f):.2f}  |C++-PyInt8|={abs(cpp_vl - py_i):.2f}")
    print(f"  |PyFloat-PyInt8|={abs(py_f - py_i):.2f}  incremental_drift={inc_drift}")

    samples = load_samples(data_path, 500)
    diff_float: list[float] = []
    diff_int8: list[float] = []
    diff_py_paths: list[float] = []
    skipped = 0
    worst = ""
    max_df = 0.0

    for fen, _vl in samples:
        if not pos.set_fen(fen):
            skipped += 1
            continue
        cpp = engine.evaluate_nnue_raw(pos)
        pf = py_nnue_vl_float(py_model, fen)
        pi = py_nnue_vl_int8(py_model, fen)
        df = abs(cpp - pf)
        di = abs(cpp - pi)
        dp = abs(pf - pi)
        diff_float.append(df)
        diff_int8.append(di)
        diff_py_paths.append(dp)
        if df > max_df:
            max_df = df
            worst = fen

    arr_f = np.array(diff_float, dtype=np.float64)
    arr_i = np.array(diff_int8, dtype=np.float64)
    arr_p = np.array(diff_py_paths, dtype=np.float64)
    print(f"\n[compare] n={len(arr_f)} skipped={skipped}")
    print(f"  C++ vs PyFloat: mean={arr_f.mean():.3f} max={arr_f.max():.3f} p99={np.percentile(arr_f, 99):.3f}")
    print(f"  C++ vs PyInt8:  mean={arr_i.mean():.3f} max={arr_i.max():.3f} p99={np.percentile(arr_i, 99):.3f}")
    print(f"  PyFloat vs Int8: mean={arr_p.mean():.3f} max={arr_p.max():.3f}")
    print(f"  C++ within 1.0 of PyFloat: {(arr_f <= 1.0).mean() * 100:.1f}%")
    print(f"  C++ within 1.0 of PyInt8:  {(arr_i <= 1.0).mean() * 100:.1f}%")
    if worst:
        print(f"  worst fen: {worst[:60]}...")

    detail = engine.search_best_detail(pos, 200, False)
    print(f"\n[search+nnue] iccs={detail['iccs']!r} depth={detail['depth']} score={detail['score']}")

    ok = (
        arr_i.max() <= 1.0
        and abs(cpp_vl - py_i) <= 1.0
        and inc_drift == 0
        and arr_p.max() <= 200.0
    )
    print(f"\n{'PASS' if ok else 'CHECK'}: C++ INT8 NNUE matches Python int8 reference")
    sys.exit(0 if ok else 1)


if __name__ == "__main__":
    main()
