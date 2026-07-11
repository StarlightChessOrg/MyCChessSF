#!/usr/bin/env python3
"""Export QuantizedNNUE (.xqint8.pt) to a C++-loadable binary (.xqnnue.bin)."""
from __future__ import annotations

import argparse
import struct
import sys
from pathlib import Path

import torch

ROOT = Path(__file__).resolve().parent
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from int8_nnue import QuantizedNNUE

_MAGIC = b"XQNNUE01"
_VERSION = 1


def export_bin(model: QuantizedNNUE, path: Path) -> None:
    m = model
    header = struct.pack(
        "<8sIddIIIIff",
        _MAGIC,
        _VERSION,
        m.label_mean,
        m.label_std,
        m.n_features,
        m.l1,
        m.l2,
        m.l3,
        float(m.ft_clip),
        float(m.act_clip),
    )

    def _write_tensor(f, t: torch.Tensor, dtype_code: str) -> None:
        arr = t.detach().cpu().contiguous().numpy()
        if dtype_code == "b":
            f.write(arr.astype("i1").tobytes())
        elif dtype_code == "i":
            f.write(arr.astype("<i4").tobytes())
        elif dtype_code == "f":
            f.write(arr.astype("<f4").tobytes())
        else:
            raise ValueError(dtype_code)

    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "wb") as f:
        f.write(header)
        _write_tensor(f, m.ft.weight_int8, "b")
        _write_tensor(f, m.ft.bias_int32, "i")
        _write_tensor(f, m.ft.scales, "f")
        _write_tensor(f, m.fc0.weight_int8, "b")
        _write_tensor(f, m.fc0.bias_int32, "i")
        _write_tensor(f, m.fc0.scales, "f")
        _write_tensor(f, m.fc1.weight_int8, "b")
        _write_tensor(f, m.fc1.bias_int32, "i")
        _write_tensor(f, m.fc1.scales, "f")
        _write_tensor(f, m.fc2.weight_int8, "b")
        _write_tensor(f, m.fc2.bias_int32, "i")
        _write_tensor(f, m.fc2.scales, "f")


def main() -> None:
    parser = argparse.ArgumentParser(description="Export xqint8 PyTorch checkpoint to xqnnue.bin")
    parser.add_argument(
        "--input",
        type=Path,
        default=ROOT / "output" / "quantized.xqint8.pt",
        help="Quantized NNUE checkpoint (.xqint8.pt)",
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=ROOT / "output" / "quantized.xqnnue.bin",
        help="Output binary for C++ loader",
    )
    args = parser.parse_args()

    if not args.input.is_file():
        raise FileNotFoundError(f"Input not found: {args.input}")

    model = QuantizedNNUE.load(str(args.input))
    export_bin(model, args.output)
    print(f"[export] {args.input} -> {args.output}  ({args.output.stat().st_size:,} bytes)")


if __name__ == "__main__":
    main()
