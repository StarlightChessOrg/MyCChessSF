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

from int8_nnue import FloatLayerLinear, QuantizedLayerLinear, QuantizedNNUE

_MAGIC = b"XQNNUE01"
_VERSION_INT8_FC = 2
_VERSION_FLOAT_FC = 3


def export_bin(model: QuantizedNNUE, path: Path) -> None:
    m = model
    if m.fc_float:
        version = _VERSION_FLOAT_FC
        header = struct.pack(
            "<8sIddIIIIfffff",
            _MAGIC,
            version,
            m.label_mean,
            m.label_std,
            m.n_features,
            m.l1,
            m.l2,
            m.l3,
            float(m.ft_clip),
            float(m.act_clip),
            0.0,
            0.0,
            0.0,
        )
    else:
        if (
            not isinstance(m.fc0, QuantizedLayerLinear)
            or m.fc0.input_scale <= 0.0
            or m.fc1.input_scale <= 0.0
            or m.fc2.input_scale <= 0.0
        ):
            raise ValueError("FC input_scale unset; run calibrate_fc_input_scales() before export")
        version = _VERSION_INT8_FC
        header = struct.pack(
            "<8sIddIIIIfffff",
            _MAGIC,
            version,
            m.label_mean,
            m.label_std,
            m.n_features,
            m.l1,
            m.l2,
            m.l3,
            float(m.ft_clip),
            float(m.act_clip),
            float(m.fc0.input_scale),
            float(m.fc1.input_scale),
            float(m.fc2.input_scale),
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
        if m.fc_float:
            for layer in (m.fc0, m.fc1, m.fc2):
                assert isinstance(layer, FloatLayerLinear)
                _write_tensor(f, layer.weight, "f")
                _write_tensor(f, layer.bias, "f")
        else:
            for layer in (m.fc0, m.fc1, m.fc2):
                assert isinstance(layer, QuantizedLayerLinear)
                _write_tensor(f, layer.weight_int8, "b")
                _write_tensor(f, layer.bias_int32, "i")
                _write_tensor(f, layer.scales, "f")


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
    if model.fc_float:
        print(
            f"[export] {args.input} -> {args.output}  ({args.output.stat().st_size:,} bytes)  "
            f"format=v3 (FT int8 + FC float32)"
        )
    else:
        assert isinstance(model.fc0, QuantizedLayerLinear)
        print(
            f"[export] {args.input} -> {args.output}  ({args.output.stat().st_size:,} bytes)  "
            f"in_scales=({model.fc0.input_scale:.6g}, {model.fc1.input_scale:.6g}, {model.fc2.input_scale:.6g})"
        )


if __name__ == "__main__":
    main()
