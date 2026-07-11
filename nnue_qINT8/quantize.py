"""Quantize a trained float NNUE checkpoint to INT8."""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

import torch
import yaml

ROOT = Path(__file__).resolve().parent
TRAINING_ROOT = ROOT.parent / "nnue_training"
REPO_ROOT = ROOT.parent

for p in (ROOT, TRAINING_ROOT, REPO_ROOT):
    if str(p) not in sys.path:
        sys.path.insert(0, str(p))

from data.dataset import NnueDataset, load_samples, make_dataloader
from evaluate import evaluate_models
from int8_nnue import QuantizedNNUE, calibrate_fc_input_scales, quantize_float_nnue
from model.nnue import NNUE


def load_config(path: Path) -> dict:
    with open(path, encoding="utf-8") as f:
        return yaml.safe_load(f)


def resolve_path(base: Path, maybe_rel: str) -> Path:
    p = Path(maybe_rel)
    return p if p.is_absolute() else (base / p).resolve()


def load_float_checkpoint(checkpoint_path: Path, device: torch.device) -> tuple[NNUE, dict]:
    ckpt = torch.load(checkpoint_path, map_location=device, weights_only=False)
    config = ckpt.get("config", {})
    model_cfg = config.get("model", {})

    model = NNUE(
        n_features=int(ckpt.get("n_features", model_cfg.get("n_features", 1260))),
        l1=int(model_cfg.get("l1", 512)),
        l2=int(model_cfg.get("l2", 32)),
        l3=int(model_cfg.get("l3", 32)),
    )
    model.load_state_dict(ckpt["model_state_dict"])
    model.to(device)
    model.eval()
    return model, ckpt


def build_calibration_loader(
    data_source: Path,
    pattern: str,
    *,
    label_mean: float,
    label_std: float,
    max_samples: int,
    batch_size: int,
    num_workers: int,
    prefetch_factor: int,
    device: torch.device,
) -> tuple[torch.utils.data.DataLoader, int]:
    samples, skipped = load_samples(data_source, pattern)
    if max_samples > 0 and len(samples) > max_samples:
        samples = samples[:max_samples]

    dataset = NnueDataset(samples, label_mean=label_mean, label_std=label_std)
    loader = make_dataloader(
        dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
        prefetch_factor=prefetch_factor,
        pin_memory=device.type == "cuda",
    )
    return loader, skipped


def print_metrics(name: str, metrics: dict[str, float]) -> None:
    print(
        f"  {name}: loss={metrics['loss']:.6f}  "
        f"mae_norm={metrics['mae_norm']:.6f}  "
        f"mae_vl={metrics['mae_vl']:.2f}  "
        f"corr_vl={metrics['corr_vl']:.4f}"
    )


def main() -> None:
    parser = argparse.ArgumentParser(description="Quantize MyCChessSF NNUE to INT8")
    parser.add_argument("--config", type=Path, default=ROOT / "configs" / "default.yaml")
    parser.add_argument(
        "--checkpoint",
        type=Path,
        default=None,
        help="Float NNUE checkpoint (.pt). Default: nnue_training/checkpoints/best.pt",
    )
    parser.add_argument(
        "--data-dir",
        type=Path,
        default=None,
        help="Calibration dataset directory (FEN\\tvl txt files). Default: ../nnue_data",
    )
    parser.add_argument(
        "--data-pattern",
        type=str,
        default=None,
        help='Glob pattern under data-dir, e.g. "worker_*.txt"',
    )
    parser.add_argument(
        "--max-samples",
        type=int,
        default=None,
        help="Max calibration samples (0 = all). Default from config.",
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=None,
        help="Output path for quantized model.",
    )
    args = parser.parse_args()

    cfg = load_config(args.config)
    data_cfg = cfg.get("data", {})
    quant_cfg = cfg.get("quantize", {})

    checkpoint_path = resolve_path(
        ROOT,
        str(args.checkpoint or cfg.get("checkpoint", "../nnue_training/checkpoints/best.pt")),
    )
    data_source = resolve_path(
        ROOT,
        str(args.data_dir or data_cfg.get("source", "../nnue_data")),
    )
    data_pattern = args.data_pattern or data_cfg.get("pattern", "worker_*.txt")
    max_samples = (
        args.max_samples
        if args.max_samples is not None
        else int(data_cfg.get("max_samples", 10000))
    )
    output_path = resolve_path(
        ROOT,
        str(args.output or cfg.get("output", "output/quantized.xqint8.pt")),
    )

    batch_size = int(quant_cfg.get("batch_size", 2048))
    num_workers = int(quant_cfg.get("num_workers", 4))
    prefetch_factor = int(quant_cfg.get("prefetch_factor", 2))
    device = torch.device(quant_cfg.get("device", "cpu"))

    print(f"[checkpoint] {checkpoint_path}")
    print(f"[data] source={data_source}  pattern={data_pattern!r}  max_samples={max_samples}")

    if not checkpoint_path.is_file():
        raise FileNotFoundError(f"Checkpoint not found: {checkpoint_path}")
    if not data_source.is_dir():
        raise FileNotFoundError(f"Dataset directory not found: {data_source}")

    float_model, ckpt = load_float_checkpoint(checkpoint_path, device)
    label_mean = float(ckpt["label_mean"])
    label_std = float(ckpt["label_std"])
    print(f"[label] mean={label_mean:.4f}  std={label_std:.4f}")

    calib_loader, skipped = build_calibration_loader(
        data_source,
        data_pattern,
        label_mean=label_mean,
        label_std=label_std,
        max_samples=max_samples,
        batch_size=batch_size,
        num_workers=num_workers,
        prefetch_factor=prefetch_factor,
        device=device,
    )
    print(f"[calibration] samples={len(calib_loader.dataset)}  skipped={skipped}")

    quant_model = quantize_float_nnue(float_model, source_checkpoint=str(checkpoint_path))
    quant_model.label_mean = label_mean
    quant_model.label_std = label_std

    fc0_s, fc1_s, fc2_s = calibrate_fc_input_scales(quant_model, calib_loader, device)
    print(f"[calibrate] fc input_scales: fc0={fc0_s:.6g}  fc1={fc1_s:.6g}  fc2={fc2_s:.6g}")

    ft_bytes = quant_model.ft.weight_int8.numel()
    fc_bytes = (
        quant_model.fc0.weight_int8.numel()
        + quant_model.fc1.weight_int8.numel()
        + quant_model.fc2.weight_int8.numel()
    )
    print(f"[quantize] ft_int8={ft_bytes:,} weights  fc_int8={fc_bytes:,} weights")

    print("[eval] comparing float vs int8 on calibration set")
    metrics = evaluate_models(
        float_model,
        quant_model,
        calib_loader,
        device,
        label_mean=label_mean,
        label_std=label_std,
    )
    print_metrics("float", metrics["float"])
    print_metrics("int8", metrics["int8"])

    output_path.parent.mkdir(parents=True, exist_ok=True)
    quant_model.save(str(output_path))
    print(f"[saved] {output_path}")

    bin_path = output_path.parent / "quantized.xqnnue.bin"
    from export_bin import export_bin

    export_bin(quant_model, bin_path)
    print(f"[saved] {bin_path}")


if __name__ == "__main__":
    main()
