"""Evaluate float vs INT8-quantized NNUE on a dataset."""
from __future__ import annotations

import math

import torch
from torch import nn

from int8_nnue import QuantizedNNUE


@torch.no_grad()
def evaluate_models(
    float_model: nn.Module,
    quant_model: QuantizedNNUE,
    loader,
    device: torch.device,
    *,
    label_mean: float,
    label_std: float,
) -> dict[str, dict[str, float]]:
    float_model.eval()
    metrics = {
        "float": _empty_metrics(),
        "int8": _empty_metrics(),
    }

    for indices, offsets, targets_norm, targets_raw in loader:
        indices = indices.to(device)
        offsets = offsets.to(device)
        targets_norm = targets_norm.to(device)
        targets_raw = targets_raw.to(device)

        pred_float_norm = float_model(indices, offsets)
        pred_int8_norm = quant_model.forward_norm_int8(indices, offsets)

        _update_metrics(metrics["float"], pred_float_norm, targets_norm, targets_raw, label_mean, label_std)
        _update_metrics(metrics["int8"], pred_int8_norm, targets_norm, targets_raw, label_mean, label_std)

    return {name: _finalize_metrics(m) for name, m in metrics.items()}


def _empty_metrics() -> dict[str, float]:
    return {
        "n": 0.0,
        "loss": 0.0,
        "mae_norm": 0.0,
        "mae_vl": 0.0,
        "sum_pred": 0.0,
        "sum_true": 0.0,
        "sum_pred_sq": 0.0,
        "sum_true_sq": 0.0,
        "sum_cross": 0.0,
    }


def _update_metrics(
    m: dict[str, float],
    pred_norm: torch.Tensor,
    target_norm: torch.Tensor,
    target_raw: torch.Tensor,
    label_mean: float,
    label_std: float,
) -> None:
    pred_vl = pred_norm * label_std + label_mean
    bs = float(target_norm.size(0))

    m["n"] += bs
    m["loss"] += torch.nn.functional.mse_loss(pred_norm, target_norm, reduction="sum").item()
    m["mae_norm"] += (pred_norm - target_norm).abs().sum().item()
    m["mae_vl"] += (pred_vl - target_raw).abs().sum().item()

    m["sum_pred"] += pred_vl.sum().item()
    m["sum_true"] += target_raw.sum().item()
    m["sum_pred_sq"] += (pred_vl * pred_vl).sum().item()
    m["sum_true_sq"] += (target_raw * target_raw).sum().item()
    m["sum_cross"] += (pred_vl * target_raw).sum().item()


def _finalize_metrics(m: dict[str, float]) -> dict[str, float]:
    n = m["n"]
    if n <= 0:
        return {
            "loss": math.inf,
            "mae_norm": math.inf,
            "mae_vl": math.inf,
            "corr_vl": math.nan,
        }

    mean_pred = m["sum_pred"] / n
    mean_true = m["sum_true"] / n
    var_pred = max(m["sum_pred_sq"] / n - mean_pred * mean_pred, 0.0)
    var_true = max(m["sum_true_sq"] / n - mean_true * mean_true, 0.0)
    cov = m["sum_cross"] / n - mean_pred * mean_true
    denom = math.sqrt(var_pred * var_true)
    corr = cov / denom if denom > 1e-12 else 0.0

    return {
        "loss": m["loss"] / n,
        "mae_norm": m["mae_norm"] / n,
        "mae_vl": m["mae_vl"] / n,
        "corr_vl": corr,
    }
