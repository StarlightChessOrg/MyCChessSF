"""Training losses for NNUE."""
from __future__ import annotations

import torch


def weighted_mse(
    pred: torch.Tensor,
    target: torch.Tensor,
    weights: torch.Tensor,
) -> torch.Tensor:
    wsum = weights.sum()
    if wsum <= 0:
        return pred.sum() * 0.0
    sq = (pred - target) ** 2
    return (weights * sq).sum() / wsum


def quiet_weighted_mse(
    pred: torch.Tensor,
    target: torch.Tensor,
    weights: torch.Tensor,
    is_mate: torch.Tensor,
) -> torch.Tensor:
    """Weighted MSE on non-mate (quiet) samples only."""
    quiet = ~is_mate
    if not quiet.any():
        return pred.sum() * 0.0
    return weighted_mse(pred[quiet], target[quiet], weights[quiet])
