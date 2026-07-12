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


def weighted_pairwise_ranking_loss(
    pred: torch.Tensor,
    target: torch.Tensor,
    weights: torch.Tensor,
    *,
    max_pairs: int = 4096,
) -> torch.Tensor:
    """Logistic pairwise ranking loss with symmetric pair weights."""
    n = pred.numel()
    if n < 2:
        return pred.sum() * 0.0

    pair_count = min(max_pairs, n * (n - 1))
    i = torch.randint(0, n, (pair_count,), device=pred.device)
    j = torch.randint(0, n, (pair_count,), device=pred.device)
    tie = i == j
    if tie.any():
        j = torch.where(tie, (j + 1) % n, j)

    diff_t = target[i] - target[j]
    mask = diff_t.abs() > 1e-6
    if not mask.any():
        return pred.sum() * 0.0

    sign = torch.sign(diff_t[mask])
    diff_p = pred[i][mask] - pred[j][mask]
    pair_loss = torch.nn.functional.softplus(-sign * diff_p)
    pair_w = (weights[i][mask] + weights[j][mask]) * 0.5
    wsum = pair_w.sum()
    if wsum <= 0:
        return pred.sum() * 0.0
    return (pair_w * pair_loss).sum() / wsum


def pairwise_ranking_loss(
    pred: torch.Tensor,
    target: torch.Tensor,
    *,
    max_pairs: int = 4096,
) -> torch.Tensor:
    """Unweighted logistic pairwise ranking loss (RankNet-style)."""
    n = pred.numel()
    if n < 2:
        return pred.sum() * 0.0
    weights = torch.ones(n, device=pred.device, dtype=pred.dtype)
    return weighted_pairwise_ranking_loss(pred, target, weights, max_pairs=max_pairs)


def mate_aware_weighted_loss(
    pred: torch.Tensor,
    target: torch.Tensor,
    weights: torch.Tensor,
    is_mate: torch.Tensor,
    *,
    mse_weight: float = 0.5,
    ranking_weight: float = 0.5,
    ranking_max_pairs: int = 4096,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """Quiet samples: MSE + ranking. Mate samples: ranking only."""
    quiet = ~is_mate

    if quiet.any() and mse_weight > 0.0:
        mse = weighted_mse(pred[quiet], target[quiet], weights[quiet])
    else:
        mse = pred.sum() * 0.0

    if ranking_weight > 0.0:
        rank = weighted_pairwise_ranking_loss(
            pred,
            target,
            weights,
            max_pairs=ranking_max_pairs,
        )
    else:
        rank = pred.sum() * 0.0

    total = mse_weight * mse + ranking_weight * rank
    return total, mse, rank


def combined_mse_ranking_loss(
    pred: torch.Tensor,
    target: torch.Tensor,
    *,
    mse_weight: float = 0.5,
    ranking_weight: float = 0.5,
    ranking_max_pairs: int = 4096,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """Unweighted hybrid loss (legacy helper)."""
    n = pred.numel()
    weights = torch.ones(n, device=pred.device, dtype=pred.dtype)
    is_mate = torch.zeros(n, device=pred.device, dtype=torch.bool)
    return mate_aware_weighted_loss(
        pred,
        target,
        weights,
        is_mate,
        mse_weight=mse_weight,
        ranking_weight=ranking_weight,
        ranking_max_pairs=ranking_max_pairs,
    )
