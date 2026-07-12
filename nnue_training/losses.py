"""Training losses for NNUE."""
from __future__ import annotations

import torch


def pairwise_ranking_loss(
    pred: torch.Tensor,
    target: torch.Tensor,
    *,
    max_pairs: int = 4096,
) -> torch.Tensor:
    """Logistic pairwise ranking loss (RankNet-style).

    Random pairs (i, j) with target_i != target_j; penalizes pred ordering that
    disagrees with the label ordering. Order is invariant to affine target scale,
    so normalized z-score space matches raw vl ordering.
    """
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
    return torch.nn.functional.softplus(-sign * diff_p).mean()


def combined_mse_ranking_loss(
    pred: torch.Tensor,
    target: torch.Tensor,
    *,
    mse_weight: float = 0.5,
    ranking_weight: float = 0.5,
    ranking_max_pairs: int = 4096,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """Return ``(total, mse, ranking)`` scalar losses."""
    mse = torch.mean((pred - target) ** 2)
    if ranking_weight <= 0.0:
        rank = pred.sum() * 0.0
        return mse_weight * mse, mse, rank

    rank = pairwise_ranking_loss(pred, target, max_pairs=ranking_max_pairs)
    total = mse_weight * mse + ranking_weight * rank
    return total, mse, rank
