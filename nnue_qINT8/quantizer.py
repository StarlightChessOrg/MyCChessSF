"""Symmetric INT8 weight quantization helpers."""
from __future__ import annotations

import torch
from torch import Tensor


def _max_abs(t: Tensor, dim: int | None) -> Tensor:
    if dim is None:
        return t.abs().max().clamp(min=1e-8)
    return t.abs().amax(dim=dim).clamp(min=1e-8)


def symmetric_quantize_per_channel(
    weight: Tensor,
    *,
    channel_dim: int,
) -> tuple[Tensor, Tensor]:
    """
    Quantize ``weight`` to int8 with one scale per output channel.

    ``channel_dim`` indexes the dimension treated as output channels
    (e.g. 0 for ``Linear.weight`` shaped ``[out, in]``).
    """
    if weight.dim() != 2:
        raise ValueError(f"Expected 2D weight, got shape {tuple(weight.shape)}")

    reduce_dims = [d for d in range(weight.dim()) if d != channel_dim]
    max_abs = _max_abs(weight, dim=reduce_dims)
    scales = max_abs / 127.0

    if channel_dim == 0:
        q = torch.round(weight / scales.unsqueeze(1)).clamp(-127, 127).to(torch.int8)
    else:
        q = torch.round(weight / scales.unsqueeze(0)).clamp(-127, 127).to(torch.int8)
    return q, scales.to(torch.float32)


def quantize_bias_per_channel(
    bias: Tensor,
    scales: Tensor,
    *,
    channel_dim: int,
) -> Tensor:
    """Map float bias to int32 accumulator space aligned with weight scales."""
    if channel_dim == 0:
        return torch.round(bias / scales).to(torch.int32)
    return torch.round(bias / scales).to(torch.int32)


def symmetric_quantize_per_column(matrix: Tensor) -> tuple[Tensor, Tensor]:
    """Quantize ``[rows, cols]`` matrix with one scale per column."""
    max_abs = _max_abs(matrix, dim=0)
    scales = max_abs / 127.0
    q = torch.round(matrix / scales.unsqueeze(0)).clamp(-127, 127).to(torch.int8)
    return q, scales.to(torch.float32)


def quantize_bias_per_column(bias: Tensor, scales: Tensor) -> Tensor:
    return torch.round(bias / scales).to(torch.int32)
