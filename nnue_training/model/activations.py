"""NNUE-style clipped activations (float training)."""
from __future__ import annotations

import torch
from torch import Tensor


def clipped_relu(x: Tensor, max_val: float = 1.0) -> Tensor:
    return torch.clamp(torch.relu(x), max=max_val)


def sq_clipped_relu(x: Tensor, max_val: float = 1.0) -> Tensor:
    return torch.clamp(x * x, max=max_val)
