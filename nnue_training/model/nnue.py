"""MyCChessSF NNUE: sparse FT + small FC head."""
from __future__ import annotations

import torch
import torch.nn as nn
from torch import Tensor

from model.activations import clipped_relu, sq_clipped_relu


class NNUE(nn.Module):
    """
    Sparse Feature Transformer + FC evaluation head.

    Input: concatenated feature indices + offsets (``nn.EmbeddingBag``).
    Output: scalar in z-score normalized label space.
    """

    def __init__(
        self,
        n_features: int = 1260,
        l1: int = 512,
        l2: int = 32,
        l3: int = 32,
        *,
        ft_clip: float = 256.0,
        act_clip: float = 1.0,
    ) -> None:
        super().__init__()
        if l1 % 2 != 0:
            raise ValueError("l1 must be even for pairwise FT transform")
        self.n_features = n_features
        self.l1 = l1
        self.l2 = l2
        self.l3 = l3
        self.ft_clip = ft_clip
        self.act_clip = act_clip

        self.ft_embed = nn.EmbeddingBag(n_features, l1, mode="sum")
        self.ft_bias = nn.Parameter(torch.zeros(l1))

        self.fc0 = nn.Linear(l1, l2)
        self.fc1 = nn.Linear(l2 * 2, l3)
        self.fc2 = nn.Linear(l3 * 2, 1)

        self._init_weights()

    def _init_weights(self) -> None:
        nn.init.normal_(self.ft_embed.weight, mean=0.0, std=0.1)
        nn.init.zeros_(self.ft_bias)
        for layer in (self.fc0, self.fc1, self.fc2):
            nn.init.xavier_uniform_(layer.weight)
            nn.init.zeros_(layer.bias)

    def _feature_transform(self, acc: Tensor) -> Tensor:
        x = torch.clamp(acc, min=0.0, max=self.ft_clip)
        a = x[:, 0::2]
        b = x[:, 1::2]
        prod = (a * b) / 512.0
        out = torch.empty_like(x)
        out[:, 0::2] = prod
        out[:, 1::2] = prod
        return out

    def forward(self, indices: Tensor, offsets: Tensor) -> Tensor:
        acc = self.ft_embed(indices, offsets) + self.ft_bias
        h = self._feature_transform(acc)

        x0 = self.fc0(h)
        x0_cat = torch.cat(
            (sq_clipped_relu(x0, self.act_clip), clipped_relu(x0, self.act_clip)),
            dim=1,
        )

        x1 = self.fc1(x0_cat)
        x1_cat = torch.cat(
            (sq_clipped_relu(x1, self.act_clip), clipped_relu(x1, self.act_clip)),
            dim=1,
        )

        return self.fc2(x1_cat).squeeze(-1)
