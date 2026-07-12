"""INT8-quantized MyCChessSF NNUE (storage + reference inference)."""
from __future__ import annotations

from dataclasses import dataclass

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch import Tensor

from quantizer import (
    quantize_bias_per_channel,
    quantize_bias_per_column,
    symmetric_quantize_per_channel,
    symmetric_quantize_per_column,
)


def _clipped_relu(x: Tensor, max_val: float) -> Tensor:
    return torch.clamp(torch.relu(x), max=max_val)


def _sq_clipped_relu(x: Tensor, max_val: float) -> Tensor:
    return torch.clamp(x * x, max=max_val)


def _feature_transform(acc: Tensor, *, ft_clip: float) -> Tensor:
    x = torch.clamp(acc, min=0.0, max=ft_clip)
    a = x[:, 0::2]
    b = x[:, 1::2]
    prod = (a * b) / 512.0
    out = torch.empty_like(x)
    out[:, 0::2] = prod
    out[:, 1::2] = prod
    return out


def _quantize_input(x: Tensor, input_scale: float) -> Tensor:
    if input_scale <= 0.0:
        raise ValueError(f"input_scale must be positive, got {input_scale}")
    q = torch.round(x / input_scale).clamp(0, 127).to(torch.int32)
    return q


@dataclass
class QuantizedLayerLinear:
    weight_int8: Tensor
    bias_int32: Tensor
    scales: Tensor
    input_scale: float = 0.0

    @property
    def out_features(self) -> int:
        return int(self.weight_int8.size(0))

    @property
    def in_features(self) -> int:
        return int(self.weight_int8.size(1))

    def dequant_weight(self) -> Tensor:
        return self.weight_int8.float() * self.scales.unsqueeze(1)

    def forward(self, x: Tensor) -> Tensor:
        w = self.dequant_weight()
        b = self.bias_int32.float() * self.scales
        return F.linear(x, w, b)

    def forward_int8(self, x: Tensor) -> Tensor:
        if self.input_scale <= 0.0:
            raise ValueError("input_scale is unset; run calibrate_fc_input_scales() first")
        x_q = _quantize_input(x, self.input_scale)
        dot = (self.weight_int8.unsqueeze(0).to(torch.int32) * x_q.unsqueeze(1).to(torch.int32)).sum(dim=2)
        b = self.bias_int32.float().unsqueeze(0)
        return self.scales.unsqueeze(0) * b + self.scales.unsqueeze(0) * dot.float() * self.input_scale


@dataclass
class QuantizedFeatureTransformer:
    weight_int8: Tensor
    bias_int32: Tensor
    scales: Tensor

    @property
    def n_features(self) -> int:
        return int(self.weight_int8.size(0))

    @property
    def l1(self) -> int:
        return int(self.weight_int8.size(1))

    def dequant_weight(self) -> Tensor:
        return self.weight_int8.float() * self.scales.unsqueeze(0)

    def forward(self, indices: Tensor, offsets: Tensor) -> Tensor:
        w = self.dequant_weight()
        acc = F.embedding_bag(indices, w, offsets, mode="sum")
        b = self.bias_int32.float() * self.scales
        return acc + b


@dataclass
class QuantizedNNUE:
    ft: QuantizedFeatureTransformer
    fc0: QuantizedLayerLinear
    fc1: QuantizedLayerLinear
    fc2: QuantizedLayerLinear
    label_mean: float
    label_std: float
    n_features: int
    l1: int
    l2: int
    l3: int
    ft_clip: float
    act_clip: float
    source_checkpoint: str

    def forward_norm(self, indices: Tensor, offsets: Tensor) -> Tensor:
        acc = self.ft.forward(indices, offsets)
        h = _feature_transform(acc, ft_clip=self.ft_clip)

        x0 = self.fc0.forward(h)
        x0_cat = torch.cat(
            (_sq_clipped_relu(x0, self.act_clip), _clipped_relu(x0, self.act_clip)),
            dim=1,
        )

        x1 = self.fc1.forward(x0_cat)
        x1_cat = torch.cat(
            (_sq_clipped_relu(x1, self.act_clip), _clipped_relu(x1, self.act_clip)),
            dim=1,
        )

        return self.fc2.forward(x1_cat).squeeze(-1)

    def forward_norm_int8(self, indices: Tensor, offsets: Tensor) -> Tensor:
        acc = self.ft.forward(indices, offsets)
        h = _feature_transform(acc, ft_clip=self.ft_clip)

        x0 = self.fc0.forward_int8(h)
        x0_cat = torch.cat(
            (_sq_clipped_relu(x0, self.act_clip), _clipped_relu(x0, self.act_clip)),
            dim=1,
        )

        x1 = self.fc1.forward_int8(x0_cat)
        x1_cat = torch.cat(
            (_sq_clipped_relu(x1, self.act_clip), _clipped_relu(x1, self.act_clip)),
            dim=1,
        )

        return self.fc2.forward_int8(x1_cat).squeeze(-1)

    def forward_vl(self, indices: Tensor, offsets: Tensor) -> Tensor:
        return self.forward_norm(indices, offsets) * self.label_std + self.label_mean

    def forward_vl_int8(self, indices: Tensor, offsets: Tensor) -> Tensor:
        return self.forward_norm_int8(indices, offsets) * self.label_std + self.label_mean

    def state_dict(self) -> dict:
        return {
            "format": "mycchesssf_xqint8_v2",
            "label_mean": self.label_mean,
            "label_std": self.label_std,
            "n_features": self.n_features,
            "l1": self.l1,
            "l2": self.l2,
            "l3": self.l3,
            "ft_clip": self.ft_clip,
            "act_clip": self.act_clip,
            "source_checkpoint": self.source_checkpoint,
            "ft_weight_int8": self.ft.weight_int8,
            "ft_bias_int32": self.ft.bias_int32,
            "ft_scales": self.ft.scales,
            "fc0_weight_int8": self.fc0.weight_int8,
            "fc0_bias_int32": self.fc0.bias_int32,
            "fc0_scales": self.fc0.scales,
            "fc0_input_scale": self.fc0.input_scale,
            "fc1_weight_int8": self.fc1.weight_int8,
            "fc1_bias_int32": self.fc1.bias_int32,
            "fc1_scales": self.fc1.scales,
            "fc1_input_scale": self.fc1.input_scale,
            "fc2_weight_int8": self.fc2.weight_int8,
            "fc2_bias_int32": self.fc2.bias_int32,
            "fc2_scales": self.fc2.scales,
            "fc2_input_scale": self.fc2.input_scale,
        }

    @classmethod
    def from_state_dict(cls, data: dict) -> QuantizedNNUE:
        fmt = data.get("format")
        if fmt not in ("mycchesssf_xqint8_v1", "mycchesssf_xqint8_v2"):
            raise ValueError(f"Unsupported quantized format: {fmt!r}")

        act_clip = float(data["act_clip"])
        ft_clip = float(data["ft_clip"])

        def _layer(prefix: str) -> QuantizedLayerLinear:
            input_scale = float(data.get(f"{prefix}_input_scale", 0.0))
            if input_scale <= 0.0:
                if prefix == "fc0":
                    input_scale = max((ft_clip * ft_clip / 512.0) / 127.0, 1e-7)
                else:
                    input_scale = max(act_clip / 127.0, 1e-7)
            return QuantizedLayerLinear(
                weight_int8=data[f"{prefix}_weight_int8"],
                bias_int32=data[f"{prefix}_bias_int32"],
                scales=data[f"{prefix}_scales"],
                input_scale=input_scale,
            )

        ft = QuantizedFeatureTransformer(
            weight_int8=data["ft_weight_int8"],
            bias_int32=data["ft_bias_int32"],
            scales=data["ft_scales"],
        )
        return cls(
            ft=ft,
            fc0=_layer("fc0"),
            fc1=_layer("fc1"),
            fc2=_layer("fc2"),
            label_mean=float(data["label_mean"]),
            label_std=float(data["label_std"]),
            n_features=int(data["n_features"]),
            l1=int(data["l1"]),
            l2=int(data["l2"]),
            l3=int(data["l3"]),
            ft_clip=ft_clip,
            act_clip=act_clip,
            source_checkpoint=str(data.get("source_checkpoint", "")),
        )

    def save(self, path: str) -> None:
        torch.save(self.state_dict(), path)

    @classmethod
    def load(cls, path: str) -> QuantizedNNUE:
        data = torch.load(path, map_location="cpu", weights_only=False)
        return cls.from_state_dict(data)


@torch.no_grad()
def calibrate_fc_input_scales(
    model: QuantizedNNUE,
    loader: torch.utils.data.DataLoader,
    device: torch.device,
) -> tuple[float, float, float]:
    """Scan calibration data for per-layer FC input maxima; set input_scale = max/127."""
    max_fc0 = 0.0
    max_fc1 = 0.0
    max_fc2 = 0.0

    for batch in loader:
        indices, offsets, _targets_norm, _targets_raw, _is_mate = batch
        indices = indices.to(device)
        offsets = offsets.to(device)

        acc = model.ft.forward(indices, offsets)
        h = _feature_transform(acc, ft_clip=model.ft_clip)
        max_fc0 = max(max_fc0, float(h.max().item()))

        x0 = model.fc0.forward(h)
        x0_cat = torch.cat(
            (_sq_clipped_relu(x0, model.act_clip), _clipped_relu(x0, model.act_clip)),
            dim=1,
        )
        max_fc1 = max(max_fc1, float(x0_cat.max().item()))

        x1 = model.fc1.forward(x0_cat)
        x1_cat = torch.cat(
            (_sq_clipped_relu(x1, model.act_clip), _clipped_relu(x1, model.act_clip)),
            dim=1,
        )
        max_fc2 = max(max_fc2, float(x1_cat.max().item()))

    fc0_scale = max(max_fc0 / 127.0, 1e-7)
    fc1_scale = max(max_fc1 / 127.0, model.act_clip / 127.0, 1e-7)
    fc2_scale = max(max_fc2 / 127.0, model.act_clip / 127.0, 1e-7)

    model.fc0.input_scale = fc0_scale
    model.fc1.input_scale = fc1_scale
    model.fc2.input_scale = fc2_scale
    return fc0_scale, fc1_scale, fc2_scale


def quantize_float_nnue(model: nn.Module, *, source_checkpoint: str) -> QuantizedNNUE:
    ft_w = model.ft_embed.weight.detach().cpu()
    ft_b = model.ft_bias.detach().cpu()

    ft_q, ft_scales = symmetric_quantize_per_column(ft_w)
    ft_b_q = quantize_bias_per_column(ft_b, ft_scales)

    def _q_linear(layer: nn.Linear) -> QuantizedLayerLinear:
        w_q, scales = symmetric_quantize_per_channel(layer.weight.detach().cpu(), channel_dim=0)
        b_q = quantize_bias_per_channel(layer.bias.detach().cpu(), scales, channel_dim=0)
        return QuantizedLayerLinear(weight_int8=w_q, bias_int32=b_q, scales=scales, input_scale=0.0)

    return QuantizedNNUE(
        ft=QuantizedFeatureTransformer(weight_int8=ft_q, bias_int32=ft_b_q, scales=ft_scales),
        fc0=_q_linear(model.fc0),
        fc1=_q_linear(model.fc1),
        fc2=_q_linear(model.fc2),
        label_mean=0.0,
        label_std=1.0,
        n_features=int(model.n_features),
        l1=int(model.l1),
        l2=int(model.l2),
        l3=int(model.l3),
        ft_clip=float(model.ft_clip),
        act_clip=float(model.act_clip),
        source_checkpoint=source_checkpoint,
    )
