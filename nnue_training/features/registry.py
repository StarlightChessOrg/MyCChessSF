"""Feature extractor registry for NNUE training."""
from __future__ import annotations

from typing import Callable

import numpy as np

from features.half_ka_v2_hm import N_FEATURES as HALFKA_N_FEATURES
from features.half_ka_v2_hm import fen_to_feature_indices as halfka_fen_to_indices

FeatureFn = Callable[[str], np.ndarray]

DEFAULT_FEATURE_KIND = "half_ka_v2_hm"

_REGISTRY: dict[str, tuple[int, FeatureFn, str]] = {
    DEFAULT_FEATURE_KIND: (
        HALFKA_N_FEATURES,
        halfka_fen_to_indices,
        "Pikafish HalfKAv2_hm (16536-dim, king + attack buckets + mid mirror)",
    ),
}


def resolve_feature_kind(config: dict) -> str:
    data_cfg = config.get("data", {})
    model_cfg = config.get("model", {})
    kind = (
        data_cfg.get("feature_kind")
        or model_cfg.get("feature_kind")
        or DEFAULT_FEATURE_KIND
    )
    if kind not in _REGISTRY:
        known = ", ".join(sorted(_REGISTRY))
        raise ValueError(f"Unknown feature_kind {kind!r}; expected one of: {known}")
    return kind


def get_feature_spec(kind: str) -> tuple[int, FeatureFn, str]:
    return _REGISTRY[kind]
