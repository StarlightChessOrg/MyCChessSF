"""Load nnue_data FEN + vl samples."""
from __future__ import annotations

import glob
import re
import sys
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import torch
from torch.utils.data import DataLoader, Dataset

_REPO_ROOT = Path(__file__).resolve().parents[2]
if str(_REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(_REPO_ROOT))

from features.xqwl_psq import fen_to_feature_indices
from mycchess_sf.fen_parse import parse_fen_board

_WORKER_RE = re.compile(r"worker_(\d+)\.txt$")


@dataclass(frozen=True)
class Sample:
    fen: str
    vl: float


def _is_valid_fen(fen: str) -> bool:
    try:
        parse_fen_board(fen)
    except ValueError:
        return False
    return True


def _parse_line(line: str) -> Sample | None:
    line = line.strip()
    if not line:
        return None
    parts = line.split("\t")
    if len(parts) != 2:
        return None
    fen, vl_str = parts
    if not _is_valid_fen(fen):
        return None
    try:
        vl = float(vl_str)
    except ValueError:
        return None
    return Sample(fen=fen, vl=vl)


def _worker_id(path: Path) -> int | None:
    m = _WORKER_RE.search(path.name)
    return int(m.group(1)) if m else None


def load_samples(source: str | Path, pattern: str = "worker_*.txt") -> tuple[list[Sample], int]:
    root = Path(source)
    files = sorted(glob.glob(str(root / pattern)))
    samples: list[Sample] = []
    skipped = 0
    for fp in files:
        with open(fp, encoding="utf-8", errors="replace") as f:
            for line in f:
                s = _parse_line(line)
                if s is not None:
                    samples.append(s)
                elif line.strip():
                    skipped += 1
    return samples, skipped


def split_samples(
    source: str | Path,
    pattern: str,
    val_workers: list[int],
) -> tuple[list[Sample], list[Sample], int]:
    root = Path(source)
    val_ids = set(val_workers)
    val_set: list[Sample] = []
    train_set: list[Sample] = []
    skipped = 0

    files = sorted(glob.glob(str(root / pattern)))
    for fp in files:
        path = Path(fp)
        wid = _worker_id(path)
        bucket = val_set if wid is not None and wid in val_ids else train_set
        with open(path, encoding="utf-8", errors="replace") as f:
            for line in f:
                s = _parse_line(line)
                if s is not None:
                    bucket.append(s)
                elif line.strip():
                    skipped += 1
    return train_set, val_set, skipped


def compute_zscore_stats(samples: list[Sample]) -> tuple[float, float]:
    if not samples:
        raise ValueError("Cannot compute z-score stats on empty sample list")
    values = np.array([s.vl for s in samples], dtype=np.float64)
    mean = float(values.mean())
    std = float(values.std())
    if std < 1e-8:
        std = 1.0
    return mean, std


class NnueDataset(Dataset):
    def __init__(
        self,
        samples: list[Sample],
        *,
        label_mean: float,
        label_std: float,
    ) -> None:
        self.samples = samples
        self.label_mean = label_mean
        self.label_std = label_std

    def __len__(self) -> int:
        return len(self.samples)

    def __getitem__(self, idx: int) -> tuple[np.ndarray, float, float]:
        s = self.samples[idx]
        indices = fen_to_feature_indices(s.fen)
        target_norm = (s.vl - self.label_mean) / self.label_std
        return indices, target_norm, s.vl


def collate_fn(batch: list[tuple[np.ndarray, float, float]]) -> tuple[torch.Tensor, ...]:
    indices_list: list[int] = []
    offsets: list[int] = [0]
    targets_norm: list[float] = []
    targets_raw: list[float] = []

    for feat_indices, target_norm, raw in batch:
        indices_list.extend(feat_indices.tolist())
        offsets.append(offsets[-1] + len(feat_indices))
        targets_norm.append(target_norm)
        targets_raw.append(raw)

    return (
        torch.tensor(indices_list, dtype=torch.long),
        torch.tensor(offsets[:-1], dtype=torch.long),
        torch.tensor(targets_norm, dtype=torch.float32),
        torch.tensor(targets_raw, dtype=torch.float32),
    )


def make_dataloader(
    dataset: NnueDataset,
    *,
    batch_size: int,
    shuffle: bool,
    num_workers: int = 0,
    prefetch_factor: int = 2,
    pin_memory: bool = False,
) -> DataLoader:
    kwargs: dict = {
        "dataset": dataset,
        "batch_size": batch_size,
        "shuffle": shuffle,
        "num_workers": num_workers,
        "collate_fn": collate_fn,
        "pin_memory": pin_memory and num_workers > 0,
    }
    if num_workers > 0:
        kwargs["persistent_workers"] = True
        kwargs["prefetch_factor"] = max(1, prefetch_factor)
    return DataLoader(**kwargs)
