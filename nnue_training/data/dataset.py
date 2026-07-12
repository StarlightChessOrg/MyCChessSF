"""Load nnue_data FEN + vl samples."""
from __future__ import annotations

import glob
import os
import random
import re
import sys
from concurrent.futures import ProcessPoolExecutor, as_completed
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

DEFAULT_DATA_PATTERN = "worker_*/chunk_*.txt"
_WORKER_DIR_RE = re.compile(r"^worker_(\d+)$")


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
    m = _WORKER_DIR_RE.match(path.parent.name)
    return int(m.group(1)) if m else None


def _resolve_load_workers(load_workers: int) -> int:
    if load_workers <= 0:
        load_workers = os.cpu_count() or 1
    return max(1, load_workers)


def _parse_file_range(path_str: str, start: int, end: int) -> tuple[list[Sample], int, int]:
    samples: list[Sample] = []
    skipped = 0
    lines = 0

    with open(path_str, "rb") as f:
        f.seek(start)
        if start > 0:
            f.readline()

        while f.tell() < end:
            raw = f.readline()
            if not raw:
                break
            lines += 1
            line = raw.decode("utf-8", errors="replace")
            sample = _parse_line(line)
            if sample is not None:
                samples.append(sample)
            elif line.strip():
                skipped += 1

    return samples, skipped, lines


def _load_task_group(tasks: list[tuple[str, int, int]]) -> tuple[list[Sample], int, int]:
    samples: list[Sample] = []
    skipped = 0
    lines = 0
    for path_str, start, end in tasks:
        chunk_samples, chunk_skipped, chunk_lines = _parse_file_range(path_str, start, end)
        samples.extend(chunk_samples)
        skipped += chunk_skipped
        lines += chunk_lines
    return samples, skipped, lines


def _byte_range_tasks(path: Path, num_chunks: int) -> list[tuple[str, int, int]]:
    size = path.stat().st_size
    if num_chunks <= 1 or size == 0:
        return [(str(path), 0, size)]

    path_str = str(path)
    chunk_size = size // num_chunks
    tasks: list[tuple[str, int, int]] = []

    with open(path, "rb") as f:
        start = 0
        for i in range(num_chunks):
            if i == num_chunks - 1:
                end = size
            else:
                f.seek(start + chunk_size)
                f.readline()
                end = f.tell()
            if end > start:
                tasks.append((path_str, start, end))
            start = end

    return tasks


def _build_load_task_groups(files: list[Path], load_workers: int) -> list[list[tuple[str, int, int]]]:
    if len(files) == 1:
        return [[task] for task in _byte_range_tasks(files[0], load_workers)]

    whole_tasks = [(str(path), 0, path.stat().st_size) for path in files]
    groups: list[list[tuple[str, int, int]]] = [[] for _ in range(load_workers)]
    for index, task in enumerate(whole_tasks):
        groups[index % load_workers].append(task)
    return [group for group in groups if group]


def _load_samples_serial(
    files: list[Path],
    *,
    progress_every: int,
) -> tuple[list[Sample], int]:
    samples: list[Sample] = []
    skipped = 0
    total_lines = 0

    for path in files:
        print(f"[data] loading {path.name} ...", flush=True)
        file_samples, file_skipped, file_lines = _parse_file_range(str(path), 0, path.stat().st_size)
        samples.extend(file_samples)
        skipped += file_skipped
        total_lines += file_lines
        if progress_every > 0 and total_lines % progress_every < file_lines:
            print(
                f"[data]   {total_lines:,} lines read, "
                f"{len(samples):,} valid, {skipped:,} skipped",
                flush=True,
            )

    print(
        f"[data] load complete: {total_lines:,} lines, "
        f"{len(samples):,} valid, {skipped:,} skipped",
        flush=True,
    )
    return samples, skipped


def _load_samples_parallel(
    files: list[Path],
    *,
    load_workers: int,
) -> tuple[list[Sample], int]:
    task_groups = _build_load_task_groups(files, load_workers)
    print(
        f"[data] parallel load: {len(files)} file(s), "
        f"{load_workers} worker(s), {len(task_groups)} task group(s)",
        flush=True,
    )

    samples: list[Sample] = []
    skipped = 0
    total_lines = 0
    completed = 0

    with ProcessPoolExecutor(max_workers=load_workers) as pool:
        futures = [pool.submit(_load_task_group, group) for group in task_groups]
        for future in as_completed(futures):
            group_samples, group_skipped, group_lines = future.result()
            samples.extend(group_samples)
            skipped += group_skipped
            total_lines += group_lines
            completed += 1
            print(
                f"[data]   group {completed}/{len(task_groups)} done, "
                f"{total_lines:,} lines read, {len(samples):,} valid, {skipped:,} skipped",
                flush=True,
            )

    print(
        f"[data] load complete: {total_lines:,} lines, "
        f"{len(samples):,} valid, {skipped:,} skipped",
        flush=True,
    )
    return samples, skipped


def load_samples(
    source: str | Path,
    pattern: str = DEFAULT_DATA_PATTERN,
    *,
    load_workers: int = 0,
    progress_every: int = 500_000,
) -> tuple[list[Sample], int]:
    root = Path(source)
    files = sorted(Path(fp) for fp in glob.glob(str(root / pattern)))
    if not files:
        return [], 0

    workers = _resolve_load_workers(load_workers)
    if workers <= 1:
        return _load_samples_serial(files, progress_every=progress_every)
    return _load_samples_parallel(files, load_workers=workers)


def _split_file_by_worker(path_str: str, val_ids: set[int]) -> tuple[list[Sample], list[Sample], int]:
    path = Path(path_str)
    worker = _worker_id(path)
    to_val = worker is not None and worker in val_ids

    file_samples, skipped, _ = _parse_file_range(path_str, 0, path.stat().st_size)
    if to_val:
        return [], file_samples, skipped
    return file_samples, [], skipped


def _split_files_serial(files: list[Path], val_ids: set[int]) -> tuple[list[Sample], list[Sample], int]:
    train_set: list[Sample] = []
    val_set: list[Sample] = []
    skipped = 0

    for index, path in enumerate(files, start=1):
        file_train, file_val, file_skipped = _split_file_by_worker(str(path), val_ids)
        train_set.extend(file_train)
        val_set.extend(file_val)
        skipped += file_skipped
        if index % 500 == 0 or index == len(files):
            print(
                f"[data]   split {index}/{len(files)} files, "
                f"train={len(train_set):,} val={len(val_set):,}",
                flush=True,
            )

    return train_set, val_set, skipped


def _split_files_parallel(
    files: list[Path],
    val_ids: set[int],
    *,
    load_workers: int,
) -> tuple[list[Sample], list[Sample], int]:
    print(
        f"[data] parallel split: {len(files)} file(s), {load_workers} worker(s)",
        flush=True,
    )
    train_set: list[Sample] = []
    val_set: list[Sample] = []
    skipped = 0
    completed = 0

    with ProcessPoolExecutor(max_workers=load_workers) as pool:
        futures = [pool.submit(_split_file_by_worker, str(path), val_ids) for path in files]
        for future in as_completed(futures):
            file_train, file_val, file_skipped = future.result()
            train_set.extend(file_train)
            val_set.extend(file_val)
            skipped += file_skipped
            completed += 1
            if completed % 500 == 0 or completed == len(files):
                print(
                    f"[data]   split {completed}/{len(files)} files, "
                    f"train={len(train_set):,} val={len(val_set):,}",
                    flush=True,
                )

    return train_set, val_set, skipped


def split_samples(
    source: str | Path,
    pattern: str,
    val_workers: list[int],
    *,
    load_workers: int = 0,
) -> tuple[list[Sample], list[Sample], int]:
    root = Path(source)
    val_ids = set(val_workers)
    files = sorted(Path(fp) for fp in glob.glob(str(root / pattern)))
    if not files:
        return [], [], 0

    workers = _resolve_load_workers(load_workers)
    if workers <= 1 or len(files) <= 1:
        return _split_files_serial(files, val_ids)
    return _split_files_parallel(files, val_ids, load_workers=workers)


def split_samples_by_ratio(
    source: str | Path,
    pattern: str,
    val_ratio: float,
    *,
    seed: int = 42,
    load_workers: int = 0,
) -> tuple[list[Sample], list[Sample], int]:
    if not 0.0 < val_ratio < 1.0:
        raise ValueError(f"val_ratio must be in (0, 1), got {val_ratio}")

    print(f"[data] val_ratio={val_ratio}  seed={seed}", flush=True)
    samples, skipped = load_samples(source, pattern, load_workers=load_workers)
    if not samples:
        raise ValueError(f"No samples found under {source} with pattern {pattern!r}")

    print(f"[data] shuffling {len(samples):,} samples ...", flush=True)
    rng = random.Random(seed)
    indices = list(range(len(samples)))
    rng.shuffle(indices)
    n_val = max(1, int(len(samples) * val_ratio))
    val_indices = set(indices[:n_val])

    print(f"[data] splitting train/val (val={n_val:,}) ...", flush=True)
    train_set = [samples[i] for i in range(len(samples)) if i not in val_indices]
    val_set = [samples[i] for i in range(len(samples)) if i in val_indices]
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
