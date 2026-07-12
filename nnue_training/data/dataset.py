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
from typing import TYPE_CHECKING, Any

import numpy as np

if TYPE_CHECKING:
    from torch.utils.data import DataLoader

_REPO_ROOT = Path(__file__).resolve().parents[2]
if str(_REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(_REPO_ROOT))

from features.xqwl_psq import fen_to_feature_indices
from labels import DEFAULT_MATE_THRESHOLD, is_mate_label
from mycchess_sf.fen_parse import parse_fen_board

DEFAULT_DATA_PATTERN = "worker_*/chunk_*.txt"
MAX_AUTO_IO_WORKERS = 8
_WORKER_DIR_RE = re.compile(r"^worker_(\d+)$")


@dataclass(frozen=True)
class Sample:
    fen: str
    vl: float
    feat_indices: np.ndarray | None = None


def fen_dedupe_key(fen: str) -> str:
    """Position key for dedup: board + side to move (ignores move counters)."""
    parts = fen.strip().split()
    if len(parts) < 2:
        return fen.strip()
    return f"{parts[0]} {parts[1]}"


def dedupe_samples_by_fen(samples: list[Sample]) -> tuple[list[Sample], dict[str, int]]:
    """Merge duplicate positions; keep the median ``vl`` per FEN key."""
    if not samples:
        return [], {"before": 0, "after": 0, "removed": 0, "groups_merged": 0}

    groups: dict[str, tuple[str, list[float]]] = {}
    for sample in samples:
        key = fen_dedupe_key(sample.fen)
        if key not in groups:
            groups[key] = (sample.fen, [sample.vl])
        else:
            fen, values = groups[key]
            values.append(sample.vl)

    deduped: list[Sample] = []
    groups_merged = 0
    for fen, values in groups.values():
        if len(values) > 1:
            groups_merged += 1
        deduped.append(Sample(fen=fen, vl=float(np.median(values))))

    before = len(samples)
    after = len(deduped)
    return deduped, {
        "before": before,
        "after": after,
        "removed": before - after,
        "groups_merged": groups_merged,
    }


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


def parse_worker_count(
    value: object,
    *,
    default_auto: bool = True,
    max_auto: int = MAX_AUTO_IO_WORKERS,
) -> tuple[int, str]:
    """Parse IO worker config. ``auto`` caps at ``max_auto`` to avoid Windows spawn OOM."""
    cpu = os.cpu_count() or 1

    if isinstance(value, str) and value.lower() == "auto":
        count = min(max_auto, cpu)
        return count, f"auto ({count}, cpu={cpu})"

    if value is None:
        if default_auto:
            count = min(max_auto, cpu)
            return count, f"auto ({count}, cpu={cpu})"
        return 1, "1"

    count = int(value)
    if count <= 0:
        resolved = min(max_auto, cpu)
        return resolved, f"auto ({resolved}, cpu={cpu})"
    return count, str(count)


def resolve_train_workers(
    value: object,
    *,
    use_cuda: bool,
    precomputed: bool,
) -> tuple[int, str]:
    """DataLoader workers. Large precomputed datasets default to 0 to avoid pickling GB of RAM."""
    if isinstance(value, str) and value.lower() == "auto":
        if precomputed:
            return 0, "auto (0, precomputed in-memory)"
        if use_cuda:
            count = min(8, os.cpu_count() or 1)
            return count, f"auto ({count})"
        return 0, "auto (0)"

    if value is None:
        value = 0

    count = int(value)
    if count <= 0:
        if precomputed:
            return 0, "0 (precomputed in-memory)"
        if use_cuda:
            resolved = min(8, os.cpu_count() or 1)
            return resolved, str(resolved)
        return 0, "0"
    return count, str(count)


def _resolve_load_workers(load_workers: int) -> int:
    count, _ = parse_worker_count(load_workers)
    return count


def _featurize_sample(sample: Sample) -> Sample:
    return Sample(
        fen=sample.fen,
        vl=sample.vl,
        feat_indices=fen_to_feature_indices(sample.fen),
    )


def _featurize_chunk(samples: list[Sample]) -> list[Sample]:
    return [_featurize_sample(sample) for sample in samples]


def precompute_features(
    samples: list[Sample],
    *,
    load_workers: object = "auto",
) -> list[Sample]:
    if not samples or samples[0].feat_indices is not None:
        return samples

    workers, workers_label = parse_worker_count(load_workers)
    print(
        f"[data] precomputing PSQ features for {len(samples):,} samples "
        f"(workers={workers_label}) ...",
        flush=True,
    )

    if workers <= 1:
        from tqdm import tqdm

        return [_featurize_sample(sample) for sample in tqdm(samples, desc="features", unit="sample")]

    chunk_size = max(2000, len(samples) // (workers * 8))
    chunks = [samples[i : i + chunk_size] for i in range(0, len(samples), chunk_size)]
    print(
        f"[data] feature precompute: {workers} worker(s), {len(chunks)} chunk(s)",
        flush=True,
    )

    result: list[Sample] = []
    completed = 0
    with ProcessPoolExecutor(max_workers=workers) as pool:
        futures = [pool.submit(_featurize_chunk, chunk) for chunk in chunks]
        for future in as_completed(futures):
            result.extend(future.result())
            completed += 1
            print(
                f"[data]   feature chunk {completed}/{len(chunks)} done, "
                f"{len(result):,}/{len(samples):,} samples",
                flush=True,
            )
    return result


def pack_precomputed_dataset(
    samples: list[Sample],
    *,
    label_mean: float,
    label_std: float,
    mate_threshold: float = DEFAULT_MATE_THRESHOLD,
) -> "PrecomputedNnueDataset":
    if not samples:
        raise ValueError("Cannot pack empty sample list")

    missing = sum(1 for sample in samples[: min(8, len(samples))] if sample.feat_indices is None)
    if missing:
        raise ValueError("pack_precomputed_dataset requires precomputed feat_indices")

    n_samples = len(samples)
    total_features = sum(len(sample.feat_indices) for sample in samples)
    print(
        f"[data] packing {n_samples:,} samples into CSR arrays "
        f"({total_features:,} feature indices) ...",
        flush=True,
    )

    feat_indices = np.empty(total_features, dtype=np.int32)
    feat_offsets = np.empty(n_samples + 1, dtype=np.int64)
    targets_norm = np.empty(n_samples, dtype=np.float32)
    targets_raw = np.empty(n_samples, dtype=np.float32)
    is_mate = np.empty(n_samples, dtype=np.bool_)

    pos = 0
    for i, sample in enumerate(samples):
        indices = sample.feat_indices
        assert indices is not None
        length = len(indices)
        feat_offsets[i] = pos
        feat_indices[pos : pos + length] = indices
        pos += length
        targets_norm[i] = (sample.vl - label_mean) / label_std
        targets_raw[i] = sample.vl
        is_mate[i] = is_mate_label(sample.vl, threshold=mate_threshold)
    feat_offsets[n_samples] = pos

    print(
        f"[data] pack complete: indices={feat_indices.nbytes / 1024 / 1024:.1f} MiB, "
        f"offsets={feat_offsets.nbytes / 1024:.1f} KiB, "
        f"mate={int(is_mate.sum()):,} quiet={int((~is_mate).sum()):,}",
        flush=True,
    )
    return PrecomputedNnueDataset(
        feat_indices=feat_indices,
        feat_offsets=feat_offsets,
        targets_norm=targets_norm,
        targets_raw=targets_raw,
        is_mate=is_mate,
        mate_threshold=mate_threshold,
    )


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
    load_workers: object = "auto",
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
    load_workers: object = "auto",
    dedupe_fen: bool = True,
) -> tuple[list[Sample], list[Sample], int]:
    root = Path(source)
    val_ids = set(val_workers)
    files = sorted(Path(fp) for fp in glob.glob(str(root / pattern)))
    if not files:
        return [], [], 0

    workers = _resolve_load_workers(load_workers)
    if workers <= 1 or len(files) <= 1:
        train_set, val_set, skipped = _split_files_serial(files, val_ids)
    else:
        train_set, val_set, skipped = _split_files_parallel(files, val_ids, load_workers=workers)

    if dedupe_fen:
        train_set, train_stats = dedupe_samples_by_fen(train_set)
        val_set, val_stats = dedupe_samples_by_fen(val_set)
        print(
            f"[data] dedupe train: {train_stats['before']:,} -> {train_stats['after']:,} "
            f"(removed {train_stats['removed']:,}, merged {train_stats['groups_merged']:,})",
            flush=True,
        )
        print(
            f"[data] dedupe val:   {val_stats['before']:,} -> {val_stats['after']:,} "
            f"(removed {val_stats['removed']:,}, merged {val_stats['groups_merged']:,})",
            flush=True,
        )

    return train_set, val_set, skipped


def split_samples_by_ratio(
    source: str | Path,
    pattern: str,
    val_ratio: float,
    *,
    seed: int = 42,
    load_workers: object = "auto",
    dedupe_fen: bool = True,
) -> tuple[list[Sample], list[Sample], int]:
    if not 0.0 < val_ratio < 1.0:
        raise ValueError(f"val_ratio must be in (0, 1), got {val_ratio}")

    print(f"[data] val_ratio={val_ratio}  seed={seed}  dedupe_fen={dedupe_fen}", flush=True)
    samples, skipped = load_samples(source, pattern, load_workers=load_workers)
    if not samples:
        raise ValueError(f"No samples found under {source} with pattern {pattern!r}")

    if dedupe_fen:
        samples, stats = dedupe_samples_by_fen(samples)
        print(
            f"[data] dedupe: {stats['before']:,} -> {stats['after']:,} unique FEN "
            f"(removed {stats['removed']:,}, merged {stats['groups_merged']:,} groups)",
            flush=True,
        )

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


def compute_zscore_stats(
    samples: list[Sample],
    *,
    exclude_mate: bool = True,
    mate_threshold: float = DEFAULT_MATE_THRESHOLD,
) -> tuple[float, float]:
    if not samples:
        raise ValueError("Cannot compute z-score stats on empty sample list")
    values = np.array([s.vl for s in samples], dtype=np.float64)
    if exclude_mate:
        quiet = np.array([not is_mate_label(v, threshold=mate_threshold) for v in values])
        if not quiet.any():
            raise ValueError("All samples are mate-zone labels; cannot compute quiet z-score stats")
        values = values[quiet]
    mean = float(values.mean())
    std = float(values.std())
    if std < 1e-8:
        std = 1.0
    return mean, std


def count_mate_labels(
    samples: list[Sample],
    *,
    mate_threshold: float = DEFAULT_MATE_THRESHOLD,
) -> tuple[int, int]:
    mate = sum(1 for s in samples if is_mate_label(s.vl, threshold=mate_threshold))
    return mate, len(samples) - mate


class NnueDataset:
    def __init__(
        self,
        samples: list[Sample],
        *,
        label_mean: float,
        label_std: float,
        mate_threshold: float = DEFAULT_MATE_THRESHOLD,
    ) -> None:
        self.samples = samples
        self.label_mean = label_mean
        self.label_std = label_std
        self.mate_threshold = mate_threshold

    def __len__(self) -> int:
        return len(self.samples)

    def __getitem__(self, idx: int) -> tuple[np.ndarray, float, float, bool]:
        s = self.samples[idx]
        indices = s.feat_indices if s.feat_indices is not None else fen_to_feature_indices(s.fen)
        target_norm = (s.vl - self.label_mean) / self.label_std
        mate = is_mate_label(s.vl, threshold=self.mate_threshold)
        return indices, target_norm, s.vl, mate


class PrecomputedNnueDataset:
    """Compact CSR storage for precomputed sparse features (avoids pickling millions of Sample objects)."""

    def __init__(
        self,
        *,
        feat_indices: np.ndarray,
        feat_offsets: np.ndarray,
        targets_norm: np.ndarray,
        targets_raw: np.ndarray,
        is_mate: np.ndarray,
        mate_threshold: float = DEFAULT_MATE_THRESHOLD,
    ) -> None:
        self.feat_indices = feat_indices
        self.feat_offsets = feat_offsets
        self.targets_norm = targets_norm
        self.targets_raw = targets_raw
        self.is_mate = is_mate
        self.mate_threshold = mate_threshold

    def __len__(self) -> int:
        return len(self.targets_norm)

    def __getitem__(self, idx: int) -> tuple[np.ndarray, float, float, bool]:
        start = int(self.feat_offsets[idx])
        end = int(self.feat_offsets[idx + 1])
        return (
            self.feat_indices[start:end],
            float(self.targets_norm[idx]),
            float(self.targets_raw[idx]),
            bool(self.is_mate[idx]),
        )


def collate_fn(batch: list[tuple[np.ndarray, float, float, bool]]) -> tuple[Any, ...]:
    import torch

    feat_arrays = [feat_indices for feat_indices, _, _, _ in batch]
    indices_arr = np.concatenate(feat_arrays) if len(feat_arrays) > 1 else feat_arrays[0]
    offsets = np.empty(len(batch), dtype=np.int64)
    offset = 0
    for i, feat_indices in enumerate(feat_arrays):
        offsets[i] = offset
        offset += len(feat_indices)

    targets_norm = np.array([target_norm for _, target_norm, _, _ in batch], dtype=np.float32)
    targets_raw = np.array([raw for _, _, raw, _ in batch], dtype=np.float32)
    is_mate = np.array([mate for _, _, _, mate in batch], dtype=np.bool_)

    return (
        torch.from_numpy(np.ascontiguousarray(indices_arr, dtype=np.int64)),
        torch.from_numpy(offsets),
        torch.from_numpy(targets_norm),
        torch.from_numpy(targets_raw),
        torch.from_numpy(is_mate),
    )


def make_dataloader(
    dataset: Any,
    *,
    batch_size: int,
    shuffle: bool,
    num_workers: int = 0,
    prefetch_factor: int = 2,
    pin_memory: bool = False,
) -> DataLoader:
    from torch.utils.data import DataLoader

    kwargs: dict = {
        "dataset": dataset,
        "batch_size": batch_size,
        "shuffle": shuffle,
        "num_workers": num_workers,
        "collate_fn": collate_fn,
        "pin_memory": pin_memory,
    }
    if num_workers > 0:
        kwargs["persistent_workers"] = True
        kwargs["prefetch_factor"] = max(1, prefetch_factor)
    return DataLoader(**kwargs)
