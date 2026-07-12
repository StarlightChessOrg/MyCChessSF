"""Static (quiet) position filter for NNUE training samples."""
from __future__ import annotations

import os
from concurrent.futures import ProcessPoolExecutor, as_completed
from dataclasses import dataclass

from labels import DEFAULT_MATE_THRESHOLD, is_mate_label

MAX_AUTO_IO_WORKERS = 8


@dataclass(frozen=True)
class StaticFilterStats:
    before: int = 0
    after: int = 0
    mate_zone: int = 0
    in_check: int = 0
    pst_unstable: int = 0
    no_metadata: int = 0

    @property
    def removed(self) -> int:
        return self.before - self.after


def _parse_workers(value: object) -> tuple[int, str]:
    cpu = os.cpu_count() or 1
    if isinstance(value, str) and value.lower() == "auto":
        count = min(MAX_AUTO_IO_WORKERS, cpu)
        return count, f"auto ({count}, cpu={cpu})"
    if value is None:
        count = min(MAX_AUTO_IO_WORKERS, cpu)
        return count, f"auto ({count}, cpu={cpu})"
    count = max(1, int(value))
    return count, str(count)


def is_static_sample(
    sample: object,
    *,
    mate_threshold: float = DEFAULT_MATE_THRESHOLD,
    pst_margin: float = 70.0,
) -> tuple[bool, str]:
    """Return ``(keep, reason)`` using PST/in_check columns from nnue_data."""
    if sample.pst is None or sample.in_check is None:
        return False, "no_metadata"

    if is_mate_label(sample.vl, threshold=mate_threshold):
        return False, "mate_zone"

    if sample.in_check:
        return False, "in_check"

    if abs(float(sample.vl) - float(sample.pst)) > pst_margin:
        return False, "pst_unstable"

    return True, "ok"


def _filter_static_chunk(
    chunk: list[tuple[str, float, float | None, bool | None]],
    *,
    mate_threshold: float,
    pst_margin: float,
) -> tuple[list[tuple[str, float, float | None, bool | None]], dict[str, int]]:
    kept: list[tuple[str, float, float | None, bool | None]] = []
    counts = {
        "mate_zone": 0,
        "in_check": 0,
        "pst_unstable": 0,
        "no_metadata": 0,
    }

    class _Row:
        __slots__ = ("fen", "vl", "pst", "in_check")

        def __init__(self, fen: str, vl: float, pst: float | None, in_check: bool | None) -> None:
            self.fen = fen
            self.vl = vl
            self.pst = pst
            self.in_check = in_check

    for fen, vl, pst, in_check in chunk:
        row = _Row(fen, vl, pst, in_check)
        ok, reason = is_static_sample(
            row,
            mate_threshold=mate_threshold,
            pst_margin=pst_margin,
        )
        if ok:
            kept.append((fen, vl, pst, in_check))
        else:
            counts[reason] += 1
    return kept, counts


def _require_metadata_columns(samples: list) -> None:
    missing = sum(1 for s in samples if s.pst is None or s.in_check is None)
    if missing:
        raise ValueError(
            f"quiet_only requires 4-column nnue_data (FEN\\tsearch\\tpst\\tin_check); "
            f"{missing:,}/{len(samples):,} rows missing PST or in_check. "
            f"Augment legacy files in WSL:\n"
            f"  bash scripts/augment_merged_wsl.sh /path/to/merged.txt nnue_data/merged.txt\n"
            f"See docs/nnue-data-generation.md and README.md."
        )


def filter_static_samples(
    samples: list,
    *,
    mate_threshold: float = DEFAULT_MATE_THRESHOLD,
    pst_margin: float = 70.0,
    load_workers: object = "auto",
) -> tuple[list, StaticFilterStats]:
    """Keep only static-position samples using PST/in_check from data columns."""
    from data.dataset import Sample

    stats = StaticFilterStats(before=len(samples))
    if not samples:
        return samples, stats

    _require_metadata_columns(samples)

    workers, workers_label = _parse_workers(load_workers)
    tuples = [(s.fen, s.vl, s.pst, s.in_check) for s in samples]
    print(
        f"[data] static filter: {stats.before:,} samples (workers={workers_label}, "
        f"pst_margin={pst_margin:.0f}) ...",
        flush=True,
    )

    if workers <= 1 or len(tuples) < 50_000:
        kept_tuples, counts = _filter_static_chunk(
            tuples,
            mate_threshold=mate_threshold,
            pst_margin=pst_margin,
        )
    else:
        chunk_size = max(25_000, len(tuples) // (workers * 4))
        chunks = [tuples[i : i + chunk_size] for i in range(0, len(tuples), chunk_size)]
        print(
            f"[data]   static filter parallel: {len(chunks)} chunk(s), {workers} worker(s)",
            flush=True,
        )
        kept_tuples = []
        counts = {
            "mate_zone": 0,
            "in_check": 0,
            "pst_unstable": 0,
            "no_metadata": 0,
        }
        with ProcessPoolExecutor(max_workers=workers) as pool:
            futures = [
                pool.submit(
                    _filter_static_chunk,
                    chunk,
                    mate_threshold=mate_threshold,
                    pst_margin=pst_margin,
                )
                for chunk in chunks
            ]
            for completed, future in enumerate(as_completed(futures), start=1):
                chunk_kept, chunk_counts = future.result()
                kept_tuples.extend(chunk_kept)
                for key in counts:
                    counts[key] += chunk_counts[key]
                print(
                    f"[data]   static filter chunk {completed}/{len(chunks)}  "
                    f"kept={len(kept_tuples):,}/{stats.before:,}",
                    flush=True,
                )

    stats.after = len(kept_tuples)
    stats.mate_zone = counts["mate_zone"]
    stats.in_check = counts["in_check"]
    stats.pst_unstable = counts["pst_unstable"]
    stats.no_metadata = counts["no_metadata"]
    kept = [
        Sample(fen=fen, vl=vl, pst=pst, in_check=in_check)
        for fen, vl, pst, in_check in kept_tuples
    ]
    return kept, stats


def format_static_filter_stats(stats: StaticFilterStats) -> str:
    parts = [
        f"{stats.before:,} -> {stats.after:,}",
        f"removed {stats.removed:,}",
        f"mate={stats.mate_zone:,}",
        f"check={stats.in_check:,}",
        f"pst={stats.pst_unstable:,}",
    ]
    if stats.no_metadata:
        parts.append(f"no_metadata={stats.no_metadata:,}")
    return " ".join(parts)
