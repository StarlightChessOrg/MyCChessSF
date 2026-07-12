"""Static (quiet) position filter for NNUE training samples."""
from __future__ import annotations

import os
import sys
from concurrent.futures import ProcessPoolExecutor, as_completed
from dataclasses import dataclass
from functools import lru_cache
from pathlib import Path

from labels import DEFAULT_MATE_THRESHOLD, is_mate_label

_REPO_ROOT = Path(__file__).resolve().parents[1]
MAX_AUTO_IO_WORKERS = 8


def _ensure_xqwlight_on_path() -> None:
    repo = str(_REPO_ROOT)
    if repo not in sys.path:
        sys.path.insert(0, repo)


@lru_cache(maxsize=1)
def _position_factory():
    _ensure_xqwlight_on_path()
    try:
        from xqwlight_core import Position
    except ImportError:
        return None
    return Position


@dataclass(frozen=True)
class StaticFilterStats:
    before: int = 0
    after: int = 0
    mate_zone: int = 0
    in_check: int = 0
    pst_unstable: int = 0
    bad_fen: int = 0

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


def is_static_position(
    fen: str,
    vl: float,
    pos: object,
    *,
    mate_threshold: float = DEFAULT_MATE_THRESHOLD,
    pst_margin: float = 70.0,
) -> tuple[bool, str]:
    """
    Return ``(keep, reason)`` for a training sample.

    Static positions:
    - not in mate / win band (|vl| < mate_threshold)
    - side to move not in check
    - search label close to PST: |vl - PST| <= pst_margin
    """
    if is_mate_label(vl, threshold=mate_threshold):
        return False, "mate_zone"

    if not pos.set_fen(fen):
        return False, "bad_fen"

    if bool(pos.in_check()):
        return False, "in_check"

    pst = int(pos.evaluate())
    if abs(float(vl) - float(pst)) > pst_margin:
        return False, "pst_unstable"

    return True, "ok"


def _filter_static_chunk(
    chunk: list[tuple[str, float]],
    *,
    mate_threshold: float,
    pst_margin: float,
) -> tuple[list[tuple[str, float]], dict[str, int]]:
    factory = _position_factory()
    if factory is None:
        raise ImportError("xqwlight_core unavailable in worker process")
    pos = factory()
    kept: list[tuple[str, float]] = []
    counts = {
        "mate_zone": 0,
        "in_check": 0,
        "pst_unstable": 0,
        "bad_fen": 0,
    }
    for fen, vl in chunk:
        ok, reason = is_static_position(
            fen,
            vl,
            pos,
            mate_threshold=mate_threshold,
            pst_margin=pst_margin,
        )
        if ok:
            kept.append((fen, vl))
        else:
            counts[reason] += 1
    return kept, counts


def filter_static_samples(
    samples: list,
    *,
    mate_threshold: float = DEFAULT_MATE_THRESHOLD,
    pst_margin: float = 70.0,
    load_workers: object = "auto",
    require_engine: bool = True,
) -> tuple[list, StaticFilterStats]:
    """
    Keep only static-position samples.

    Requires ``xqwlight_core`` (build from repo root, extension importable on ``sys.path``).
    """
    from data.dataset import Sample

    stats = StaticFilterStats(before=len(samples))
    if not samples:
        return samples, stats

    factory = _position_factory()
    if factory is None:
        if require_engine:
            raise ImportError(
                "quiet_only requires xqwlight_core (build via scripts/build_linux.sh "
                "and ensure the extension is importable from the repo root)."
            )
        return samples, stats

    workers, workers_label = _parse_workers(load_workers)
    tuples = [(s.fen, s.vl) for s in samples]
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
        counts = {"mate_zone": 0, "in_check": 0, "pst_unstable": 0, "bad_fen": 0}
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
    stats.bad_fen = counts["bad_fen"]
    kept = [Sample(fen=fen, vl=vl) for fen, vl in kept_tuples]
    return kept, stats


def format_static_filter_stats(stats: StaticFilterStats) -> str:
    return (
        f"{stats.before:,} -> {stats.after:,} "
        f"(removed {stats.removed:,}: mate={stats.mate_zone:,}, check={stats.in_check:,}, "
        f"pst={stats.pst_unstable:,}, bad_fen={stats.bad_fen:,})"
    )
