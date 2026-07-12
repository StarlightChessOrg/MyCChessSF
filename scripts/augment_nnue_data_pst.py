#!/usr/bin/env python3
"""Augment legacy ``FEN\\tvl`` nnue_data with PST and in_check columns.

Output format::

    FEN\\tsearch_vl\\tpst_vl\\tin_check

Requires ``xqwlight_core`` (build via ``scripts/build_linux.sh``). Intended for WSL/Linux.
"""
from __future__ import annotations

import argparse
import os
import sys
from concurrent.futures import ProcessPoolExecutor, as_completed
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

MAX_AUTO_WORKERS = 8


def _parse_workers(value: str) -> tuple[int, str]:
    cpu = os.cpu_count() or 1
    if value.lower() == "auto":
        count = min(MAX_AUTO_WORKERS, cpu)
        return count, f"auto ({count}, cpu={cpu})"
    count = max(1, int(value))
    return count, str(count)


def _position_factory():
    try:
        from xqwlight_core import Position
    except ImportError as exc:
        raise ImportError(
            "xqwlight_core not found. Run from repo root after scripts/build_linux.sh."
        ) from exc
    return Position


def _augment_line(line: str, pos: object) -> str | None:
    line = line.rstrip("\n\r")
    if not line.strip():
        return None
    parts = line.split("\t")
    if len(parts) >= 4:
        return line
    if len(parts) != 2:
        return None
    fen, vl_str = parts
    if not pos.set_fen(fen):
        return None
    pst = int(pos.evaluate())
    in_check = 1 if bool(pos.in_check()) else 0
    return f"{fen}\t{vl_str}\t{pst}\t{in_check}"


def _augment_text_block(text: str) -> tuple[str, dict[str, int]]:
    pos = _position_factory()()
    out_lines: list[str] = []
    stats = {"kept": 0, "skipped": 0, "already": 0}
    for line in text.splitlines():
        if not line.strip():
            continue
        parts = line.split("\t")
        if len(parts) >= 4:
            out_lines.append(line)
            stats["already"] += 1
            stats["kept"] += 1
            continue
        augmented = _augment_line(line, pos)
        if augmented is None:
            stats["skipped"] += 1
            continue
        out_lines.append(augmented)
        stats["kept"] += 1
    return "\n".join(out_lines) + ("\n" if out_lines else ""), stats


def _read_byte_range(path: Path, start: int, end: int) -> str:
    with open(path, "rb") as fp:
        fp.seek(start)
        data = fp.read(end - start)
    text = data.decode("utf-8", errors="replace")
    if start > 0:
        nl = text.find("\n")
        text = text[nl + 1 :] if nl >= 0 else ""
    if end < path.stat().st_size:
        nl = text.rfind("\n")
        text = text[: nl + 1] if nl >= 0 else ""
    return text


def _byte_ranges(path: Path, workers: int) -> list[tuple[int, int]]:
    size = path.stat().st_size
    if workers <= 1 or size < 4 * 1024 * 1024:
        return [(0, size)]
    chunk = size // workers
    ranges: list[tuple[int, int]] = []
    start = 0
    for index in range(workers):
        end = size if index == workers - 1 else start + chunk
        ranges.append((start, end))
        start = end
    return ranges


def augment_file(
    input_path: Path,
    output_path: Path,
    *,
    workers: object = "auto",
) -> dict[str, int]:
    if not input_path.is_file():
        raise FileNotFoundError(input_path)

    worker_count, worker_label = _parse_workers(str(workers))
    ranges = _byte_ranges(input_path, worker_count)
    print(
        f"[augment] input={input_path}  output={output_path}  "
        f"workers={worker_label}  chunks={len(ranges)}",
        flush=True,
    )

    output_path.parent.mkdir(parents=True, exist_ok=True)
    tmp_path = output_path.with_suffix(output_path.suffix + ".tmp")

    totals = {"kept": 0, "skipped": 0, "already": 0, "lines": 0}

    if len(ranges) == 1:
        text = _read_byte_range(input_path, ranges[0][0], ranges[0][1])
        block, stats = _augment_text_block(text)
        with open(tmp_path, "w", encoding="utf-8", newline="\n") as fp:
            fp.write(block)
        for key in ("kept", "skipped", "already"):
            totals[key] += stats[key]
        totals["lines"] = stats["kept"] + stats["skipped"]
    else:
        ordered_blocks: list[str | None] = [None] * len(ranges)
        with ProcessPoolExecutor(max_workers=worker_count) as pool:
            futures = {
                pool.submit(_read_byte_range, input_path, start, end): index
                for index, (start, end) in enumerate(ranges)
            }
            read_blocks: dict[int, str] = {}
            for future in as_completed(futures):
                index = futures[future]
                read_blocks[index] = future.result()

            augment_futures = {
                pool.submit(_augment_text_block, read_blocks[index]): index
                for index in sorted(read_blocks)
            }
            for completed, future in enumerate(as_completed(augment_futures), start=1):
                index = augment_futures[future]
                block, stats = future.result()
                ordered_blocks[index] = block
                for key in ("kept", "skipped", "already"):
                    totals[key] += stats[key]
                print(
                    f"[augment]   chunk {completed}/{len(ranges)}  kept={totals['kept']:,}",
                    flush=True,
                )

        with open(tmp_path, "w", encoding="utf-8", newline="\n") as fp:
            for block in ordered_blocks:
                if block:
                    fp.write(block)

    tmp_path.replace(output_path)
    print(
        f"[augment] done: kept={totals['kept']:,}  skipped={totals['skipped']:,}  "
        f"already_4col={totals['already']:,}  -> {output_path}",
        flush=True,
    )
    return totals


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--input",
        type=Path,
        required=True,
        help="Legacy FEN\\tvl file (e.g. merged.txt)",
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=None,
        help="Output path (default: overwrite --input via temp file)",
    )
    parser.add_argument(
        "--workers",
        default="auto",
        help="Parallel workers (default: auto, max 8)",
    )
    args = parser.parse_args()

    input_path = args.input.resolve()
    output_path = args.output.resolve() if args.output else input_path
    augment_file(input_path, output_path, workers=args.workers)


if __name__ == "__main__":
    main()
