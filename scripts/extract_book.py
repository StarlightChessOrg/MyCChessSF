#!/usr/bin/env python3
"""Extract BOOK.DAT from data/compressed_files/book.7z into deployment/db/."""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
DEFAULT_ARCHIVE = ROOT / "data" / "compressed_files" / "book.7z"
DEFAULT_OUTPUT_DIR = ROOT / "deployment" / "db"


def extract_book(*, archive: Path, output_dir: Path) -> Path:
    if not archive.is_file():
        raise FileNotFoundError(f"archive not found: {archive}")
    try:
        import py7zr
    except ImportError as exc:
        raise RuntimeError("pip install py7zr") from exc

    output_dir.mkdir(parents=True, exist_ok=True)
    with py7zr.SevenZipFile(archive, "r") as z:
        names = z.getnames()
        if "BOOK.DAT" not in names:
            raise ValueError(f"BOOK.DAT missing in {archive.name}: {names}")
        z.extract(targets=["BOOK.DAT"], path=output_dir)

    out = output_dir / "BOOK.DAT"
    n_entries = out.stat().st_size // 8
    print(f"[extract_book] {archive} -> {out}  ({out.stat().st_size:,} bytes, ~{n_entries:,} entries)")
    return out


def main() -> None:
    parser = argparse.ArgumentParser(description="Extract BOOK.DAT for deployment")
    parser.add_argument("--archive", type=Path, default=DEFAULT_ARCHIVE)
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT_DIR)
    args = parser.parse_args()
    try:
        extract_book(archive=args.archive, output_dir=args.output_dir)
    except (FileNotFoundError, ValueError, RuntimeError) as e:
        print(f"[extract_book] {e}", file=sys.stderr)
        sys.exit(1)


if __name__ == "__main__":
    main()
