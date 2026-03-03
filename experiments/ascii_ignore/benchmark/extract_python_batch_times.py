#!/usr/bin/env python3

import argparse
import csv
from pathlib import Path


def _parse_ns(raw: str) -> int:
    cleaned = raw.strip().replace(",", "")
    if not cleaned:
        return 0
    return int(float(cleaned))


def extract_times(csv_path: Path) -> tuple[int, int]:
    read_total_ns = 0
    write_total_ns = 0

    with csv_path.open("r", encoding="utf-8", newline="") as f:
        reader = csv.DictReader(f)
        for row in reader:
            range_name = row.get("Range", "").strip().lower()
            total_time_ns = _parse_ns(row.get("Total Time (ns)", "0"))

            if range_name.endswith("read python batch"):
                read_total_ns += total_time_ns
            elif range_name.endswith("write python batch"):
                write_total_ns += total_time_ns

    return read_total_ns, write_total_ns


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Extract read/write python batch times from nvtx_sum CSV."
    )
    parser.add_argument("csv_path", type=Path, help="Path to nvtx_sum CSV file")
    args = parser.parse_args()

    read_ns, write_ns = extract_times(args.csv_path)
    print(f"read python batch (s):  {read_ns / 1e9:.2f}")
    print(f"write python batch (s): {write_ns / 1e9:.2f}")


if __name__ == "__main__":
    main()
