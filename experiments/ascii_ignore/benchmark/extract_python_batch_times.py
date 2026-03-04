#!/usr/bin/env python3

import argparse
import csv
import subprocess
from pathlib import Path


def _parse_ns(raw: str) -> int:
    cleaned = raw.strip().replace(",", "")
    if not cleaned:
        return 0
    return int(float(cleaned))


def _parse_count(raw: str) -> int:
    cleaned = raw.strip().replace(",", "")
    if not cleaned:
        return 0
    return int(float(cleaned))


def extract_times(csv_path: Path) -> dict[str, int]:
    metrics = {
        "read_python_batch_total_ns": 0,
        "read_python_batch_instances": 0,
        "write_python_batch_total_ns": 0,
        "write_python_batch_instances": 0,
        "to_arrow_host_total_ns": 0,
        "to_arrow_host_instances": 0,
        "from_arrow_total_ns": 0,
        "from_arrow_instances": 0,
    }

    with csv_path.open("r", encoding="utf-8", newline="") as f:
        reader = csv.DictReader(f)
        for row in reader:
            range_name = row.get("Range", "").strip().lower()
            normalized_range = range_name.replace("::", ":")
            total_time_ns = _parse_ns(row.get("Total Time (ns)", "0"))
            instances = _parse_count(row.get("Instances", "0"))

            if range_name.endswith("read python batch"):
                metrics["read_python_batch_total_ns"] += total_time_ns
                metrics["read_python_batch_instances"] += instances
            elif range_name.endswith("write python batch"):
                metrics["write_python_batch_total_ns"] += total_time_ns
                metrics["write_python_batch_instances"] += instances

            if normalized_range.endswith("to_arrow_host"):
                metrics["to_arrow_host_total_ns"] += total_time_ns
                metrics["to_arrow_host_instances"] += instances

            if normalized_range.endswith("libcudf:from_arrow"):
                metrics["from_arrow_total_ns"] += total_time_ns
                metrics["from_arrow_instances"] += instances

    return metrics


def generate_nvtx_sum_csv(nsys_rep_path: Path, output_path: Path) -> Path:
    cmd = [
        "nsys",
        "stats",
        "--report",
        "nvtx_sum",
        "--format",
        "csv",
        "-o",
        str(output_path),
        str(nsys_rep_path),
    ]
    subprocess.run(cmd, check=True)

    candidates = [
        output_path.parent / f"{output_path.name}_nvtx_sum.csv",
        output_path.with_suffix(".csv"),
        output_path,
    ]
    for candidate in candidates:
        if candidate.exists():
            return candidate

    raise FileNotFoundError(
        f"Could not find generated nvtx_sum CSV near output path: {output_path}"
    )


def main() -> None:
    parser = argparse.ArgumentParser(
        description=(
            "Extract read/write python batch times from nvtx_sum CSV, "
            "or generate the CSV from an .nsys-rep first."
        )
    )
    parser.add_argument(
        "input_path",
        type=Path,
        help="Path to nvtx_sum CSV file or .nsys-rep file",
    )
    parser.add_argument(
        "-o",
        "--output-path",
        type=Path,
        help="Output path prefix for 'nsys stats' (required for .nsys-rep input)",
    )
    args = parser.parse_args()

    input_path = args.input_path.resolve()
    if not input_path.exists():
        raise FileNotFoundError(f"Input file not found: {input_path}")

    if input_path.suffix == ".csv":
        csv_path = input_path
    elif input_path.suffix == ".nsys-rep":
        if args.output_path is None:
            parser.error("--output-path is required when input_path is an .nsys-rep file")
        csv_path = generate_nvtx_sum_csv(
            nsys_rep_path=input_path, output_path=args.output_path.resolve()
        )
    else:
        parser.error(
            "input_path must be either a .csv file or an .nsys-rep file"
        )

    metrics = extract_times(csv_path)
    print(f"write python batch (s): {metrics['write_python_batch_total_ns'] / 1e9:.6f}")
    print(f"write python batch instances: {metrics['write_python_batch_instances']}")
    print()
    print(f"read python batch (s):  {metrics['read_python_batch_total_ns'] / 1e9:.6f}")
    print(f"read python batch instances:  {metrics['read_python_batch_instances']}")
    print()
    print(f"to_arrow_host (s):      {metrics['to_arrow_host_total_ns'] / 1e9:.6f}")
    print(f"to_arrow_host instances:      {metrics['to_arrow_host_instances']}")
    print()
    print(f"libcudf::from_arrow (s): {metrics['from_arrow_total_ns'] / 1e9:.6f}")
    print(f"libcudf::from_arrow instances: {metrics['from_arrow_instances']}")


if __name__ == "__main__":
    main()
