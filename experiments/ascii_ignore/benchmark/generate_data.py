#!/usr/bin/env python3
"""Standalone data generation driver for all UDFs.

Usage:
    python generate_data.py --udf ascii_ignore --rows 1000000
    python generate_data.py --udf redact_pii   --rows 1000000
    python generate_data.py --udf metric_status --rows 1000000
"""

import argparse
import importlib
import os
import sys
from pathlib import Path

from pyspark.sql import SparkSession

SCRIPT_DIR = Path(__file__).resolve().parent
PROJECT_DIR = SCRIPT_DIR.parent
SRC_DIR = PROJECT_DIR / "src"
sys.path.insert(0, str(SRC_DIR))

from udf_registry import available_names, get_config  # type: ignore[import-not-found]

GEN_DATA_DIR = SCRIPT_DIR


def _human_readable(n: int) -> str:
    if n >= 1_000_000 and n % 1_000_000 == 0:
        return f"{n // 1_000_000}M"
    if n >= 1_000 and n % 1_000 == 0:
        return f"{n // 1_000}K"
    return str(n)


def main() -> None:
    parser = argparse.ArgumentParser(description="Generate benchmark data for UDF experiments")
    parser.add_argument(
        "--udf",
        required=True,
        choices=available_names(),
        help="UDF to generate data for",
    )
    parser.add_argument(
        "--rows", "-r",
        type=int,
        default=1_000_000,
        help="Number of rows to generate (default: 1M)",
    )
    parser.add_argument(
        "--partitions", "-p",
        type=int,
        default=64,
        help="Number of output partitions (default: 64)",
    )
    parser.add_argument(
        "--output-path", "-o",
        type=str,
        default=None,
        help="Output parquet path (default: data/<udf>_bench_data_<rows>_rows.parquet)",
    )
    parser.add_argument(
        "--spark-conf",
        action="append",
        default=[],
        help="Extra Spark config in key=value format (e.g. --spark-conf spark.jars=/path/to/rapids.jar)",
    )
    args = parser.parse_args()

    extra_spark_configs = {}
    for entry in args.spark_conf:
        if "=" not in entry:
            raise ValueError(f"Invalid --spark-conf '{entry}'. Expected key=value format.")
        key, value = entry.split("=", 1)
        extra_spark_configs[key.strip()] = value

    cfg = get_config(args.udf)
    gen_module_name = f"gen_{cfg.name}_data"
    gen_module_path = GEN_DATA_DIR / f"{gen_module_name}.py"
    if not gen_module_path.exists():
        raise FileNotFoundError(f"Data generation module not found: {gen_module_path}")

    # Import the gen_data module
    sys.path.insert(0, str(GEN_DATA_DIR))
    gen_mod = importlib.import_module(gen_module_name)

    output_path = args.output_path
    if output_path is None:
        output_path = str(
            SCRIPT_DIR / "data" / f"{cfg.name}_bench_data_{_human_readable(args.rows)}_rows.parquet"
        )

    os.makedirs(os.path.dirname(output_path), exist_ok=True)

    print(f"Generating {args.rows:,d} rows for UDF '{cfg.name}'")
    print(f"Output: {output_path}")

    builder = (
        SparkSession.builder
        .appName(f"gen_data_{cfg.name}")
        .master("local[*]")
        .config("spark.driver.memory", "16g")
    )
    for key, value in extra_spark_configs.items():
        builder = builder.config(key, value)
    spark = builder.getOrCreate()

    try:
        df = gen_mod.generate_synthetic_data(
            spark=spark,
            num_rows=args.rows,
            num_partitions=args.partitions,
        )

        df.write.mode("overwrite").parquet(output_path)
        actual = spark.read.parquet(output_path).count()
        print(f"Wrote {actual:,d} rows to {output_path}")
    finally:
        spark.stop()


if __name__ == "__main__":
    main()
