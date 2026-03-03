# SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION.
# SPDX-License-Identifier: Apache-2.0

import argparse
import json
import logging
import os
import sys
import time
import traceback
import shutil
from typing import List, Optional

from cuaether_assistant.constants import (
    BENCH_RUN_ERROR_MSG_FOLLOWED_BY_ERROR_TYPE,
    RUNTIME_MSG_FOLLOWED_BY_TIME,
)
from cuaether_assistant.templates.utils import (
    initialize_spark,
    load_module,
    spark_configs_for_pythonpath,
)
from cuaether_assistant.utils import compute_file_name_without_extension

logging.basicConfig(format="%(message)s", level=logging.INFO)
logger = logging.getLogger(__name__)


class UDFBenchmark:
    def __init__(
        self,
        mode: str,
        udf_name: str,
        result_path: str,
        spark_configs: List[str],
        cli_args: List[str],
        test_module_path: str,
        gen_udf_data_module_path: str,
        udf_class_path: Optional[str] = None,
        sql_file_path: Optional[str] = None,
        pycudf_file_path: Optional[str] = None,
    ):
        self.test_module_path = test_module_path
        self.udf_name = udf_name
        self.udf_class_path = udf_class_path
        self.result_path = result_path
        self.spark_configs = spark_configs
        self.cli_args = cli_args
        self.sql_file_path = sql_file_path
        self.pycudf_file_path = pycudf_file_path

        self.is_jvm_udf = mode == "jvm-udf"
        self.is_sql = mode == "sql"
        self.is_pycudf = mode == "pycudf"
        self.is_pyudf = mode == "pyudf"

        # PYTHONPATH is only needed when Spark runs Python (PyUDF or PyCUDF)
        if self.is_pyudf or self.is_pycudf:
            pythonpath_dirs = [
                os.path.dirname(os.path.abspath(self.test_module_path)),
            ]
            if self.is_pycudf:
                assert self.pycudf_file_path is not None
                pythonpath_dirs.append(
                    os.path.dirname(os.path.abspath(self.pycudf_file_path))
                )
            self.spark_configs = list(
                self.spark_configs
            ) + spark_configs_for_pythonpath(pythonpath_dirs)

        # Initialize Spark session
        self.spark = initialize_spark(self.spark_configs)

        # Load gen_data_helpers
        self.gen_data_helpers = load_module(
            "gen_data_helpers",
            gen_udf_data_module_path,
        )

        # Create output directory
        self.result_dir = os.path.dirname(self.result_path)
        if self.result_dir:
            os.makedirs(self.result_dir, exist_ok=True)

        self.test_module = None

        # SQL benchmark
        if self.is_sql:
            return

        # Load test_module once for JVM UDF, PyCUDF, and PyUDF (all need register_udf)
        self.test_module = load_module(
            "test_module", self.test_module_path, ["register_udf"]
        )

        # Java UDF benchmark
        if self.is_jvm_udf:
            return

        # PySpark GPU UDF benchmark
        if self.is_pycudf:
            # Load GPU UDF from pycudf file for registration (same udf_func attribute as CPU path)
            assert self.pycudf_file_path is not None
            pycudf_module = load_module(self.udf_name, self.pycudf_file_path)
            if not hasattr(pycudf_module, self.udf_name):
                raise ValueError(
                    f"PyCUDF function '{self.udf_name}' not found in file. "
                    f"Make sure the function is defined in {self.pycudf_file_path}"
                )
            self.udf_func = getattr(pycudf_module, self.udf_name)
            return

        # PyUDF (PySpark CPU UDF) benchmark: get UDF function from test module
        if not hasattr(self.test_module, self.udf_name):
            raise ValueError(
                f"Python UDF '{self.udf_name}' not found in test module. "
                f"Make sure the UDF is imported in the test file."
            )
        self.udf_func = getattr(self.test_module, self.udf_name)

    def _run_benchmark(self, data_path: str, output_path: str) -> float:
        """Run the benchmark"""
        # Register UDF based on the four benchmark modes (matching __init__ order)
        if self.is_jvm_udf:
            # Mode 1: Java/Scala UDF - register via test module with class_path
            assert (
                self.test_module is not None
            ), "test_module should be loaded for Java/Scala UDF"
            self.test_module.register_udf(
                spark=self.spark, udf_name=self.udf_name, class_path=self.udf_class_path
            )
        elif self.is_sql:
            # Mode 2: SQL - no UDF registration needed
            pass
        else:
            # Mode 3 & 4: PyCUDF (GPU) or PyUDF (CPU) - register via test module with udf_func
            # (udf_func is set in __init__ from pycudf file for GPU, or from test module for CPU)
            assert (
                self.test_module is not None
            ), "test_module should be loaded for PyCUDF/PyUDF benchmark"
            assert hasattr(
                self, "udf_func"
            ), "udf_func should be loaded for PyCUDF/PyUDF"
            self.test_module.register_udf(
                spark=self.spark, udf_name=self.udf_name, udf_func=self.udf_func
            )

        start_time = time.time()

        # Read parquet
        df = self.spark.read.parquet(data_path)

        if self.is_sql:
            # Execute SQL query
            result_df = self.gen_data_helpers.execute_sql(
                spark=self.spark, sql_file_path=self.sql_file_path, df=df
            )
        else:
            # Execute UDF
            result_df = self.gen_data_helpers.execute_udf(
                spark=self.spark, udf_name=self.udf_name, df=df
            )

        # Write parquet
        result_df.write.mode("overwrite").parquet(output_path)

        end_time = time.time()
        total_time = end_time - start_time

        logger.info(f"{RUNTIME_MSG_FOLLOWED_BY_TIME} {total_time:.2f}")

        return total_time

    def _save_benchmark_report(self, total_time, data_path, error_info=None):
        """Save benchmark results to JSON file"""
        report = {
            "udf_name": self.udf_name,
            "data_path": data_path,
            "status": "error" if error_info else "success",
            "e2e_runtime": total_time,
            "cli_command": {
                "args": self.cli_args[1:] if len(self.cli_args) > 1 else [],
                "full_command": (
                    f"python {' '.join(self.cli_args)}"
                    if self.cli_args
                    else "python bench_runner.py"
                ),
            },
            "spark_configs": dict(self.spark.sparkContext.getConf().getAll()),
        }

        if error_info:
            report["error"] = {
                "error_message": error_info["error_message"],
                "stack_trace": error_info["stack_trace"],
            }

        with open(self.result_path, "w") as f:
            json.dump(report, f, indent=2)

    def _try_save_benchmark_report(
        self, total_time: float, data_path: str, error_info=None
    ) -> None:
        """Try to save benchmark report, logging warnings on failure"""
        try:
            self._save_benchmark_report(total_time, data_path, error_info)
            if error_info:
                logger.info(
                    f"Benchmark failed. Full report with error information saved to {self.result_path}"
                )
            else:
                logger.info(
                    f"Benchmark completed successfully. Full report saved to {self.result_path}"
                )
        except Exception as e:
            logger.warning(f"Failed to save benchmark report: {e}")

    def run(self, data_path: str) -> None:
        """Run complete benchmark for a single UDF/SQL query"""
        logger.info(
            f"\n=== Running '{self.udf_name}' benchmark on dataset {data_path} ==="
        )

        # Create output directory for benchmark parquet files
        data_filename = os.path.splitext(os.path.basename(data_path))[0]
        output_path = os.path.join(
            self.result_dir, f"{self.udf_name.replace('.', '_')}_{data_filename}_output"
        )

        try:
            total_time = self._run_benchmark(data_path, output_path)
            self._try_save_benchmark_report(total_time, data_path)
            if os.path.exists(output_path):
                shutil.rmtree(output_path, ignore_errors=True)

        except Exception as e:
            error_info = {
                "error_message": str(e),
                "stack_trace": traceback.format_exc(),
            }

            # Save benchmark report with error information
            self._try_save_benchmark_report(-1, data_path, error_info)

            logger.error(
                f"{BENCH_RUN_ERROR_MSG_FOLLOWED_BY_ERROR_TYPE} {type(e).__name__}"
            )
            raise
        finally:
            self.spark.stop()


def build_parser() -> argparse.ArgumentParser:
    """Build the argument parser for bench_runner.py."""
    # Common arguments for all modes
    common = argparse.ArgumentParser(add_help=False)
    common.add_argument(
        "-d",
        "--data-path",
        type=str,
        required=True,
        help="Path to benchmark data parquet file",
    )
    common.add_argument(
        "-r",
        "--result-path",
        type=str,
        default=None,
        help="Path to write result report",
    )
    common.add_argument(
        "-g",
        "--gen-udf-data-module-path",
        type=str,
        required=True,
        help="Path to gen_udf_data module",
    )
    common.add_argument(
        "-t",
        "--test-module-path",
        type=str,
        required=True,
        help="Path to unit test module containing register_udf function",
    )
    common.add_argument(
        "-u",
        "--udf-name",
        type=str,
        required=True,
        help="Name of the UDF to benchmark",
    )
    common.add_argument(
        "-sc",
        "--spark-conf",
        action="append",
        default=[],
        help="Spark configuration in key=value format (e.g., --spark-conf spark.driver.memory=8g)",
    )

    parser = argparse.ArgumentParser(description="Benchmark UDF performance")
    subparsers = parser.add_subparsers(dest="mode", required=True)

    # Subparsers
    jvm_parser = subparsers.add_parser(
        "jvm-udf", parents=[common], help="Benchmark a Java/Scala UDF"
    )
    jvm_parser.add_argument(
        "-c",
        "--udf-class-path",
        type=str,
        required=True,
        help="Java fully qualified class name for the UDF",
    )

    sql_parser = subparsers.add_parser(
        "sql", parents=[common], help="Benchmark a SQL expression"
    )
    sql_parser.add_argument(
        "-s",
        "--sql-file-path",
        type=str,
        required=True,
        help="Path to SQL file",
    )

    pycudf_parser = subparsers.add_parser(
        "pycudf", parents=[common], help="Benchmark a PyCUDF (PySpark GPU) UDF"
    )
    pycudf_parser.add_argument(
        "-p",
        "--pycudf-file-path",
        type=str,
        required=True,
        help="Path to pyCUDF file",
    )

    subparsers.add_parser(
        "pyudf", parents=[common], help="Benchmark a PyUDF (PySpark CPU) UDF"
    )

    return parser


def main() -> None:
    parser = build_parser()
    args = parser.parse_args()

    # Validate input files
    if not os.path.exists(args.data_path):
        raise FileNotFoundError(f"Data file not found: {args.data_path}")

    if not os.path.exists(args.test_module_path):
        raise FileNotFoundError(f"Test module not found: {args.test_module_path}")

    if not os.path.exists(args.gen_udf_data_module_path):
        raise FileNotFoundError(
            f"Gen UDF data module not found: {args.gen_udf_data_module_path}"
        )

    # Get path to implementation, dependent on mode
    udf_class_path = getattr(args, "udf_class_path", None)
    sql_file_path = getattr(args, "sql_file_path", None)
    pycudf_file_path = getattr(args, "pycudf_file_path", None)

    if sql_file_path is not None and not os.path.exists(sql_file_path):
        raise FileNotFoundError(f"SQL file not found: {sql_file_path}")

    if pycudf_file_path is not None and not os.path.exists(pycudf_file_path):
        raise FileNotFoundError(f"pyCUDF file not found: {pycudf_file_path}")

    # Compute default result report path
    result_path = args.result_path
    if result_path is None:
        data_filename = compute_file_name_without_extension(args.data_path)
        os.makedirs("results", exist_ok=True)
        result_path = os.path.join(
            "results",
            f"{args.udf_name}_{data_filename}_{time.strftime('%Y%m%d_%H%M%S')}_result.json",
        )

    # Capture the full command line for reproducibility
    cli_args = sys.argv.copy()

    benchmark = UDFBenchmark(
        mode=args.mode,
        udf_name=args.udf_name,
        udf_class_path=udf_class_path,
        result_path=result_path,
        spark_configs=args.spark_conf,
        cli_args=cli_args,
        test_module_path=args.test_module_path,
        gen_udf_data_module_path=args.gen_udf_data_module_path,
        sql_file_path=sql_file_path,
        pycudf_file_path=pycudf_file_path,
    )
    benchmark.run(args.data_path)


if __name__ == "__main__":
    main()
