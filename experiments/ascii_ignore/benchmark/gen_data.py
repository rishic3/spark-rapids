# SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION.
# SPDX-License-Identifier: Apache-2.0

import argparse
import logging
import os
from typing import List, Optional

from pyspark.sql import DataFrame

from cuaether_assistant.constants import (
    GEN_DATA_ERROR_MSG_FOLLOWED_BY_ERROR_TYPE,
    GEN_DATA_SUCCESS_MSG_FOLLOWED_BY_PATH,
    ROW_COUNT_MISMATCH_ERROR_MSG,
)
from cuaether_assistant.templates.utils import (
    initialize_spark,
    load_module,
    spark_configs_for_pythonpath,
)
from cuaether_assistant.utils import format_number_human_readable

logging.basicConfig(format="%(message)s", level=logging.INFO)
logger = logging.getLogger(__name__)


class GenData:
    def __init__(
        self,
        rows: int,
        partitions: int,
        spark_configs: List[str],
        gen_udf_data_module_path: str,
        test_module_path: Optional[str] = None,
    ):
        self.rows = rows
        self.partitions = partitions

        spark_configs_to_use = list(spark_configs)
        if test_module_path is not None:
            spark_configs_to_use.extend(
                spark_configs_for_pythonpath(
                    [os.path.dirname(os.path.abspath(test_module_path))]
                )
            )
        self.spark_configs = spark_configs_to_use

        # Initialize Spark session
        self.spark = initialize_spark(self.spark_configs)

        # Load gen data helper functions
        self.gen_data_helpers = load_module(
            "gen_data_helpers", gen_udf_data_module_path
        )

    def _generate_dataset(self, num_rows: int) -> DataFrame:
        """Generate synthetic dataset"""
        try:
            return self.gen_data_helpers.generate_synthetic_data(
                spark=self.spark, num_rows=num_rows, num_partitions=self.partitions
            )
        except Exception as e:
            logger.exception(
                f"{GEN_DATA_ERROR_MSG_FOLLOWED_BY_ERROR_TYPE} {type(e).__name__}"
            )
            raise

    def validate_only(
        self,
        test_module_path: str,
        udf_name: str,
        udf_class_path: Optional[str],
        validate_mode: str = "udf",
        sql_file_path: Optional[str] = None,
    ) -> None:
        """
        Validate the generated dataset by running through the UDF or SQL query.

        Args:
            test_module_path: Path to test module containing register_udf function
            udf_name: Name of the UDF
            udf_class_path: Fully qualified class name for Java/Scala UDF (None for Python UDFs)
            validate_mode: Type of validator to use ("udf" or "sql")
            sql_file_path: Path to SQL file (required if validate_mode is "sql")
        """
        try:
            # For Python UDFs, test dir was already added to PYTHONPATH in __init__
            # when GenData was constructed with test_module_path.

            # Load test module and register UDF (required for both validate_mode "udf" and "sql")
            self.test_module = load_module(
                "test_module", test_module_path, ["register_udf"]
            )
            # Same registration entry point for both JVM and Python UDFs (consistent with bench_runner).
            # Only the argument differs: class_path for Java/Scala, udf_func for Python.
            if udf_class_path is not None:
                self.test_module.register_udf(
                    spark=self.spark, udf_name=udf_name, class_path=udf_class_path
                )
            else:
                if not hasattr(self.test_module, udf_name):
                    raise ValueError(
                        f"Python UDF '{udf_name}' not found in test module. "
                        f"Make sure the UDF is imported in the test file."
                    )
                udf_func = getattr(self.test_module, udf_name)
                self.test_module.register_udf(
                    spark=self.spark, udf_name=udf_name, udf_func=udf_func
                )

            # Generate dataset
            output_df = self._generate_dataset(self.rows)

            # Run appropriate validator
            if validate_mode == "udf":
                from cuaether_assistant.templates.validate import validate_udf

                validate_udf(
                    spark=self.spark,
                    df=output_df,
                    udf_name=udf_name,
                    gen_data_helpers=self.gen_data_helpers,
                )
            elif validate_mode == "sql":
                from cuaether_assistant.templates.validate import validate_sql

                assert sql_file_path is not None, "SQL file path is required"
                validate_sql(
                    spark=self.spark,
                    df=output_df,
                    udf_name=udf_name,
                    sql_file_path=sql_file_path,
                    gen_data_helpers=self.gen_data_helpers,
                )
            else:
                raise ValueError(f"Unknown validator mode: {validate_mode}")

            logger.info("Validation passed on sample dataset")

            print("Generated benchmark dataframe:")
            output_df.show(n=10, truncate=False)

            # Verify num rows match
            actual_rows = output_df.count()
            if actual_rows != self.rows:
                raise ValueError(
                    f"{ROW_COUNT_MISMATCH_ERROR_MSG} generated {actual_rows} rows, expected {self.rows} rows"
                )
        finally:
            self.spark.stop()

    def generate_dataset(self, output_path: str) -> None:
        """
        Generate the dataset and write to output path.

        Args:
            output_path: Path to write parquet file
        """
        try:
            # Create output directory if it doesn't exist
            os.makedirs(os.path.dirname(output_path), exist_ok=True)

            output_df = self._generate_dataset(self.rows)

            # Repartition to desired output partitions
            if output_df.rdd.getNumPartitions() != self.partitions:
                output_df = output_df.repartition(self.partitions)

            output_df.write.mode("overwrite").parquet(output_path)
            logger.info(f"{GEN_DATA_SUCCESS_MSG_FOLLOWED_BY_PATH} {output_path}")
        finally:
            self.spark.stop()


def build_parser() -> argparse.ArgumentParser:
    """Build the argument parser for gen_data.py"""
    parser = argparse.ArgumentParser(description="Generate benchmark data with Spark")
    parser.add_argument(
        "-o",
        "--output-path",
        type=str,
        default=None,
        help="Output path to write parquet file",
    )
    parser.add_argument(
        "-r",
        "--rows",
        type=int,
        default=1_000_000,
        help="Number of rows to generate (default: 1M)",
    )
    parser.add_argument(
        "--partitions",
        type=int,
        required=True,
        help="Number of output partitions for the generated dataset",
    )
    parser.add_argument(
        "-g",
        "--gen-udf-data-module-path",
        type=str,
        required=True,
        help="Path to gen_udf_data module",
    )
    parser.add_argument(
        "-t",
        "--test-module-path",
        type=str,
        help="Path to unit test module containing register_udf function (required if --validate-mode is used)",
    )
    parser.add_argument(
        "-u",
        "--udf-name",
        type=str,
        help="Name of the UDF to validate with (required if --validate-mode is used)",
    )
    parser.add_argument(
        "-c",
        "--udf-class-path",
        type=str,
        default=None,
        help="Java fully qualified class name for the UDF (required for --validate-mode with Java/Scala UDFs)",
    )
    parser.add_argument(
        "-s",
        "--sql-file-path",
        type=str,
        default=None,
        help="Path to SQL file (required if --validate-mode is 'sql')",
    )
    parser.add_argument(
        "-sc",
        "--spark-conf",
        action="append",
        default=[],
        help="Spark configuration in key=value format (e.g., --spark-conf spark.driver.memory=8g)",
    )
    parser.add_argument(
        "-v",
        "--validate-mode",
        type=str,
        default=None,
        choices=["udf", "sql"],
        help="Validate the generated dataset by running through the UDF or SQL query; \
              only performs validation, does not write to disk",
    )
    return parser


def main():
    parser = build_parser()
    args = parser.parse_args()

    if not os.path.exists(args.gen_udf_data_module_path):
        raise FileNotFoundError(
            f"Gen UDF data module not found: {args.gen_udf_data_module_path}"
        )

    if args.validate_mode is not None:
        if not args.test_module_path:
            raise ValueError(
                "--test-module-path is required when --validate-mode is used"
            )
        if not args.udf_name:
            raise ValueError("--udf-name is required when --validate-mode is used")
        if args.validate_mode == "sql" and not args.sql_file_path:
            raise ValueError(
                "--sql-file-path is required when --validate-mode is 'sql'"
            )
        if not os.path.exists(args.test_module_path):
            raise FileNotFoundError(f"Test module not found: {args.test_module_path}")

    gen_data = GenData(
        rows=args.rows,
        partitions=args.partitions,
        spark_configs=args.spark_conf,
        gen_udf_data_module_path=args.gen_udf_data_module_path,
        test_module_path=(
            args.test_module_path if args.validate_mode is not None else None
        ),
    )

    if args.validate_mode is not None:
        gen_data.validate_only(
            test_module_path=args.test_module_path,
            udf_name=args.udf_name,
            udf_class_path=args.udf_class_path,
            validate_mode=args.validate_mode,
            sql_file_path=args.sql_file_path,
        )
    else:
        # Compute default data output path
        output_path = args.output_path
        if output_path is None:
            output_path = f"data/bench_data_{format_number_human_readable(args.rows)}_rows.parquet"
            logger.info(f"Writing to default output path '{output_path}'")

        # Generate dataset
        gen_data.generate_dataset(output_path=output_path)


if __name__ == "__main__":
    main()
