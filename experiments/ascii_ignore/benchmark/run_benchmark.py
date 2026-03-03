# SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION.
# SPDX-License-Identifier: Apache-2.0

import argparse
import os
import shutil
import socket
import sys
import time
from pathlib import Path
from typing import Any, Callable, Dict, List, Optional

from pyspark.sql import SparkSession
from pyspark.sql.types import StringType


SCRIPT_DIR = Path(__file__).resolve().parent
PROJECT_DIR = SCRIPT_DIR.parent
SRC_DIR = PROJECT_DIR / "src"
HOSTNAME = socket.gethostname()

sys.path.insert(0, str(SRC_DIR))

from ascii_ignore import ascii_ignore  # type: ignore[import-not-found]
from ascii_ignore_gpu import ascii_ignore_gpu  # type: ignore[import-not-found]


BASE_SPARK_CONFIGS: Dict[str, str] = {
    "spark.rapids.sql.enabled": "true",
    "spark.plugins": "com.nvidia.spark.SQLPlugin",
    "spark.locality.wait": "0s",
    "spark.sql.cache.serializer": "com.nvidia.spark.ParquetCachedBatchSerializer",
    "spark.rapids.sql.format.parquet.reader.type": "MULTITHREADED",
    "spark.rapids.sql.reader.batchSizeBytes": "1000MB",
    "spark.sql.files.maxPartitionBytes": "512MB",
    "spark.rapids.sql.metrics.level": "DEBUG",
    "spark.rapids.memory.gpu.allocFraction": "0.5",
    "spark.eventLog.dir": "event_logs",
    "spark.eventLog.enabled": "true",
    # "spark.rapids.sql.python.gpu.enabled": "true",
    # "spark.rapids.python.memory.gpu.pooling.enabled": "false",
}

STANDALONE_CONFIGS: Dict[str, str] = {
    "spark.master": f"spark://{HOSTNAME}:7077",
    "spark.driver.memory": "4g",
    "spark.executor.memory": "16g",
    "spark.executor.cores": "16",
    "spark.task.resource.gpu.amount": "0.0625",
    "spark.executor.resource.gpu.amount": "1",
    "spark.executor.extraJavaOptions": "-Dai.rapids.cudf.nvtx.enabled=true",
}

LOCAL_CONFIGS: Dict[str, str] = {
    "spark.master": "local[*]",
    "spark.driver.memory": "16g",
    "spark.driver.extraJavaOptions": "-Dai.rapids.cudf.nvtx.enabled=true",
}


def parse_spark_conf_entries(entries: List[str]) -> Dict[str, str]:
    parsed: Dict[str, str] = {}
    for entry in entries:
        if "=" not in entry:
            raise ValueError(
                f"Invalid --spark-conf '{entry}'. Expected key=value format."
            )
        key, value = entry.split("=", 1)
        key = key.strip()
        if not key:
            raise ValueError(f"Invalid --spark-conf '{entry}'. Key cannot be empty.")
        parsed[key] = value
    return parsed


def create_spark(
    app_name: str,
    spark_configs: Dict[str, str],
    extra_py_files: Optional[List[Path]] = None,
) -> SparkSession:
    worker_python = spark_configs.get("spark.pyspark.python")
    driver_python = spark_configs.get("spark.pyspark.driver.python")
    if worker_python:
        os.environ["PYSPARK_PYTHON"] = worker_python
    else:
        os.environ.setdefault("PYSPARK_PYTHON", sys.executable)
    if driver_python:
        os.environ["PYSPARK_DRIVER_PYTHON"] = driver_python
    else:
        os.environ.setdefault("PYSPARK_DRIVER_PYTHON", sys.executable)

    spark_builder: Any = SparkSession.builder
    builder = spark_builder.appName(app_name)
    for key, value in spark_configs.items():
        builder = builder.config(key, value)

    spark = builder.getOrCreate()
    for py_file in extra_py_files or []:
        spark.sparkContext.addPyFile(str(py_file))
    return spark


def register_udf(spark: SparkSession, udf_name: str, udf_func: Callable) -> None:
    if hasattr(udf_func, "returnType"):
        spark.udf.register(udf_name, udf_func)
    else:
        spark.udf.register(udf_name, udf_func, StringType())


def run_single_benchmark(
    mode: str,
    cluster: str,
    data_path: str,
    rapids_jar_path: str,
    extra_spark_configs: Dict[str, str],
) -> float:
    if mode == "cpu":
        udf_name = "ascii_ignore"
        udf_func: Callable = ascii_ignore
    else:
        udf_name = "ascii_ignore_gpu"
        udf_func = ascii_ignore_gpu

    app_name = f"{udf_name}_{Path(data_path).stem}_{time.strftime('%Y%m%d_%H%M%S')}"
    spark_configs = dict(BASE_SPARK_CONFIGS)
    spark_configs.update(STANDALONE_CONFIGS if cluster == "standalone" else LOCAL_CONFIGS)
    spark_configs["spark.jars"] = rapids_jar_path
    spark_configs.update(extra_spark_configs)
    spark = create_spark(
        app_name=app_name,
        spark_configs=spark_configs,
        extra_py_files=list(SRC_DIR.glob("*.py")),
    )
    output_path = str(SCRIPT_DIR / f"_tmp_{udf_name}_output")

    try:
        register_udf(spark, udf_name, udf_func)
        start = time.time()
        df = spark.read.parquet(data_path)
        df.createOrReplaceTempView("bench_table")
        result_df = spark.sql(
            f"SELECT *, {udf_name}(input_str) as result FROM bench_table"
        )
        result_df.write.mode("overwrite").parquet(output_path)
        return time.time() - start
    finally:
        shutil.rmtree(output_path, ignore_errors=True)
        spark.stop()


def main() -> None:
    parser = argparse.ArgumentParser(description="Benchmark ascii_ignore")
    parser.add_argument("--mode", choices=["cpu", "gpu"], required=True)
    parser.add_argument("--cluster", choices=["standalone", "local"], default="standalone",
                        help="Cluster mode: 'standalone' (default) or 'local'")
    parser.add_argument("--data-path", required=True, help="Input parquet data path")
    parser.add_argument("--rapids-jar-path", required=True, help="Path to RAPIDS plugin jar")
    parser.add_argument("--spark-conf", action="append", default=[], help="Spark configs in key=value format")
    args = parser.parse_args()

    data_file = Path(args.data_path).resolve()
    if not data_file.exists():
        raise FileNotFoundError(f"Data file not found: {data_file}")
    jar_file = Path(args.rapids_jar_path).resolve()
    if not jar_file.exists():
        raise FileNotFoundError(f"RAPIDS jar not found: {jar_file}")

    extra_spark_configs = parse_spark_conf_entries(args.spark_conf)

    runtime = run_single_benchmark(
        mode=args.mode,
        cluster=args.cluster,
        data_path=str(data_file),
        rapids_jar_path=str(jar_file),
        extra_spark_configs=extra_spark_configs,
    )
    udf_name = "ascii_ignore" if args.mode == "cpu" else "ascii_ignore_gpu"
    print(f"E2E runtime ({args.mode}/{udf_name}): {runtime:.2f}s")


if __name__ == "__main__":
    main()
