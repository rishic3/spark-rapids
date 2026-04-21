# SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION.
# SPDX-License-Identifier: Apache-2.0

import argparse
import importlib
import os
import shutil
import socket
import sys
import time
from pathlib import Path
from typing import Any, Callable, Dict, List, Optional

from pyspark import AccumulatorParam
from pyspark.sql import SparkSession
from pyspark.sql.types import StringType


class _DictAccumulatorParam(AccumulatorParam):
    """Accumulator that sums numeric dict values across tasks."""
    def zero(self, value):
        return {k: type(v)() for k, v in value.items()}

    def addInPlace(self, value1, value2):
        for k in value2:
            value1[k] = value1.get(k, 0) + value2[k]
        return value1


SCRIPT_DIR = Path(__file__).resolve().parent
PROJECT_DIR = SCRIPT_DIR.parent
SRC_DIR = PROJECT_DIR / "src"
HOSTNAME = socket.gethostname()
DEFAULT_THREADS = 16

sys.path.insert(0, str(SRC_DIR))

from udf_registry import available_names, get_config  # type: ignore[import-not-found]
from history_server_client import fetch_op_time  # type: ignore[import-not-found]


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
    "spark.python.worker.reuse": "true",
    "spark.sql.execution.arrow.maxRecordsPerBatch": "10000",
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
    "spark.master": f"local[{DEFAULT_THREADS}]",
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


_GPU_TIMING_ZERO = {
    "cold_init_plus_first_h2d_s": 0.0,
    "warm_h2d_s": 0.0,
    "warm_d2h_s": 0.0,
    "total_d2h_s": 0.0,
    "total_compute_s": 0.0,
    "batches": 0,
    "cold_batches": 0,
    "rows": 0,
}

_CPU_TIMING_ZERO = {
    "total_compute_s": 0.0,
    "batches": 0,
    "rows": 0,
}


# (mode) -> (module_suffix, udf_suffix, is_gpu)
# - mode        : user-facing --mode value
# - module_suffix: appended to {cfg.name} to build the python module name
# - udf_suffix  : appended to {cfg.name} to build the UDF + make_timed_ identifier
# - is_gpu      : whether to use GPU timing accumulator schema
MODE_SPEC: Dict[str, tuple] = {
    "cpu":       ("",           "pandas",    False),
    "gpu":       ("_gpu",       "gpu",       True),
    "cpu-arrow": ("_arrow",     "arrow",     False),
    "gpu-arrow": ("_gpu_arrow", "gpu_arrow", True),
}


def run_single_benchmark(
    mode: str,
    cluster: str,
    data_path: str,
    rapids_jar_path: str,
    extra_spark_configs: Dict[str, str],
    coalesce: Optional[int] = None,
    udf_key: str = "ascii_ignore",
    history_server_port: Optional[int] = 18080,
) -> float:
    cfg = get_config(udf_key)

    if mode not in MODE_SPEC:
        raise ValueError(f"Unknown mode '{mode}'. Expected one of: {list(MODE_SPEC)}")
    module_suffix, udf_suffix, is_gpu = MODE_SPEC[mode]

    module_name = f"{cfg.name}{module_suffix}"
    udf_name = f"{cfg.name}_{udf_suffix}"
    udf_mod = importlib.import_module(module_name)
    udf_func = getattr(udf_mod, udf_name)

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
    print(f"Spark master: {spark.conf.get('spark.master')}")
    print(f"Spark Arrow maxRecordsPerBatch: {spark.conf.get('spark.sql.execution.arrow.maxRecordsPerBatch')}")
    output_path = str(SCRIPT_DIR / f"_tmp_{udf_name}_output")

    timing_zero = _GPU_TIMING_ZERO if is_gpu else _CPU_TIMING_ZERO
    timing_acc = spark.sparkContext.accumulator(
        dict(timing_zero), _DictAccumulatorParam()
    )
    make_timed = getattr(udf_mod, f"make_timed_{cfg.name}_{udf_suffix}")
    udf_func = make_timed(timing_acc)

    try:
        register_udf(spark, udf_name, udf_func)
        start = time.time()
        df = spark.read.parquet(data_path)
        if coalesce is not None:
            df = df.coalesce(coalesce)
        df.createOrReplaceTempView("bench_table")

        sql = cfg.sql_template.format(udf_name=udf_name)
        result_df = spark.sql(sql)
        result_df.write.mode("overwrite").parquet(output_path)
        elapsed = time.time() - start
    finally:
        shutil.rmtree(output_path, ignore_errors=True)
        if timing_acc is not None and not is_gpu:
            t = timing_acc.value
            print("[UDF_TIMING]")
            print(f"  {'batches:':38s} {t['batches']:10d}")
            print(f"  {'rows:':38s} {t['rows']:10d}")
            print(f"  {'total compute:':38s} {t['total_compute_s']:10.4f}s")
            per_batch_s = t["total_compute_s"] / t["batches"] if t["batches"] > 0 else 0.0
            per_row_us = (t["total_compute_s"] * 1e6 / t["rows"]) if t["rows"] > 0 else 0.0
            print(f"  {'compute per batch:':38s} {per_batch_s:10.6f}s")
            print(f"  {'compute per row:':38s} {per_row_us:10.3f}us")
        elif timing_acc is not None and is_gpu:
            t = timing_acc.value
            warm_batches = t["batches"] - t["cold_batches"]
            warm_h2d_per_batch = (
                t["warm_h2d_s"] / warm_batches if warm_batches > 0 else 0.0
            )
            est_cold_h2d = warm_h2d_per_batch * t["cold_batches"]
            est_pure_cold_init = max(0.0, t["cold_init_plus_first_h2d_s"] - est_cold_h2d)
            est_total_h2d = t["warm_h2d_s"] + est_cold_h2d
            print(f"[UDF_TIMING]")
            print(f"  {'batches:':38s} {t['batches']:10d}")
            print(f"  {'cold_batches:':38s} {t['cold_batches']:10d}")
            print(f"  {'warm_batches:':38s} {warm_batches:10d}")
            print(f"  {'rows:':38s} {t['rows']:10d}")
            print(f"  {'--------------------------------':38s}")
            print(f"  {'total compute (cold+warm):':38s} {t['total_compute_s']:10.4f}s")
            print(f"  {'total D2H (cold+warm):':38s} {t['total_d2h_s']:10.4f}s")
            print(f"  {'estimated total H2D:':38s} {est_total_h2d:10.4f}s")
            print(f"  {'--------------------------------':38s}")
            print(f"  {'warm total H2D:':38s} {t['warm_h2d_s']:10.4f}s ({warm_batches} batches)")
            print(f"  {'warm H2D per batch:':38s} {warm_h2d_per_batch:10.6f}s")
            print(
                f"  {'cold init + first H2D:':38s} "
                f"{t['cold_init_plus_first_h2d_s']:10.4f}s "
                f"(across {t['cold_batches']} workers)"
            )
            print(f"  {'estimated cold H2D:':38s} {est_cold_h2d:10.4f}s")
            print(f"  {'estimated pure cold init:':38s} {est_pure_cold_init:10.4f}s")
        spark.stop()

    if history_server_port is not None:
        node_name = "GpuArrowEvalPython"
        op = fetch_op_time(app_name=app_name, node_name=node_name, port=history_server_port)
        print(f"[HISTORY_SERVER] app='{app_name}' node='{node_name}'")
        if op is None:
            print(
                f"  (no op time found on port {history_server_port}; is the history "
                f"server running and reading {spark_configs.get('spark.eventLog.dir')}?)"
            )
        else:
            print(
                f"  op time: {op['total']:.4f}s "
                f"({op['min']:.4f}, {op['med']:.4f}, {op['max']:.4f})"
            )

    return elapsed


def main() -> None:
    parser = argparse.ArgumentParser(description="Benchmark UDF transfer overhead in Spark-RAPIDS")
    parser.add_argument(
        "--mode",
        choices=list(MODE_SPEC.keys()),
        help="Mode to run: cpu | gpu | cpu-arrow | gpu-arrow",
    )
    parser.add_argument(
        "--udf",
        default="ascii_ignore",
        choices=available_names(),
        help="UDF to benchmark (default: ascii_ignore)",
    )
    parser.add_argument("--cluster", choices=["standalone", "local"], default="local",
                        help="Cluster mode: 'standalone' or 'local' (default)")
    parser.add_argument("--data-path", required=True, help="Input parquet data path")
    parser.add_argument("--rapids-jar-path", required=True, help="Path to RAPIDS plugin jar")
    parser.add_argument("--spark-conf", action="append", default=[], help="Spark configs in key=value format")
    parser.add_argument("--coalesce", type=int, default=None, help="Number of partitions to coalesce to")
    parser.add_argument(
        "--history-server-port",
        type=int,
        default=18080,
        help="Spark history server port for op-time lookup (default: 18080). Set to 0 to disable.",
    )
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
        coalesce=args.coalesce,
        udf_key=args.udf,
        history_server_port=args.history_server_port or None,
    )
    _, udf_suffix, _ = MODE_SPEC[args.mode]
    udf_label = f"{args.udf}_{udf_suffix}"
    print(f"E2E runtime (s) ({args.mode}/{udf_label}): {runtime:.2f}")


if __name__ == "__main__":
    main()
