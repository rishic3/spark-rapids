"""Synthetic data generation for the ``metric_status`` benchmark.

Generates 20 float64 telemetry columns with realistic distributions
(Gaussian, clipped to physical bounds).  The resulting dataset is
intentionally wide so that Arrow serialization of many numeric columns
dominates over the trivial threshold-comparison compute.
"""

import random
from typing import Callable, Dict

from pyspark.sql import DataFrame, SparkSession
from pyspark.sql.functions import col, udf
from pyspark.sql.types import DoubleType, IntegerType

METRIC_COLUMNS = [
    "cpu_pct", "gpu_pct", "mem_pct", "disk_io_pct", "net_bw_pct",
    "frame_time_ms", "encode_latency_ms", "decode_latency_ms", "rtt_ms",
    "jitter_ms", "packet_loss_pct", "input_latency_ms",
    "gpu_temp_c", "gpu_mem_pct", "cpu_temp_c", "swap_pct",
    "queue_depth", "error_rate_pct", "retry_pct", "thread_count",
]

# (mean, stddev, min, max) — tuned so ~10 % of rows trip warning and ~3 %
# trip critical, giving a realistic distribution of statuses.
_DISTRIBUTIONS = {
    "cpu_pct":            (55.0, 25.0, 0.0, 100.0),
    "gpu_pct":            (60.0, 25.0, 0.0, 100.0),
    "mem_pct":            (50.0, 20.0, 0.0, 100.0),
    "disk_io_pct":        (30.0, 25.0, 0.0, 100.0),
    "net_bw_pct":         (40.0, 25.0, 0.0, 100.0),
    "frame_time_ms":      (16.0, 15.0, 1.0, 200.0),
    "encode_latency_ms":  (8.0, 12.0, 0.5, 150.0),
    "decode_latency_ms":  (5.0, 10.0, 0.5, 150.0),
    "rtt_ms":             (50.0, 60.0, 1.0, 500.0),
    "jitter_ms":          (5.0, 12.0, 0.0, 200.0),
    "packet_loss_pct":    (0.2, 1.5, 0.0, 30.0),
    "input_latency_ms":   (20.0, 25.0, 1.0, 300.0),
    "gpu_temp_c":         (65.0, 15.0, 30.0, 105.0),
    "gpu_mem_pct":        (50.0, 25.0, 0.0, 100.0),
    "cpu_temp_c":         (55.0, 15.0, 25.0, 105.0),
    "swap_pct":           (10.0, 20.0, 0.0, 100.0),
    "queue_depth":        (20.0, 80.0, 0.0, 1000.0),
    "error_rate_pct":     (0.1, 1.0, 0.0, 20.0),
    "retry_pct":          (1.0, 4.0, 0.0, 50.0),
    "thread_count":       (80.0, 60.0, 1.0, 1000.0),
}


def create_gen_data_udfs() -> Dict[str, Callable]:
    """Create helper UDFs to generate synthetic data for each column."""

    def generate_id(seed: int) -> int:
        return seed

    def _make_metric_gen(col_name: str):
        mean, std, lo, hi = _DISTRIBUTIONS[col_name]

        def _generate(seed: int) -> float:
            if seed is None:
                return None
            rng = random.Random(seed + hash(col_name))
            return float(max(lo, min(hi, rng.gauss(mean, std))))

        return _generate

    udfs: Dict[str, Callable] = {
        "id": udf(generate_id, returnType=IntegerType()),
    }
    for col_name in METRIC_COLUMNS:
        udfs[col_name] = udf(_make_metric_gen(col_name), returnType=DoubleType())
    return udfs


def generate_synthetic_data(
    spark: SparkSession, num_rows: int, num_partitions: int
) -> DataFrame:
    """Generate synthetic data.  Schema: id (int), <20 float64 metric columns>."""
    gen_udfs = create_gen_data_udfs()
    base_df = spark.range(0, num_rows, numPartitions=num_partitions)

    select_exprs = [gen_udfs["id"](col("id")).alias("id")]
    for col_name in METRIC_COLUMNS:
        select_exprs.append(gen_udfs[col_name](col("id")).alias(col_name))

    return base_df.select(*select_exprs)


def execute_udf(spark: SparkSession, udf_name: str, df: DataFrame) -> DataFrame:
    """Apply the named UDF to the benchmark DataFrame."""
    df.createOrReplaceTempView("bench_table")
    struct_cols = ", ".join(METRIC_COLUMNS)
    return spark.sql(
        f"SELECT *, {udf_name}(struct({struct_cols})) as result FROM bench_table"
    )
