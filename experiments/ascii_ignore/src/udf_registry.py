"""Central registry mapping UDF names to their benchmark configuration.

Each UDF follows a convention-based naming pattern:
  CPU impl:   {name}_impl.py   →  {name}_pandas_impl()
  GPU impl:   {name}_gpu_impl.py →  {name}_cudf_impl()
  CPU UDF:    {name}.py         →  {name}_pandas,  make_timed_{name}_pandas()
  GPU UDF:    {name}_gpu.py     →  {name}_gpu,     make_timed_{name}_gpu()
  Data gen:   gen_{name}_data.py →  generate_synthetic_data(), execute_udf()
"""

from dataclasses import dataclass, field
from typing import List


@dataclass
class UdfConfig:
    """Configuration for a single UDF in the benchmark suite."""

    name: str
    input_columns: List[str]
    sql_template: str
    input_mode: str = "series"  # "series" for single-column, "dataframe" for multi-column


REGISTRY = {
    "ascii_ignore": UdfConfig(
        name="ascii_ignore",
        input_columns=["input_str"],
        sql_template="SELECT *, {udf_name}(input_str) as result FROM bench_table",
        input_mode="series",
    ),
    "redact_pii": UdfConfig(
        name="redact_pii",
        input_columns=["input_text"],
        sql_template="SELECT *, {udf_name}(input_text) as result FROM bench_table",
        input_mode="series",
    ),
    "metric_status": UdfConfig(
        name="metric_status",
        input_columns=[
            "cpu_pct", "gpu_pct", "mem_pct", "disk_io_pct", "net_bw_pct",
            "frame_time_ms", "encode_latency_ms", "decode_latency_ms", "rtt_ms",
            "jitter_ms", "packet_loss_pct", "input_latency_ms",
            "gpu_temp_c", "gpu_mem_pct", "cpu_temp_c", "swap_pct",
            "queue_depth", "error_rate_pct", "retry_pct", "thread_count",
        ],
        sql_template=(
            "SELECT *, {udf_name}(struct("
            "cpu_pct, gpu_pct, mem_pct, disk_io_pct, net_bw_pct, "
            "frame_time_ms, encode_latency_ms, decode_latency_ms, rtt_ms, "
            "jitter_ms, packet_loss_pct, input_latency_ms, "
            "gpu_temp_c, gpu_mem_pct, cpu_temp_c, swap_pct, "
            "queue_depth, error_rate_pct, retry_pct, thread_count"
            ")) as result FROM bench_table"
        ),
        input_mode="dataframe",
    ),
}


def get_config(name: str) -> UdfConfig:
    if name not in REGISTRY:
        avail = ", ".join(REGISTRY.keys())
        raise ValueError(f"Unknown UDF '{name}'. Available: {avail}")
    return REGISTRY[name]


def available_names() -> List[str]:
    return list(REGISTRY.keys())
