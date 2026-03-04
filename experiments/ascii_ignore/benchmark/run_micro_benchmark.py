# SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION.
# SPDX-License-Identifier: Apache-2.0

import argparse
import gc
import sys
import time
from pathlib import Path
from typing import Callable, Iterator, List, Sequence


DEFAULT_WARMUP = 2
DEFAULT_MEASURED = 4
DEFAULT_COLUMN = "input_str"

SCRIPT_DIR = Path(__file__).resolve().parent
PROJECT_DIR = SCRIPT_DIR.parent
SRC_DIR = PROJECT_DIR / "src"
sys.path.insert(0, str(SRC_DIR))


def _run_benchmark(warmup: int, measured: int, block: Callable[[], object]) -> List[float]:
    for _ in range(warmup):
        result = block()
        del result

    times: List[float] = []
    for _ in range(measured):
        start = time.perf_counter()
        result = block()
        elapsed = time.perf_counter() - start
        del result
        times.append(elapsed)
    times.sort()
    return times


def _build_batches(series, batch_size: int) -> Sequence:
    if batch_size <= 0:
        return (series,)

    return tuple(
        series.iloc[start : start + batch_size] for start in range(0, len(series), batch_size)
    )


def _iter_batches(batches: Sequence) -> Iterator:
    for batch in batches:
        yield batch


def _make_batched_runner(batches: Sequence, udf_func: Callable) -> Callable[[], object]:
    def _run_in_batches():
        for batch in _iter_batches(batches):
            result = udf_func(batch)
            del result
        return None

    return _run_in_batches


def _synchronize_gpu() -> None:
    try:
        from numba import cuda  # type: ignore

        cuda.current_context().synchronize()
        return
    except Exception:
        pass

    try:
        import cupy  # type: ignore

        cupy.cuda.runtime.deviceSynchronize()
    except Exception:
        pass


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="In-memory benchmark for pure UDF execution time.")
    parser.add_argument("mode", choices=["cpu", "gpu", "all"], help="Execution mode")
    parser.add_argument("-d", "--data-path", required=True, help="Parquet data file or directory")
    parser.add_argument("-c", "--column", default=DEFAULT_COLUMN, help="Input string column name")
    parser.add_argument(
        "-r",
        "--rows",
        type=int,
        default=-1,
        help="Max rows to benchmark (-1 means all rows)",
    )
    parser.add_argument("-w", "--warmup", type=int, default=DEFAULT_WARMUP)
    parser.add_argument("-m", "--measured", type=int, default=DEFAULT_MEASURED)
    parser.add_argument(
        "-b",
        "--batch-size",
        type=int,
        default=-1,
        help="Rows per UDF batch (-1 means full dataset in one batch)",
    )
    return parser.parse_args()


def _validate_args(args: argparse.Namespace) -> Path:
    data_path = Path(args.data_path).resolve()
    if not data_path.exists():
        raise FileNotFoundError(f"Data path not found: {data_path}")
    if args.rows == 0 or args.rows < -1:
        raise ValueError("--rows must be -1 or a positive integer")
    if args.warmup < 0:
        raise ValueError("--warmup must be >= 0")
    if args.measured <= 0:
        raise ValueError("--measured must be > 0")
    if args.batch_size == 0 or args.batch_size < -1:
        raise ValueError("--batch-size must be -1 or a positive integer")
    return data_path


def _format_runtime_line(mode: str, rows: int, times: List[float]) -> str:
    median_s = times[len(times) // 2]
    min_s = times[0]
    return (
        f"   {mode.upper():<4} | {rows:>14,d} rows | "
        f"median {median_s:>10.4f} s | min {min_s:>10.4f} s"
    )


def _run_single_mode(mode: str, args: argparse.Namespace, data_path: Path) -> List[float]:
    if mode == "cpu":
        import pandas as pd
        from ascii_ignore_impl import ascii_ignore_pandas_impl  # type: ignore[import-not-found]

        df = pd.read_parquet(data_path)
        if args.column not in df.columns:
            raise KeyError(f"Column '{args.column}' not found in dataset")
        if args.rows > 0:
            df = df.head(args.rows)
        input_series = df[args.column]
        batches = _build_batches(input_series, args.batch_size)
        rows = len(input_series)
        mem_mb = float(df.memory_usage(index=True, deep=True).sum()) / (1024.0 * 1024.0)
        run_once = _make_batched_runner(batches, ascii_ignore_pandas_impl)
    else:
        import cudf
        from ascii_ignore_gpu_impl import ascii_ignore_cudf_impl  # type: ignore[import-not-found]

        df = cudf.read_parquet(data_path)
        if args.column not in df.columns:
            raise KeyError(f"Column '{args.column}' not found in dataset")
        if args.rows > 0:
            df = df.head(args.rows)
        input_series = df[args.column]
        batches = _build_batches(input_series, args.batch_size)
        rows = len(input_series)
        mem_mb = float(df.memory_usage(deep=True).sum()) / (1024.0 * 1024.0)
        run_once = _make_batched_runner(batches, ascii_ignore_cudf_impl)

    effective_batch_size = rows if args.batch_size <= 0 else args.batch_size
    num_batches = len(batches)

    print(
        f"Loaded {rows:,d} rows x {len(df.columns)} columns "
        f"({mem_mb:.1f} MB) from: {data_path}"
    )
    print(
        f"Microbenchmark (in-memory): mode={mode}, "
        f"warmup={args.warmup}, measured={args.measured}, column={args.column}, "
        f"batch_size={effective_batch_size}, batches={num_batches}"
    )

    times = _run_benchmark(warmup=args.warmup, measured=args.measured, block=run_once)
    if mode == "gpu":
        _synchronize_gpu()
        del run_once, batches, input_series, df
        gc.collect()
        _synchronize_gpu()
    print(_format_runtime_line(mode, rows, times))
    return times


def main() -> None:
    args = _parse_args()
    data_path = _validate_args(args)
    modes = ["cpu", "gpu"] if args.mode == "all" else [args.mode]
    results = {}

    for mode in modes:
        results[mode] = _run_single_mode(mode, args, data_path)

    if args.mode == "all":
        cpu_min_ms = results["cpu"][0] * 1000.0
        gpu_min_ms = results["gpu"][0] * 1000.0
        if gpu_min_ms > 0:
            print(f">> Speedup: {cpu_min_ms / gpu_min_ms:.2f}x (CPU/GPU best)")


if __name__ == "__main__":
    main()
