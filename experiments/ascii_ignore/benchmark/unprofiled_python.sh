#!/bin/bash

unset LD_PRELOAD
unset QUADD_ENABLE_PROFILER
unset CUDA_INJECTION64_PATH

# Optional: disable Python NVTX emission from the worker process.
# Set PYWORKER_DISABLE_NVTX=1 (or true/yes) to enable this.
case "${PYWORKER_DISABLE_NVTX:-0}" in
  1|true|TRUE|True|yes|YES|Yes)
    export NVTX_DISABLE=1
    ;;
esac

PYTHON_BIN="${PYSPARK_UNPROFILED_PYTHON_BIN:-${PYSPARK_DRIVER_PYTHON:-/home/rishic/anaconda3/envs/spark-rapids/bin/python}}"
exec "${PYTHON_BIN}" "$@"
