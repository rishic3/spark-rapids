#!/bin/bash

unset LD_PRELOAD
unset QUADD_ENABLE_PROFILER
unset CUDA_INJECTION64_PATH

PYTHON_BIN="${PYSPARK_UNPROFILED_PYTHON_BIN:-${PYSPARK_DRIVER_PYTHON:-/home/rishic/anaconda3/envs/spark-rapids/bin/python}}"
exec "${PYTHON_BIN}" "$@"
