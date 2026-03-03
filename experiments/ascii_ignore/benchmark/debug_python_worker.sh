#!/bin/bash
# Debug launcher for Spark Python workers.
# Use as spark.pyspark.python / spark.executorEnv.PYSPARK_PYTHON to capture
# native crash evidence when workers die under profiler injection.
#
# IMPORTANT: Do not redirect worker stdout. Spark/PySpark uses worker stdout as
# a binary protocol channel between JVM and Python.

set -euo pipefail

PYTHON_BIN="${PYSPARK_UNPROFILED_PYTHON_BIN:-${PYSPARK_DRIVER_PYTHON:-/home/rishic/anaconda3/envs/spark-rapids/bin/python}}"
LOG_DIR="${DEBUG_PY_WORKER_LOG_DIR:-/tmp/spark-pyworker-debug}"
RUN_ID="${DEBUG_PY_WORKER_RUN_ID:-$(date +%Y%m%d-%H%M%S)}"
WORKER_LOG="${LOG_DIR}/worker-${RUN_ID}-${PPID}-$$.log"

mkdir -p "${LOG_DIR}"
touch "${WORKER_LOG}"

# Keep logs per worker process for later correlation with Spark task failures.
{
  echo "==== worker launch $(date -Is) ===="
  echo "pid=$$ ppid=$PPID"
  echo "argv=$*"
  echo "python_bin=${PYTHON_BIN}"
  echo "LD_PRELOAD=${LD_PRELOAD:-}"
  echo "QUADD_ENABLE_PROFILER=${QUADD_ENABLE_PROFILER:-}"
  echo "CUDA_INJECTION64_PATH=${CUDA_INJECTION64_PATH:-}"
} >> "${WORKER_LOG}"

# Native failures often skip Python traceback. These improve crash visibility.
ulimit -c unlimited || true
export PYTHONFAULTHANDLER=1

if [[ "${DEBUG_PY_WORKER_UNSET_INJECTION:-0}" == "1" ]]; then
  unset LD_PRELOAD
  unset QUADD_ENABLE_PROFILER
  unset CUDA_INJECTION64_PATH
fi

exec "${PYTHON_BIN}" "$@" 2>> "${WORKER_LOG}"
