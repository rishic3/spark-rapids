#!/usr/bin/env bash
# Start a Spark standalone worker under nsys profile.
# Usage: ./start-nsys-worker.sh [master_url] [output_name]
#
# Defaults: master_url = spark://<hostname>:7077
#
# SPARK_NO_DAEMONIZE keeps the worker JVM in the foreground so nsys can
# trace it (and its forked executor) as direct children. Without this,
# the JVM re-parents and nsys loses control of it.
#
# Press Ctrl+C to stop profiling and shut down the worker.
#
# Based on docs/dev/nvtx_profiling.md.

set -euo pipefail

MASTER_URL="${1:-spark://$(hostname):7077}"
OUTPUT_NAME="${2:-nsys/worker_$(date +%Y%m%d-%H%M%S)}"

: "${SPARK_HOME:=/opt/spark-3.5.5}"
: "${WORKER_CORES:=16}"
: "${WORKER_MEMORY:=16G}"

export SPARK_WORKER_OPTS="-Dspark.worker.resource.gpu.amount=1 \
-Dspark.worker.resource.gpu.discoveryScript=${SPARK_HOME}/examples/src/main/scripts/getGpusResources.sh"

# Keep the worker in the foreground so nsys traces the full process tree.
export SPARK_NO_DAEMONIZE=1

echo "[nsys] Output: ${OUTPUT_NAME}.nsys-rep"
echo "[nsys] Press Ctrl+C to stop profiling and shut down the worker."

nsys profile \
    --trace=nvtx,cuda \
    --sample=none \
    --backtrace=none \
    --output="${OUTPUT_NAME}" \
    --force-overwrite=true \
    ${SPARK_HOME}/sbin/start-worker.sh \
        -c ${WORKER_CORES} -m ${WORKER_MEMORY} ${MASTER_URL}
