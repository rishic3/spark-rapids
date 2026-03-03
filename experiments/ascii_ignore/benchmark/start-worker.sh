#!/usr/bin/env bash
# Start a Spark standalone worker with GPU resources.
# Usage: ./start-worker.sh [master_url]
#
# Defaults: master_url = spark://<hostname>:7077

set -euo pipefail

MASTER_URL="${1:-spark://$(hostname):7077}"

: "${SPARK_HOME:=/opt/spark-3.5.5}"
: "${WORKER_CORES:=16}"
: "${WORKER_MEMORY:=16G}"

export SPARK_WORKER_OPTS="-Dspark.worker.resource.gpu.amount=1 \
-Dspark.worker.resource.gpu.discoveryScript=${SPARK_HOME}/examples/src/main/scripts/getGpusResources.sh"

SPARK_NO_DAEMONIZE=1 ${SPARK_HOME}/sbin/start-worker.sh \
    -c ${WORKER_CORES} -m ${WORKER_MEMORY} ${MASTER_URL}
