#!/usr/bin/env bash

set -euo pipefail

: "${SPARK_HOME:=/opt/spark-3.5.5}"

${SPARK_HOME}/sbin/stop-worker.sh
