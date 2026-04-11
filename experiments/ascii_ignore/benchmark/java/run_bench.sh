#!/bin/bash
# Run JVM-side transfer overhead microbenchmark.
#
# Usage:
#   ./run_bench.sh --data-path /path/to/parquet [--rows N] [--batch-size N] \
#                  [--warmup 3] [--measured 5] [--pool-fraction 0.5]

set -e

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "$SCRIPT_DIR"

echo "Running transfer overhead microbenchmark..."

mvn -q compile exec:java \
    -Dexec.mainClass=com.nvidia.bench.TransferOverheadBench \
    -Dexec.classpathScope=compile \
    "-Dexec.args=$*"
