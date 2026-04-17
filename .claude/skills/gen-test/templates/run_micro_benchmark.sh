#!/bin/bash
# Run in-memory GPU microbenchmark for an extracted Spark RAPIDS operator.

set -e

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "$SCRIPT_DIR"

print_usage() {
    echo "Usage: $0 --data-path PATH [--rows N] [--warmup N] [--measured N] [--pool-fraction F] [--profile]"
}

DATA_PATH=""
PROFILE=""

PREV=""
for arg in "$@"; do
    case "$PREV" in
        --data-path) DATA_PATH="$arg";;
    esac
    [ "$arg" = "--profile" ] && PROFILE="true"
    PREV="$arg"
done

if [ -z "$DATA_PATH" ]; then
    echo "Error: --data-path is required"
    print_usage
    exit 1
fi

MVN_CMD=(
    mvn -q compile exec:java
    -Dexec.mainClass=com.udf.bench.MicroBenchRunner
    -Dexec.classpathScope=compile
    "-Dexec.args=$*"
)

if [ -n "$PROFILE" ]; then
    REPORT_PATH="results/microbench_$(date +%Y%m%d_%H%M%S)"
    mkdir -p results
    echo "Running GPU microbenchmark on $DATA_PATH with nsys profiling..."
    echo "nsys report will be saved to: ${REPORT_PATH}.nsys-rep"
    nsys profile \
        -c cudaProfilerApi \
        --capture-range-end=stop \
        --trace=cuda,nvtx \
        --nvtx-domain-include="libcudf" \
        -o "$REPORT_PATH" \
        "${MVN_CMD[@]}"
else
    echo "Running GPU microbenchmark on $DATA_PATH..."
    "${MVN_CMD[@]}"
fi
