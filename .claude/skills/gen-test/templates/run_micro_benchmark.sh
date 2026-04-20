#!/bin/bash
# Run the GPU-only microbenchmark for the extracted RapidsUDF.
#
# Default: sweep every case subdirectory under data/ in a single JVM
#   ./run_micro_benchmark.sh                    (sweeps data/*)
#   ./run_micro_benchmark.sh --data-root data2  (sweep custom root)
#
# Single-case mode (required for --profile so the nsys report has one case):
#   ./run_micro_benchmark.sh --data-path data/long --profile

set -e

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "$SCRIPT_DIR"

print_usage() {
    echo "Usage: $0 [--data-root DIR | --data-path DIR] [--rows N] [--warmup N] [--measured N] [--pool-fraction F] [--profile]"
}

DATA_PATH=""
DATA_ROOT=""
PROFILE=""
PREV=""
for arg in "$@"; do
    case "$PREV" in
        --data-path) DATA_PATH="$arg";;
        --data-root) DATA_ROOT="$arg";;
    esac
    [ "$arg" = "--profile" ] && PROFILE="true"
    PREV="$arg"
done

if [ -z "$DATA_PATH" ] && [ -z "$DATA_ROOT" ]; then
    DATA_ROOT="data"
    set -- --data-root "$DATA_ROOT" "$@"
fi

if [ -n "$PROFILE" ] && [ -z "$DATA_PATH" ]; then
    echo "Error: --profile requires --data-path (single case)"
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
    echo "Running GPU microbenchmark sweep on ${DATA_ROOT:-$DATA_PATH}..."
    "${MVN_CMD[@]}"
fi
