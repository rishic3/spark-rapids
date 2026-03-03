# nsys notes

`nsys profile` tries to inject libraries into and follow child processes. When profiling the JVM, nsys tries to follow spawned Python workers.

## Error

When the Python worker is running a UDF that invokes Python cuDF, the Python worker immediately dies.  
The error message that surfaces is (in local mode) a ConnectionReset, or (in standalone) segfault from cuDF's GpuArrowWriter when trying to write to the Python socket (same cause, it is the Python worker dying).

### Details

The Python worker dies before emitting a traceback. However if we capture the native error logs, the error happens during `nvtx.get_domain()` from nsys trying to capture a range around a Python cuDF call inside the UDF.  
 
```bash
==== worker launch 2026-03-03T10:33:22-08:00 ====
pid=3656910 ppid=3656422
argv=-m pyspark.daemon
python_bin=/home/rishic/anaconda3/envs/cuagent/bin/python
LD_PRELOAD=/opt/nvidia/nsight-systems/2024.6.2/target-linux-x64/libToolsInjectionProxy64.so:/opt/nvidia/nsight-systems/2024.6.2/target-linux-x64/libLinuxKeyboardInterceptorProxy.so
QUADD_ENABLE_PROFILER=
CUDA_INJECTION64_PATH=/opt/nvidia/nsight-systems/2024.6.2/target-linux-x64/libToolsInjection64.so
/home/rishic/anaconda3/envs/cuagent/lib/python3.12/site-packages/pyspark/python/lib/pyspark.zip/pyspark/daemon.py:154: DeprecationWarning: This process (pid=3656910) is multi-threaded, use of fork() may lead to deadlocks in the child.
Fatal Python error: Segmentation fault

Current thread 0x0000712f6d865780 (most recent call first):
  File "/home/rishic/anaconda3/envs/cuagent/lib/python3.12/site-packages/nvtx/nvtx.py", line 50 in get_domain
  File "/home/rishic/anaconda3/envs/cuagent/lib/python3.12/site-packages/nvtx/nvtx.py", line 108 in __init__
  File "/home/rishic/anaconda3/envs/cuagent/lib/python3.12/site-packages/cudf/utils/performance_tracking.py", line 45 in wrapper
  File "/home/rishic/Code/myforks/spark-rapids/experiments/ascii_ignore/src/ascii_ignore_gpu.py", line 47 in ascii_ignore_gpu
  File "/home/rishic/anaconda3/envs/cuagent/lib/python3.12/site-packages/pyspark/python/lib/pyspark.zip/pyspark/util.py", line 83 in wrapper
  File "/home/rishic/anaconda3/envs/cuagent/lib/python3.12/site-packages/pyspark/python/lib/pyspark.zip/pyspark/worker.py", line 136 in <lambda>
  File "/home/rishic/anaconda3/envs/cuagent/lib/python3.12/site-packages/pyspark/python/lib/pyspark.zip/pyspark/worker.py", line 1070 in <genexpr>
  File "/home/rishic/anaconda3/envs/cuagent/lib/python3.12/site-packages/pyspark/python/lib/pyspark.zip/pyspark/worker.py", line 1070 in mapper
  File "/home/rishic/anaconda3/envs/cuagent/lib/python3.12/site-packages/pyspark/python/lib/pyspark.zip/pyspark/sql/pandas/serializers.py", line 463 in init_stream_yield_batches
  File "/home/rishic/anaconda3/envs/cuagent/lib/python3.12/site-packages/pyspark/python/lib/pyspark.zip/pyspark/sql/pandas/serializers.py", line 100 in dump_stream
  File "/home/rishic/anaconda3/envs/cuagent/lib/python3.12/site-packages/pyspark/python/lib/pyspark.zip/pyspark/sql/pandas/serializers.py", line 470 in dump_stream
  File "/home/rishic/anaconda3/envs/cuagent/lib/python3.12/site-packages/pyspark/python/lib/pyspark.zip/pyspark/worker.py", line 1239 in process
  File "/home/rishic/anaconda3/envs/cuagent/lib/python3.12/site-packages/pyspark/python/lib/pyspark.zip/pyspark/worker.py", line 1247 in main
  File "/home/rishic/anaconda3/envs/cuagent/lib/python3.12/site-packages/pyspark/python/lib/pyspark.zip/pyspark/daemon.py", line 74 in worker
  File "/home/rishic/anaconda3/envs/cuagent/lib/python3.12/site-packages/pyspark/python/lib/pyspark.zip/pyspark/daemon.py", line 193 in manager
  File "/home/rishic/anaconda3/envs/cuagent/lib/python3.12/site-packages/pyspark/python/lib/pyspark.zip/pyspark/daemon.py", line 218 in <module>
  File "<frozen runpy>", line 88 in _run_code
  File "<frozen runpy>", line 198 in _run_module_as_main

Extension modules: numpy.core._multiarray_umath, (I TRUNCATED THIS)... (total: 298)
```

The root cause is still not clear to me from the log. This is what Gemini thinks:
```
1. nsys Initializes Background Threads: When the PySpark Python daemon process starts, the inherited LD_PRELOAD forces nsys to load and spin up its internal C++ background threads for tracing and NVTX state management.
2. PySpark uses fork() without exec(): To create workers rapidly, the PySpark daemon uses os.fork(). In Linux, a multi-threaded fork() only copies the calling thread into the child process; all background threads (including nsys's trace threads) are immediately killed.
3. Corrupted Profiler State: The new Python worker process now holds a "zombified," corrupted copy of nsys's internal C++ memory state and locks.
4. cuDF Triggers the Crash: Inside the UDF, the worker imports cuDF. cuDF is heavily instrumented and immediately calls nvtx.get_domain() to register its tracing domains.
5. The Segfault: Because nsys is still hooked via LD_PRELOAD, it intercepts the nvtxDomainCreate C-API call. When it tries to access its corrupted internal state (or a dead mutex) to register the domain, it triggers a Segmentation Fault (SIGSEGV).
```

## Fix

Strip profiler injection by wrapping the Python executable:

```bash
#!/bin/bash
# unprofiled_python.sh

unset LD_PRELOAD
unset QUADD_ENABLE_PROFILER
unset CUDA_INJECTION64_PATH

PYTHON_BIN="${PYSPARK_UNPROFILED_PYTHON_BIN:-/path/to/python}"
exec "${PYTHON_BIN}" "$@"
```

## Local

```bash
nsys profile --sample=none --trace=nvtx,cuda --trace-fork-before-exec=false \
  --output nsys/local_$(date +%Y%m%d-%H%M%S) \
  python run_benchmark.py --mode gpu --cluster local \
  --data-path data/bench_data_1M_rows.parquet \
  --rapids-jar-path ~/.cache/cuaether-assistant/jars/rapids-4-spark_2.12-25.12.0.jar \
  --spark-conf "spark.pyspark.python=$(pwd)/unprofiled_python.sh"
```

## Standalone

1. Start worker under nsys: `./start-nsys-worker.sh` in a terminal.
2. Run benchmark in a separate terminal:

```bash
python run_benchmark.py --mode gpu --cluster standalone \
  --data-path data/bench_data_1M_rows.parquet \
  --rapids-jar-path ~/.cache/cuaether-assistant/jars/rapids-4-spark_2.12-25.12.0.jar \
  --spark-conf "spark.pyspark.python=$(pwd)/unprofiled_python.sh"
```

3. Stop profiling worker cleanly (SIGINT), not `kill -9`.

## Debugging

`debug_python_worker.sh` is another python executable wrapper with additional native error logs/gdb.

```bash
nsys profile --trace nvtx,cuda \
  python run_benchmark.py --mode gpu --cluster local \
  --data-path data/bench_data_1M_rows.parquet \
  --rapids-jar-path ~/.cache/cuaether-assistant/jars/rapids-4-spark_2.12-25.12.0.jar \
  --spark-conf "spark.pyspark.python=$(pwd)/debug_python_worker.sh" \
  --spark-conf "spark.executorEnv.PYSPARK_PYTHON=$(pwd)/debug_python_worker.sh" \
  --spark-conf "spark.python.worker.reuse=false" \
  --spark-conf "spark.master=local[1]"
```

Worker logs go to `/tmp/spark-pyworker-debug` by default.
