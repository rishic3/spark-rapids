---
name: operator-benchmark
description: Benchmarks an extracted Spark RAPIDS operator at Spark scale (CPU SQL vs. GPU RapidsUDF) and at libcudf microbenchmark scale (GPU only). This is step 2 of 3 in the operator optimization workflow (extract+test -> benchmark -> optimize). Use after the gen-test skill has produced a passing SqlOperatorComparisonTest for the extracted operator.
model: inherit
---

# Operator Benchmark

## Workflow

- [ ] Step 1: Implement `BenchUtils` (fill in TODO methods)
- [ ] Step 2: Validate with a small dataset
- [ ] Step 3: Generate full benchmark data and run SparkBenchRunner (CPU vs. GPU)
- [ ] Step 4: Implement `MicroBenchRunner.executeGpu` and run the GPU-only microbenchmark

## Prerequisites

- Project directory from Step 1 (operator-gen-test) with passing `SqlOperatorComparisonTest`
- Extracted `<OperatorName>RapidsUDF` implemented

Derive `<OperatorName>` and `<snake_name>` from the target class name.

> **Note:** Commands require access to `/tmp` (Spark temp storage) and `/dev` (GPU device). If commands fail due to sandbox restrictions, re-run them unsandboxed.

## Step 1: Implement `BenchUtils`

Open `src/main/scala/com/udf/bench/BenchUtils.scala` and fill in the three TODO methods:

1. **`generateSyntheticData(spark, numRows, numPartitions)`** — produce a DataFrame matching the schema used in `SqlOperatorComparisonTest.createTestData`. Use `spark.range` plus `rand()`-driven expressions so it scales to tens of millions of rows. Favor realistic payload sizes — the GPU win usually grows with row size / column width.

2. **`executeCpu(spark, df)`** — the CPU path:
   - `spark.conf.set("spark.rapids.sql.enabled", "false")`
   - Register `df` as `bench_table`.
   - Run the same CPU SQL used in the test, projecting only `id` plus the operator result (avoid `SELECT *`).

3. **`executeGpu(spark, df)`** — the GPU path:
   - `spark.conf.set("spark.rapids.sql.enabled", "true")`
   - Register `new <OperatorName>RapidsUDF()` under `BenchUtils.RapidsUdfName` with the correct Spark return type.
   - Run the mirror SQL invoking the UDF.

Both `executeCpu` and `executeGpu` share the SparkSession — the toggle is a runtime config, so the plugin must be loaded in the session (`run_spark_benchmark.sh` already does this).

## Step 2: Validate

Run the small validation mode — it generates a tiny dataset and calls both `executeCpu` and `executeGpu` end-to-end to sanity-check the implementations:
```bash
chmod +x *.sh
./run_gen_data.sh --rows 1000 --validate
```

If validation fails, read the error output and fix `BenchUtils`. Typical causes: missing `bench_table` temp view, wrong return type in `spark.udf.register`, or column name mismatches between `generateSyntheticData` and the SQL query.

## Step 3: Spark Benchmark

### Generate benchmark data (10M rows):
```bash
./run_gen_data.sh --rows 10000000
```

### Run Spark benchmarks (single-JVM, shared plugin, toggle flips at runtime):
```bash
# CPU run (executeCpu sets spark.rapids.sql.enabled=false)
./run_spark_benchmark.sh --mode cpu --data-path data/bench_data_10000000_rows.parquet

# GPU run (executeGpu sets spark.rapids.sql.enabled=true)
./run_spark_benchmark.sh --mode gpu --data-path data/bench_data_10000000_rows.parquet
```

Results are saved as JSON under `results/`. Compare the `e2e_runtime` fields to report Spark-level speedup.

## Step 4: GPU Microbenchmark

> Unlike the UDF workflow, the microbenchmark here is **GPU-only** — a Spark SQL operator cannot be invoked in isolation on an in-memory cuDF `Table`, so there is no CPU analogue to time here. Use the Spark benchmark from Step 3 for CPU-vs-GPU comparisons.

### 4a. Implement `MicroBenchRunner.executeGpu`

Open `src/main/scala/com/udf/bench/MicroBenchRunner.scala` and fill in the single TODO: instantiate `<OperatorName>RapidsUDF` and call `evaluateColumnar(numRows, table.getColumn(i), ...)` with the operator's input columns.

Remember to skip the `id` column in `table` when mapping to operator args.

### 4b. Run the microbenchmark

Reuse the Parquet dataset from Step 3:
```bash
./run_micro_benchmark.sh --data-path data/bench_data_10000000_rows.parquet --rows 10000000
```

The specified number of rows is coalesced into a single cuDF table. Large table sizes (>1 GB) are typically needed to amortize kernel launch costs and surface real GPU performance.

Output shows min / median `evaluateColumnar` wall time in ms. Record this as the microbenchmark **baseline** — it is the number the optimize-cudf skill will try to beat.

## Output

Upon successful completion:
- Benchmark utilities: `src/main/scala/com/udf/bench/BenchUtils.scala`
- GPU microbenchmark: `src/main/scala/com/udf/bench/MicroBenchRunner.scala`
- Generated data: `data/`
- Spark benchmark results: `results/*.json`
- Microbenchmark baseline printed to stdout (and any `results/microbench_*.nsys-rep` if `--profile` was used)

These outputs are required for **Step 3: Optimize**.
