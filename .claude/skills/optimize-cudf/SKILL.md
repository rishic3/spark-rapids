---
name: operator-optimize-cudf
description: Iteratively optimizes an extracted Spark RAPIDS operator's RapidsUDF reproduction for GPU performance. Runs a loop of profiling, one targeted change, comparison test, and GPU microbenchmark until performance converges or the iteration budget is exhausted. This is step 3 of 4 in the operator optimization workflow (extract+test -> benchmark -> optimize -> backport).
model: inherit
---

# Optimize Extracted Operator (cuDF)

## Workflow

- [ ] Step 0: Create backup and establish baseline
- [ ] Steps 1-4: Iterative optimization loop (repeat up to N iterations)
  - [ ] Step 1: Profile with nsys
  - [ ] Step 2: Implement one targeted change
  - [ ] Step 3: Run `SqlOperatorComparisonTest` (fail &rarr; discard, retry)
  - [ ] Step 4: Run GPU microbenchmark (no improvement &rarr; discard, retry)
- [ ] Final: Report results

## Prerequisites

- Project directory with passing `SqlOperatorComparisonTest`
- `BenchUtils` implemented (from the **operator-benchmark** skill)
- `MicroBenchRunner.executeGpu` implemented and a baseline microbenchmark number recorded
- Benchmark data already generated under `data/` (reuse from the benchmark step)

Derive `<OperatorName>` from the target class name.

> **Note:** Commands require access to `/tmp` (Spark temp storage) and `/dev` (GPU device). If commands fail due to sandbox restrictions, re-run them unsandboxed.

> **Scope:** All edits happen inside the extracted `<OperatorName>RapidsUDF.scala` in this project. **Do not** modify the plugin source tree — this skill optimizes the isolated reproduction. Porting winning changes back to the plugin (and synchronizing any affected Spark version shims) is an out-of-scope manual step left to the user.

## Step 0: Create Backup and Establish Baseline

1. Create a working backup of the current RapidsUDF implementation:
```bash
cp src/main/scala/com/udf/<OperatorName>RapidsUDF.scala \
   src/main/scala/com/udf/<OperatorName>RapidsUDF.scala.bak
```

2. If no `.orig.bak` exists yet, save the original extracted implementation (this file is never overwritten):
```bash
cp src/main/scala/com/udf/<OperatorName>RapidsUDF.scala \
   src/main/scala/com/udf/<OperatorName>RapidsUDF.scala.orig.bak
```

3. If no baseline microbenchmark number exists for this dataset, run it now:
```bash
./run_micro_benchmark.sh --data-path data/bench_data_<rows>_rows.parquet --rows <rows>
```

Record the baseline GPU min/median time. This is the number to beat.

## Iterative Optimization Loop

Repeat up to **N iterations** (default: 10). Stop early after **3 consecutive failed attempts** with no improvement.

Maintain an **optimization log** throughout the loop: for each iteration, record what change was attempted and whether it improved, regressed, or had no effect. This prevents repeating failed approaches and feeds the final report.

### Step 1: Profile with nsys

```bash
./run_micro_benchmark.sh --data-path data/bench_data_<rows>_rows.parquet --rows <rows> --profile
```

Summarize libcudf kernel stats:
```bash
nsys stats --report nvtx_sum --format csv -o rapidsudf results/microbench_<timestamp>.nsys-rep
```

Consult **references/OPTIMIZATION_PATTERNS.md** for interpreting profiler output and identifying optimization opportunities.

> **Tip:** Profile frequently. Without profiler data, optimization changes are guesses. Try other `nsys stats --report ...` reports (e.g. `cuda_api_sum`, `cuda_kern_exec_sum`) as needed.

### Step 2: Implement One Targeted Change

Based on profiling insight (or an optimization pattern from the reference), make **one targeted change** to `<OperatorName>RapidsUDF.scala`. Isolating changes one at a time makes it possible to attribute performance impact.

Common lever categories:
- Collapse multiple cuDF API calls into fewer (e.g. regex → `stringReplace`, chained scans → single pass).
- Eliminate intermediate columns by operating on cuDF child columns directly.
- Swap to a cheaper API (the cuDF Java API often has multiple routes to the same result).

### Step 3: Run Comparison Test

```bash
mvn test -q -Dsuites=com.udf.SqlOperatorComparisonTest
```

- **Tests pass** &rarr; proceed to Step 4.
- **Tests fail** &rarr; discard changes by restoring from backup:
  ```bash
  cp src/main/scala/com/udf/<OperatorName>RapidsUDF.scala.bak \
     src/main/scala/com/udf/<OperatorName>RapidsUDF.scala
  ```
  Log the failure reason. Increment the consecutive-failure counter. Return to Step 1.

### Step 4: Run GPU Microbenchmark

```bash
./run_micro_benchmark.sh --data-path data/bench_data_<rows>_rows.parquet --rows <rows>
```

Compare the GPU min / median against the current best (from the last checkpoint).

- **Performance improved** &rarr; create a new checkpoint:
  ```bash
  cp src/main/scala/com/udf/<OperatorName>RapidsUDF.scala \
     src/main/scala/com/udf/<OperatorName>RapidsUDF.scala.bak
  ```
  Record the new best time. Reset the consecutive-failure counter. Return to Step 1.

- **Performance did NOT improve** &rarr; discard changes:
  ```bash
  cp src/main/scala/com/udf/<OperatorName>RapidsUDF.scala.bak \
     src/main/scala/com/udf/<OperatorName>RapidsUDF.scala
  ```
  Increment the consecutive-failure counter. Return to Step 1.

## Final: Report Results

After completing all iterations (or early-stopping), report:
1. **Baseline** — starting GPU min/median time.
2. **Final** — best GPU min/median time (from the last checkpoint).
3. **Speedup** — baseline / final.
4. **Successful optimizations** — what changes improved performance and by how much each.
5. **Failed optimizations** — what was attempted but did not help.
6. **Port-back note** — remind the user that the plugin source tree was NOT modified. If they want to land the optimizations upstream, run the **operator-backport** skill, which handles porting the winning `evaluateColumnar` changes into the corresponding plugin source (including any shims), running scalastyle, building the jar, and executing the relevant integration tests.

## Output

Upon successful completion:
- Optimized RapidsUDF: `src/main/scala/com/udf/<OperatorName>RapidsUDF.scala`
- Best-version backup: `src/main/scala/com/udf/<OperatorName>RapidsUDF.scala.bak`
- Original unoptimized version: `src/main/scala/com/udf/<OperatorName>RapidsUDF.scala.orig.bak`
- nsys profiles + optimization log under `results/`
