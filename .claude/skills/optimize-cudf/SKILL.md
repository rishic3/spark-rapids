---
name: operator-optimize-cudf
description: Iteratively optimizes an extracted Spark RAPIDS operator's RapidsUDF reproduction for GPU performance. Runs a loop of profiling, one targeted change, comparison test, and per-type GPU microbenchmark sweep until performance converges or the iteration budget is exhausted. This is step 3 of 4 in the operator optimization workflow (extract+test -> benchmark -> optimize -> backport). Spark-level A/B is owned by operator-backport; this loop never invokes spark-shell.
model: inherit
---

# Optimize Extracted Operator (cuDF)

## Workflow

- [ ] Step 0: Backup + establish per-type baselines
- [ ] Steps 1-4: Iterative optimization loop (repeat up to N iterations)
  - [ ] Step 1: Profile with nsys
  - [ ] Step 2: Implement one targeted change
  - [ ] Step 3: Run `SqlOperatorComparisonTest` (fail &rarr; discard, retry)
  - [ ] Step 4: Run GPU microbenchmark across **all** per-type cases (no improvement &rarr; discard, retry)
- [ ] Final: Report results

## Prerequisites

- Project from operator-gen-test with passing `SqlOperatorComparisonTest`
- Per-type Parquet datasets generated under `data/<label>/` and per-type microbenchmark baselines recorded (operator-benchmark)
- `MicroBenchRunner.executeGpu` implemented

Derive `<OperatorName>` from the target class name.

> **Note:** Commands need `/tmp` + GPU access. If they fail under the sandbox, re-run unsandboxed.

> **Scope:** All edits happen inside the extracted `<OperatorName>RapidsUDF.scala` in this project. **Do not** modify the plugin source tree — this skill optimizes the isolated reproduction. Porting winning changes back to the plugin (and any affected Spark shims) is handled by the operator-backport skill.

## Step 0: Backup and per-type baselines

1. Snapshot the current implementation:
```bash
cp src/main/scala/com/udf/<OperatorName>RapidsUDF.scala \
   src/main/scala/com/udf/<OperatorName>RapidsUDF.scala.bak
```

2. If no `.orig.bak` exists yet, save the original extracted implementation (this file is never overwritten):
```bash
cp src/main/scala/com/udf/<OperatorName>RapidsUDF.scala \
   src/main/scala/com/udf/<OperatorName>RapidsUDF.scala.orig.bak
```

3. If per-type baselines aren't already recorded from the benchmark step, sweep them now (single-JVM mode iterates every `data/<label>/` automatically):
```bash
mkdir -p results
./run_micro_benchmark.sh --rows 10000000 | tee results/baseline.txt
```

Record one baseline (median + min ms) per case from the summary table. The optimization loop uses these as the numbers to beat.

## Iterative Optimization Loop

Up to **N iterations** (default: 10). Stop early after **3 consecutive iterations with no net improvement**.

Maintain an **optimization log** throughout the loop: for each iteration record the change attempted and the per-case delta vs. baseline. This prevents repeating failed approaches and feeds the final report.

### Step 1: Profile with nsys

Profile a representative case (typically the slowest or the one with the most kernel work). `--profile` requires `--data-path` (single case) so the nsys report contains exactly one case:

```bash
./run_micro_benchmark.sh --data-path data/<label> --rows 10000000 --profile
nsys stats --report nvtx_sum --format csv -o rapidsudf results/microbench_<timestamp>.nsys-rep
```

Consult **references/OPTIMIZATION_PATTERNS.md** for interpreting profiler output and identifying optimization opportunities. If the operator's bottleneck differs by type, consider profiling more than one case.

> **Tip:** Profile frequently. Without profiler data, optimization changes are guesses. Other useful reports: `cuda_api_sum`, `cuda_kern_exec_sum`.

### Step 2: Implement one targeted change

Based on profiling insight (or an optimization pattern), make **one targeted change** to `<OperatorName>RapidsUDF.scala`. Isolating changes one at a time makes attribution possible.

Common lever categories:
- Collapse multiple cuDF API calls into fewer (e.g. regex → `stringReplace`, chained scans → single pass).
- Eliminate intermediate columns by operating on cuDF child columns directly.
- Swap to a cheaper API (the cuDF Java API often has multiple routes to the same result).
- Add a type-guarded fast path when one type benefits from a different cuDF call than another.

### Step 3: Run comparison test

```bash
mvn test -q -Dsuites=com.udf.SqlOperatorComparisonTest
```

- **Pass** &rarr; proceed to Step 4.
- **Fail** &rarr; discard and retry:
  ```bash
  cp src/main/scala/com/udf/<OperatorName>RapidsUDF.scala.bak \
     src/main/scala/com/udf/<OperatorName>RapidsUDF.scala
  ```
  Log the failure reason. Increment the consecutive-failure counter. Return to Step 1.

### Step 4: Run microbenchmark across all per-type cases

Single sweep over every case in one JVM — a change that helps one type can regress another:

```bash
./run_micro_benchmark.sh --rows 10000000 | tee "results/iter_<N>.txt"
```

Parse the per-case summary table at the bottom of the output and compare each case against the previous checkpoint's number. Decision rules:

- **Aggregate improvement AND no case regresses by more than 5%** &rarr; checkpoint:
  ```bash
  cp src/main/scala/com/udf/<OperatorName>RapidsUDF.scala \
     src/main/scala/com/udf/<OperatorName>RapidsUDF.scala.bak
  ```
  Update the recorded best per-case times. Reset the consecutive-failure counter. Return to Step 1.

- **Otherwise** (aggregate flat/regression, or any case regresses >5%) &rarr; discard:
  ```bash
  cp src/main/scala/com/udf/<OperatorName>RapidsUDF.scala.bak \
     src/main/scala/com/udf/<OperatorName>RapidsUDF.scala
  ```
  Increment the consecutive-failure counter. Return to Step 1.

## Final: Report Results

After completing all iterations (or early-stopping), report:
1. **Baseline** — per-case GPU min/median times.
2. **Final** — per-case best GPU min/median times.
3. **Per-case speedup** — baseline / final, with the geometric mean as the headline number.
4. **Successful optimizations** — what changes improved performance and by how much each.
5. **Failed optimizations** — what was attempted but did not help.
6. **Port-back note** — the plugin source tree was NOT modified. To land the optimizations upstream, run the **operator-backport** skill.

## Output

- Optimized RapidsUDF: `src/main/scala/com/udf/<OperatorName>RapidsUDF.scala`
- Best-version backup: `src/main/scala/com/udf/<OperatorName>RapidsUDF.scala.bak`
- Original unoptimized version: `src/main/scala/com/udf/<OperatorName>RapidsUDF.scala.orig.bak`
- Per-iteration per-case microbench logs + nsys profiles under `results/`
