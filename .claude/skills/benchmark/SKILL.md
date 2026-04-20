---
name: operator-benchmark
description: Generates per-type Parquet datasets via DBGen (`bench/gen_data.sh`) and runs the libcudf-only microbenchmark (`MicroBenchRunner`) on the extracted RapidsUDF for every type the operator advertises. This is step 2 of 4 in the operator optimization workflow (extract+test -> benchmark -> optimize -> backport). Use after the gen-test skill has produced a passing `SqlOperatorComparisonTest`. Spark-level A/B is owned by the backport skill and is not part of this loop.
model: inherit
---

# Operator Benchmark

## Workflow

- [ ] Step 1: Build (or locate) the `datagen` jar
- [ ] Step 2: Fill in `bench/gen_data.sh` AND `bench/spark_bench.sh` (labels must match)
- [ ] Step 3: Generate Parquet datasets (one directory per type)
- [ ] Step 4: Implement `MicroBenchRunner.executeGpu`
- [ ] Step 5: Run the GPU microbenchmark on each per-type dataset and record the baselines

## Prerequisites

- Project from operator-gen-test with passing `SqlOperatorComparisonTest`
- Extracted `<OperatorName>RapidsUDF` implemented
- Spark 3.5.7 installed; `SPARK_HOME` will be set in Step 3
- Type set captured from the operator's `ExprChecks` (operator-gen-test Step 1.3)

> Steps 3 and 5 need `/tmp` + GPU access. If commands fail due to sandbox restrictions, re-run them unsandboxed.

## Step 1: Build the `datagen` jar

`bench/gen_data.sh` uses `DBGen` from the spark-rapids `datagen` module. Build the jar against Spark 3.5.7 if it isn't already built. From the spark-rapids repo root:

```bash
mvn package -pl datagen -Dbuildver=357 -DskipTests
```

Output: `datagen/target/datagen_2.12-<version>-spark357.jar`. Export it for the next steps:

```bash
export DATAGEN_JAR=$(ls -t $(git rev-parse --show-toplevel)/datagen/target/datagen_2.12-*-spark357.jar | head -n1)
export SPARK_HOME=/opt/spark-3.5.7
```

The glob filters on `spark357` so a stale `spark330`/`spark341` jar in `target/` won't match. `ls -t` picks the newest if multiple `spark357` builds exist — but if you're unsure, `rm datagen/target/*.jar && mvn package ...` is the safe reset.

## Step 2: Fill in `bench/gen_data.sh` and `bench/spark_bench.sh`

Fill both case lists now — they must agree on labels, and filling them together avoids a later scramble in operator-backport Step 6.

**`bench/gen_data.sh`** — replace `cases: Seq[(String, String)] = ???` with one `(label, ddl)` entry per advertised type. The DDL describes ALL input columns of the operator (binary `nvl(a, b)` over `long` → `"a long, b long"`). The `label` becomes the Parquet subdirectory name — use filename-safe lowercase (`long`, `string`, `decimal_38`, `array_long`, `struct_2`, ...).

**`bench/spark_bench.sh`** — replace `cases: Seq[(String, String)] = ???` with one `(label, sqlFragment)` entry per case. Labels MUST match `gen_data.sh`. The fragment is the projection list inside `SELECT <fragment> FROM bench` — wrap the operator in `COUNT(...)` to force row-wise work (the example in the file shows this).

> The example in `bench/gen_data.sh` (long / string / decimal_38 / array_long / struct_2) is a **representative subset**, not the full surface. Many operators advertise more types (boolean, byte, short, int, float, double, date, timestamp, binary, map, ...). Consult the operator's `ExprChecks` entry captured in operator-gen-test Step 1.3 and add any missing types.

Match the column NAMES to whatever `<OperatorName>RapidsUDF.evaluateColumnar(...)` expects positionally. The microbench reads them by index; `spark_bench.sh` references them by name in its SQL.

> Defaults: `BENCH_DATA_DIR=./data` so the parquet output lands where `MicroBenchRunner` reads from. Leave this alone for the optimize loop. The PR-repro override (an absolute path like `/tmp/<snake_name>_bench`) is documented in operator-backport.

## Step 3: Generate Parquet datasets

```bash
chmod +x bench/gen_data.sh
./bench/gen_data.sh                              # 10M rows per case (default)
BENCH_ROWS=50000000 ./bench/gen_data.sh          # larger sweep
```

Result: one Parquet directory per case under `data/<label>/`. Sanity-check the layout:

```bash
ls data/
du -sh data/*
```

Expected size per case: ~80MB–1GB depending on type and row count. Tens-of-millions-of-rows datasets are typical to amortize kernel launch costs.

## Step 4: Implement `MicroBenchRunner.executeGpu`

Open `src/main/scala/com/udf/bench/MicroBenchRunner.scala` and fill the single TODO: instantiate `<OperatorName>RapidsUDF` and call `evaluateColumnar(numRows, table.getColumn(0), table.getColumn(1), ...)` with the operator's input columns in the same order as the columns listed in `bench/gen_data.sh`'s DDL.

Note: `bench/gen_data.sh` writes only the operator's input columns (no synthetic `id` column), so column indices start at 0.

## Step 5: Run the microbenchmark across all types

Default invocation sweeps every subdirectory of `data/` in a single JVM (one RMM pool, one cuDF native load — much faster than per-case `mvn` startups):

```bash
chmod +x run_micro_benchmark.sh
./run_micro_benchmark.sh --rows 10000000
```

Output: per-case `[label] GPU | rows | median ... | min ...` lines plus a final summary table. Record every per-case median/min as the **microbenchmark baselines** — operator-optimize-cudf will try to beat each one.

> **Sanity-check the numbers before trusting them.** A median in the low tens of microseconds on 10M rows (~1 ns/row) is below realistic GPU memory-bandwidth throughput — something isn't doing real work. For null-handling operators the most common cause is `DBGen`'s default of 0% nulls (see `datagen/README.md` §Nulls), which leaves the operator's null branch unexercised. To inject nulls, edit the heredoc in `bench/gen_data.sh` to capture the table handle and call `setNullProbability` before `toDF`, e.g.
>
> ```scala
> val t = DBGen().addTable("data", ddl, numRows)
> t("a").setNullProbability(0.3); t("b").setNullProbability(0.3)
> t.toDF(spark).write.mode("overwrite").parquet(path)
> ```
>
> then regenerate and re-run Step 5.

For nsys profiling, narrow to one case (mixing cases in a single nsys report is rarely useful):

```bash
./run_micro_benchmark.sh --data-path data/long --rows 10000000 --profile
```

## Output

- Filled `bench/gen_data.sh` (the case list defines the type sweep for the rest of the workflow and is reused verbatim by operator-backport's PR repro snippet)
- Per-type Parquet datasets under `data/<label>/`
- Implemented `src/main/scala/com/udf/bench/MicroBenchRunner.scala`
- Per-type microbenchmark baselines (median / min ms) recorded for operator-optimize-cudf
- Optional `results/microbench_*.nsys-rep` reports

These outputs are required for **Step 3: Optimize**.
