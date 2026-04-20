---
name: operator-backport
description: Ports a winning RapidsUDF optimization from the isolated skill project back into the Spark RAPIDS plugin source tree, then validates via a `mvn verify` dist-jar build and unit + integration tests. This is step 4 of 4 in the operator optimization workflow (extract+test -> benchmark -> optimize -> backport).
model: inherit
---

# Backport Operator Optimization into spark-rapids

## Workflow

- [ ] Step 1: Identify port target + shim scope
- [ ] Step 2: Build the **baseline** dist jar (before any changes)
- [ ] Step 3: Apply the minimal diff (and update call sites if signatures changed)
- [ ] Step 4: Build the **optimized** dist jar (scalastyle runs inline via `verify`)
- [ ] Step 5: Unit + integration tests
- [ ] Step 6: Spark-level A/B benchmark (baseline vs. optimized jar) + emit PR-repro variant
- [ ] Final: Summarize with both jar paths + A/B numbers + paths to the two repro scripts; do **not** auto-commit

## Prerequisites

- Passing optimized `opt/<OperatorName>/` project from **operator-optimize-cudf**, with a recorded speedup.
- No uncommitted modifications under the plugin source trees — so any diff you apply is attributable to this skill. Untracked/modified files elsewhere (`.gitignore`, top-level `*.md`, `opt/`, etc.) are fine and should be ignored.

Respect the safety rules in `AGENTS.md` (minimal diffs, no `--no-verify`, `./build/make-scala-version-build-files.sh 2.13` if any `pom.xml` changes). Steps 4–5 need `/tmp` + GPU access; re-run unsandboxed if they fail due to sandboxing.

## Step 1: Identify Port Target + Shim Scope

Most plugin files live in a single location and fan out to all supported Spark versions via `shimplify` (see `docs/dev/shimplify.md`) — **you almost always edit one file**. Multiple dedicated copies under `sql-plugin/src/main/spark<ver>/scala/...` signal a diverged shim and per-copy treatment (`AGENTS.md` → Shim Layer Architecture).

1. From the RapidsUDF's header, identify the plugin file and method that was extracted.
2. Locate every copy of that file in `sql-plugin/src/main/`. If there's one, that's the edit target — confirm its `spark-rapids-shim-json-lines` header (if any) covers the intended Spark versions. If there are multiple, read each; they may need different adaptations (do not blindly copy-paste).
3. If the optimization changes any public signature (or deletes a helper), find every call site across the plugin tree (`sql-plugin/`, `shuffle-plugin/`, `sql-plugin-api/`, `delta-lake/`, `iceberg/`) before editing.

## Step 2: Build the Baseline Jar (before any changes)

Before editing anything, build a **baseline** dist jar at `HEAD` so the user can A/B compare against the optimized build. Target Spark 3.5.7 (`buildver=357`):

```bash
mvn clean verify -DskipTests -Dbuildver=357
mkdir -p opt/<OperatorName>/jars
cp dist/target/rapids-4-spark_2.12-<version>-cuda12.jar \
   opt/<OperatorName>/jars/rapids-4-spark_2.12-<version>-cuda12.baseline.jar
```

> **Why copy out of `dist/target/`:** Step 4 runs `mvn clean verify` again, which deletes `dist/target/` in its entirety — renaming the file in place does not save it. Stash both jars under `opt/<OperatorName>/jars/` (outside any maven build tree, alongside the bench scripts that consume them in Step 6). Record the stashed path for the summary.

## Step 3: Apply the Minimal Diff

Port the optimized cuDF sequence from `evaluateColumnar` into the plugin's `doColumnar` (or helper / branch) verbatim, adjusting only what the plugin context requires.

Preserve the file's license header, imports, and ordering. No reformatting. Update any call sites identified in Step 1. If there are multiple shim copies, apply per-copy.

## Step 4: Build the Optimized Jar

Rebuild the dist jar for Spark 3.5.7 with the changes applied, and immediately copy it to the same stash directory as the baseline. The `verify` phase runs scalastyle via antrun, so this single command covers both style checks and the build — fix any reported violations and rerun:

```bash
mvn clean verify -DskipTests -Dbuildver=357
cp dist/target/rapids-4-spark_2.12-<version>-cuda12.jar \
   opt/<OperatorName>/jars/rapids-4-spark_2.12-<version>-cuda12.optimized.jar
```

After this step you should see both jars side-by-side in the stash:

```bash
ls -lh opt/<OperatorName>/jars/
# rapids-4-spark_2.12-<ver>-cuda12.baseline.jar
# rapids-4-spark_2.12-<ver>-cuda12.optimized.jar
```

If Step 1 found multiple shim copies, also build at least one representative `buildver` from each to confirm they compile. `./build/buildall --profile=noSnapshots` is reserved for pre-merge broad coverage — don't run it every loop.

Record both stashed jar paths for the summary.

## Step 5: Unit + Integration Tests

**Scope unit tests to the operator.** `mvn test -pl tests` runs every suite under the `tests/` module (thousands of tests, many minutes) — avoid it unless nothing targeted exists. Locate the operator-specific suite(s) first:

```bash
# Suites that reference the operator's GPU class:
rg -l "Gpu<OperatorName>" tests/src/test/scala
# Suites named after the SQL function (e.g. "nvl", "coalesce"):
rg -l -i "class .*<sql_function_name>.*Suite" tests/src/test/scala
```

Then run only those suites:

```bash
mvn test -pl tests -Dbuildver=357 -DwildcardSuites=<FQN>[,<FQN>...]
```

Integration tests live in `integration_tests/src/main/python/` and consume whatever is currently in `dist/target/` — at this point that's the optimized jar from Step 4. Do not run `mvn clean` between Step 4 and the end of Step 5 (the stashed copies under `opt/<OperatorName>/jars/` survive a clean, but `run_pyspark_from_build.sh` reads from `dist/target/`).

Two environmental requirements:

- **Activate the `spark-rapids` conda env** — `run_pyspark_from_build.sh` shells out to `python` / `pyspark` / `pytest`; the env provides those plus the project's pinned test deps.
- **Set `SPARK_HOME`** — the script exits if it's unset, and it does not auto-detect. Point it at the 3.5.7 install so the Spark runtime matches the compile-time `buildver=357` shim.

Then find the relevant pytest file for the operator's SQL function (e.g. `conditionals_test.py` for `coalesce`/`nvl`) and run it:

```bash
conda activate spark-rapids
cd integration_tests
export SPARK_HOME=/opt/spark-3.5.7
TEST=src/main/python/<file>.py::<test_name> ./run_pyspark_from_build.sh
```

Add `--delta_lake` / `--iceberg` if the operator touches those paths. If the user chose a different `buildver` in Step 4, point `SPARK_HOME` at the matching install instead.

On failure, triage: a GPU-vs-CPU mismatch means revert and return to `operator-optimize-cudf`; a legitimate test update (e.g. a fallback test that no longer triggers because the optimization broadens GPU coverage) is fine — flag it in the summary.

> **If you edit the plugin diff to fix a test (Step 3 was not enough — e.g. you tightened null handling or added a type guard):** mirror the change back into `opt/<OperatorName>/src/main/scala/com/udf/<OperatorName>RapidsUDF.scala` and re-run the microbench sweep + comparison test from `opt/<OperatorName>/`:
>
> ```bash
> cd opt/<OperatorName>
> mvn test -q -Dsuites=com.udf.SqlOperatorComparisonTest
> ./run_micro_benchmark.sh --rows 10000000 | tee results/post_backport_fix.txt
> ```
>
> Use the new per-case medians (not the operator-optimize-cudf numbers) for the final summary — the original numbers measured a different implementation. Then redo Step 4 so the `*.optimized.jar` reflects the fix, and rerun Step 5 to confirm. If the post-fix microbench shows the change no longer beats baseline, surface that to the user before proceeding to Step 6.

## Step 6: Spark-Level A/B Benchmark

End-to-end Spark numbers using the actual plugin jars — this is the headline performance result for the PR. Runs `bench/spark_bench.sh` (the same self-contained script that operator-benchmark sets up) twice in one invocation, once per jar.

Prerequisites:
- `data/<label>/` Parquet datasets exist in `opt/<OperatorName>/` (from operator-benchmark Step 3 — regenerate with `./bench/gen_data.sh` if missing).
- `bench/spark_bench.sh`'s `cases` list filled in (labels match `data/`'s subdirs). If you skipped this in operator-benchmark, fill it now.
- `bench/gen_data.sh`'s `BENCH_DATA_DIR` and `bench/spark_bench.sh`'s `BENCH_DATA_DIR` agree (default `./data` works for both).

Run the A/B from inside the optimize project, pointing at the stashed jars from Steps 2 and 4:

```bash
cd opt/<OperatorName>
export SPARK_HOME=/opt/spark-3.5.7
./bench/spark_bench.sh \
    jars/rapids-4-spark_2.12-<version>-cuda12.baseline.jar \
    jars/rapids-4-spark_2.12-<version>-cuda12.optimized.jar \
    | tee results/spark_bench_ab.log
```

If the A/B shows no improvement (or a regression) at Spark scale despite a clean libcudf win in the optimize loop, that's a meaningful finding — note it for the user. Common causes: the operator is not the bottleneck of the test query (Spark/IO dominates), or AQE/codegen masks the libcudf delta.

The two bench scripts are themselves the PR-repro artifacts. Before pasting into the PR description, set `BENCH_DATA_DIR` to an absolute path in both scripts and use absolute jar paths in the cited `spark_bench.sh` invocation.

## Final: Summarize (do NOT commit)

Report:
1. Files changed (note single-copy vs. multi-shim).
2. Call sites updated.
3. **Stashed baseline + optimized jar paths** under `opt/<OperatorName>/jars/` (from Steps 2 and 4) — these survive subsequent `mvn clean` runs.
4. Additional `buildver`s built, if any.
5. Unit / integration test status.
6. Microbenchmark speedup (per-type, geomean) — note explicitly whether these are the original `operator-optimize-cudf` numbers or post-backport-fix re-runs (see Step 5's edit-then-remeasure rule).
7. **Spark-level A/B results from Step 6** (per-type baseline vs. optimized wall time, geomean speedup, log path).
8. **PR-ready repro scripts**: `opt/<OperatorName>/bench/gen_data.sh` and `opt/<OperatorName>/bench/spark_bench.sh` — the same scripts that produced the numbers, paste-ready into the PR description after the two in-place edits called out in Step 6 (absolute `BENCH_DATA_DIR` and absolute jar paths in the cited invocation).
9. Open questions for the user.

**Do not `git commit` or `git push`.** Committing (with `-s` for DCO) and PR authoring (including `[databricks]` / `[skip ci]` tags, no-rebase-during-review) are the user's call per `AGENTS.md`.

## Output

- Modified plugin source file(s) (and any updated call sites).
- **Baseline jar** at `opt/<OperatorName>/jars/rapids-4-spark_2.12-<version>-cuda12.baseline.jar` (pre-change, Spark 3.5.7).
- **Optimized jar** at `opt/<OperatorName>/jars/rapids-4-spark_2.12-<version>-cuda12.optimized.jar` (post-change, Spark 3.5.7).
- Green `mvn verify` (scalastyle + build) and passing unit + integration tests.
- Spark-level A/B benchmark log at `opt/<OperatorName>/results/spark_bench_ab.log`.
- Self-contained repro scripts at `opt/<OperatorName>/bench/gen_data.sh` and `opt/<OperatorName>/bench/spark_bench.sh` (paste-ready into the PR description after making `BENCH_DATA_DIR` and jar paths absolute).
- Summary ready for user review.
