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
- [ ] Final: Summarize with both jar paths; do **not** auto-commit

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
cp dist/target/rapids-4-spark_2.12-<version>-cuda12.jar \
   dist/target/rapids-4-spark_2.12-<version>-cuda12.baseline.jar
```

Copy is required — the next build step overwrites `dist/target/` contents. Record the baseline path for the summary.

## Step 3: Apply the Minimal Diff

Port the optimized cuDF sequence from `evaluateColumnar` into the plugin's `doColumnar` (or helper / branch) verbatim, adjusting only what the plugin context requires.

Preserve the file's license header, imports, and ordering. No reformatting. Update any call sites identified in Step 1. If there are multiple shim copies, apply per-copy.

## Step 4: Build the Optimized Jar

Rebuild the dist jar for Spark 3.5.7 with the changes applied, and save a labeled copy alongside the baseline. The `verify` phase runs scalastyle via antrun, so this single command covers both style checks and the build — fix any reported violations and rerun:

```bash
mvn clean verify -DskipTests -Dbuildver=357
cp dist/target/rapids-4-spark_2.12-<version>-cuda12.jar \
   dist/target/rapids-4-spark_2.12-<version>-cuda12.optimized.jar
```

If Step 1 found multiple shim copies, also build at least one representative `buildver` from each to confirm they compile. `./build/buildall --profile=noSnapshots` is reserved for pre-merge broad coverage — don't run it every loop.

Record both the baseline and optimized jar paths for the summary.

## Step 5: Unit + Integration Tests

```bash
mvn test -pl tests                                       # unit tests
mvn test -pl tests -Dsuites=<FullyQualifiedSuiteName>    # scoped to operator's suite, if any
```

Integration tests live in `integration_tests/src/main/python/` and consume the optimized dist jar from Step 4 automatically. `run_pyspark_from_build.sh` **requires `SPARK_HOME` to be set** — it will exit otherwise, and it does not auto-detect. Point it at the 3.5.7 install so the Spark runtime matches the compile-time `buildver=357` shim, then find the relevant pytest file for the operator's SQL function (e.g. `conditionals_test.py` for `coalesce`/`nvl`) and run it:

```bash
cd integration_tests
export SPARK_HOME=/opt/spark-3.5.7
TEST=src/main/python/<file>.py::<test_name> ./run_pyspark_from_build.sh
```

Add `--delta_lake` / `--iceberg` if the operator touches those paths. If the user chose a different `buildver` in Step 4, point `SPARK_HOME` at the matching install instead.

On failure, triage: a GPU-vs-CPU mismatch means revert and return to `operator-optimize-cudf`; a legitimate test update (e.g. a fallback test that no longer triggers because the optimization broadens GPU coverage) is fine — flag it in the summary.

## Final: Summarize (do NOT commit)

Report:
1. Files changed (note single-copy vs. multi-shim).
2. Call sites updated.
3. **Baseline jar path** (`*.baseline.jar`, from Step 2) + **optimized jar path** (`*.optimized.jar`, from Step 4) — side-by-side, so the user can A/B test.
4. Additional `buildver`s built, if any.
5. Unit / integration test status.
6. Speedup carried from `operator-optimize-cudf`.
7. Open questions for the user.

**Do not `git commit` or `git push`.** Committing (with `-s` for DCO) and PR authoring (including `[databricks]` / `[skip ci]` tags, no-rebase-during-review) are the user's call per `AGENTS.md`.

## Output

- Modified plugin source file(s) (and any updated call sites).
- **Baseline jar** at `dist/target/rapids-4-spark_2.12-<version>-cuda12.baseline.jar` (pre-change, Spark 3.5.7).
- **Optimized jar** at `dist/target/rapids-4-spark_2.12-<version>-cuda12.optimized.jar` (post-change, Spark 3.5.7).
- Green `mvn verify` (scalastyle + build) and passing unit + integration tests.
- Summary ready for user review.
