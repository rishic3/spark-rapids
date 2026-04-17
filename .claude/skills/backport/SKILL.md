---
name: operator-backport
description: Ports a winning RapidsUDF optimization from the isolated skill project back into the Spark RAPIDS plugin source tree, then validates via scalastyle, dist-jar build, unit tests, and integration tests. This is step 4 of 4 in the operator optimization workflow (extract+test -> benchmark -> optimize -> backport).
model: inherit
---

# Backport Operator Optimization into spark-rapids

## Workflow

- [ ] Step 1: Identify port target + shim scope
- [ ] Step 2: Apply the minimal diff (and update call sites if signatures changed)
- [ ] Step 3: Scalastyle
- [ ] Step 4: Build the dist jar
- [ ] Step 5: Unit + integration tests
- [ ] Final: Summarize with jar path; do **not** auto-commit

## Prerequisites

- Passing optimized `opt/<OperatorName>/` project from **operator-optimize-cudf**, with a recorded speedup.
- Clean `git status` in the plugin tree.

Respect the safety rules in `AGENTS.md` (minimal diffs, no `--no-verify`, `./build/make-scala-version-build-files.sh 2.13` if any `pom.xml` changes). Steps 4–5 need `/tmp` + GPU access; re-run unsandboxed if they fail due to sandboxing.

## Step 1: Identify Port Target + Shim Scope

Most plugin files live in a single location and fan out to all supported Spark versions via `shimplify` (see `docs/dev/shimplify.md`) — **you almost always edit one file**. Multiple dedicated copies under `sql-plugin/src/main/spark<ver>/scala/...` signal a diverged shim and per-copy treatment (`AGENTS.md` → Shim Layer Architecture).

1. From the RapidsUDF's header, identify the plugin file and method that was extracted.
2. Locate every copy of that file in `sql-plugin/src/main/`. If there's one, that's the edit target — confirm its `spark-rapids-shim-json-lines` header (if any) covers the intended Spark versions. If there are multiple, read each; they may need different adaptations (do not blindly copy-paste).
3. If the optimization changes any public signature (or deletes a helper), find every call site across the plugin tree (`sql-plugin/`, `shuffle-plugin/`, `sql-plugin-api/`, `delta-lake/`, `iceberg/`) before editing.

## Step 2: Apply the Minimal Diff

Port the optimized cuDF sequence from `evaluateColumnar` into the plugin's `doColumnar` (or helper / branch) verbatim, adjusting only what the plugin context requires.

Preserve the file's license header, imports, and ordering. No reformatting. Update any call sites identified in Step 1. If there are multiple shim copies, apply per-copy.

## Step 3: Scalastyle

```bash
mvn scalastyle:check -pl sql-plugin
```

Fix the reported violations.

## Step 4: Build the dist Jar

The primary deliverable is the full dist jar at `dist/target/rapids-4-spark_2.{12,13}-<version>-cuda12.jar` — this is what `run_pyspark_from_build.sh` picks up and what a user would deploy.

```bash
mvn clean verify -DskipTests                 # default buildver (330)
mvn clean verify -DskipTests -Dbuildver=341  # specific Spark version
```

If Step 1 found multiple shim copies, build at least one representative `buildver` from each. `./build/buildall --profile=noSnapshots` is reserved for pre-merge broad coverage — don't run it every loop.

Record the resolved jar path for the summary.

## Step 5: Unit + Integration Tests

```bash
mvn test -pl tests                                       # unit tests
mvn test -pl tests -Dsuites=<FullyQualifiedSuiteName>    # scoped to operator's suite, if any
```

Integration tests live in `integration_tests/src/main/python/` and consume the Step 4 dist jar automatically. Find the relevant pytest file for the operator's SQL function (e.g. `conditionals_test.py` for `coalesce`/`nvl`) and run it:

```bash
cd integration_tests
TEST=src/main/python/<file>.py::<test_name> ./run_pyspark_from_build.sh
```

Add `--delta_lake` / `--iceberg` if the operator touches those paths.

On failure, triage: a GPU-vs-CPU mismatch means revert and return to `operator-optimize-cudf`; a legitimate test update (e.g. a fallback test that no longer triggers because the optimization broadens GPU coverage) is fine — flag it in the summary.

## Final: Summarize (do NOT commit)

Report:
1. Files changed (note single-copy vs. multi-shim).
2. Call sites updated.
3. `buildver`s built + dist jar path.
4. Scalastyle / unit / integration test status.
5. Speedup carried from `operator-optimize-cudf`.
6. Open questions for the user.

**Do not `git commit` or `git push`.** Committing (with `-s` for DCO) and PR authoring (including `[databricks]` / `[skip ci]` tags, no-rebase-during-review) are the user's call per `AGENTS.md`.

## Output

- Modified plugin source file(s) (and any updated call sites).
- **Dist jar**: `dist/target/rapids-4-spark_2.{12,13}-<version>-cuda12.jar`.
- Clean scalastyle, passing unit + integration tests.
- Summary ready for user review.
