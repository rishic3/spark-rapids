---
name: operator-gen-test
description: Extracts a Spark RAPIDS plugin operator's cuDF implementation into a standalone RapidsUDF and generates a comparison test against Spark's CPU SQL baseline. This is step 1 of 4 in the operator optimization workflow (extract+test -> benchmark -> optimize -> backport). Use this skill when you want to isolate and later optimize an existing `Gpu*` expression (or a specific branch / sequence of cuDF API calls within one) from the sql-plugin codebase.
model: inherit
---

# Extract Operator + Generate Comparison Test

## Workflow

- [ ] Step 1: Identify and scope the target operator
- [ ] Step 2: Set up the project (copy template)
- [ ] Step 3: Seed test data from the repo (or build from scratch)
- [ ] Step 4: Extract the cuDF logic into `<Name>RapidsUDF`
- [ ] Step 5: Fill in `SqlOperatorComparisonTest`
- [ ] Step 6: Run the test and iterate until passing
- [ ] Step 7: Verify outputs

## Prerequisites

The user should provide:

- **Target operator** — a `Gpu*` expression class in `sql-plugin/src/main/scala/` (e.g. `GpuUpper`, `GpuSubstring`, `GpuHex`). Usually extends `GpuUnaryExpression`, `GpuBinaryExpression`, or `GpuTernaryExpression` and implements `doColumnar`.
- **SQL function name** — how the operator is invoked in Spark SQL (e.g. `upper`, `substring`, `hex`). This is the CPU baseline.
- **Optional scope narrowing** — the user may target a subset of the operator:
  - A specific `doColumnar` overload (unary vs. binary form).
  - A specific input-type branch (e.g. only the `StringType` branch of `GpuCast`).

If it is unclear what to extract, **ask the user for clarification.**

> **Granularity principle:** The extraction unit is always a whole operator invocation, not a helper method. Even when the planned optimization lives inside a private helper (e.g. `NullUtilities.mergeNulls`, `GpuDivModLike.replaceZeroWithNull`, `boolInverted`), extract the *enclosing operator's* `doColumnar` and keep the helper in place. The helper gets modified during the optimize-cudf loop, but measurements must be at operator granularity — a 10x speedup in a helper that is 2% of the operator is a 2% operator-level improvement, and isolated helper microbenchmarks would misrepresent the real impact.

Derive `<OperatorName>` (CamelCase, e.g. `GpuUpper`) and `<snake_name>` (e.g. `gpu_upper`) from the target class name.

> **Note:** Commands require access to `/tmp` (Spark temp storage) and `/dev` (GPU device). If commands fail due to sandbox restrictions, re-run them unsandboxed.

> **Scope note:** Work stays entirely inside the skill project. The extracted code is a self-contained `RapidsUDF` reproduction intended for isolated testing and optimization — it is **not** back-integrated into the plugin tree. No shim updates are required.

> **Spark version:** The template pom pins `<spark.version>3.5.7</spark.version>`, matching the `-Dbuildver=357` default used by **operator-backport** (compile-time shim) and the `/opt/spark-3.5.7` install. Leave this alone unless the user explicitly targets a different Spark version, in which case update `<spark.version>` in the scaffolded project's `pom.xml` and make sure the matching Spark is installed.

## Step 1: Identify and Scope

1. Locate the target class in `sql-plugin/src/main/scala/` or `sql-plugin/src/main/spark<ver>/scala/`. If multiple shim versions exist, **use the latest unshimmed / highest-version copy**.
2. Identify the extraction unit. This is always an operator-sized body — either:
   - The entire `doColumnar` body (simple operators), or
   - A single input-type branch of a `match`/`if` chain inside `doColumnar` (e.g. only the `StringType` branch of `GpuCast`).
   Do **not** extract a private helper alone. If the intended optimization lives inside a helper, extract the whole `doColumnar` (or whole branch) and pull in the helper as-is — the helper is modified later during optimize-cudf, but benchmarked as part of the full operator.
3. Identify the operator's Catalyst arity and Spark input/output types. Needed to:
   - Pick the right `args: ColumnVector*` indexing in the RapidsUDF.
   - Pass the correct return `DataType` to `spark.udf.register`.
   - **Capture the full advertised type surface** by looking up the operator's `ExprChecks` (or `ExecChecks`) entry in `sql-plugin/src/main/scala/com/nvidia/spark/rapids/GpuOverrides.scala`. This is the contract the plugin promises Spark — every type listed there (including `TypeSig.ARRAY`, `TypeSig.STRUCT`, `TypeSig.MAP`, `TypeSig.BINARY`, `.nested()`, decimal precisions, etc.) must be exercised by the test data in Step 3. Optimizations that work for primitives but break on, say, struct columns are common and pass primitive-only tests silently — record the full set now so the test data covers it.
4. **If the operator does not fit the templates** — e.g. it is variadic (`GpuCoalesce`, `GpuGreatest`, `GpuConcat`), does not implement `doColumnar` (evaluates directly on a `ColumnarBatch`), operates on nested types in ways the `args: ColumnVector*` signature cannot express, or requires scalar-literal fast paths that disappear behind a `RapidsUDF` registration — **stop and surface this to the user before extracting**. Describe the mismatch and what you would have to simplify or simulate; let the user decide whether to proceed, narrow the scope, or pick a different target. Do not try to shoehorn a bad fit into the template.

## Step 2: Set Up the Project

Copy the template project:
```bash
cp -r .claude/skills/gen-test/templates <project_root>/opt/<OperatorName>/
```

This provides a complete Maven project with all test and benchmark infrastructure.

## Step 3: Seed Test Data

Robust test data is essential — a weak test won't catch regressions during optimization.

> **Test data ≠ benchmark data.** This step seeds a small (10s–100s of rows) hand-curated DataFrame inside the test, focused on edge cases and full type coverage for correctness. The bulk Parquet datasets used by microbenchmarks and Spark-level A/B are generated separately in **operator-benchmark** Step 3 via `DBGen` (one Parquet directory per advertised type). Both datasets must agree on the operator's input schema (column names + types per type case), but the contents are otherwise independent.

**First, check the repo for existing coverage:**

1. `integration_tests/src/main/python/` — search for pytest cases that invoke the SQL function. Good starting files by domain:
   - Strings: `string_test.py`, `regexp_test.py`, `url_test.py`
   - Numerics: `arithmetic_ops_test.py`, `cast_test.py`, `math_test.py`
   - Dates/times: `date_time_test.py`
   - JSON/XML: `get_json_test.py`, `json_test.py`, `xml_test.py`
   - Collections: `array_test.py`, `map_test.py`
   - Hashes: `hash_test.py`
2. `tests/src/test/scala/` — search for unit/suite tests that reference the `Gpu<Name>` class or the SQL function by name.
3. `integration_tests/src/main/python/data_gen.py` — reusable data generators with seeds; good inspiration for synthetic data shape.

If you find relevant cases, port their inputs and edge cases into `createTestData`. If nothing applies (or the existing coverage is thin), construct a robust dataset yourself.

**Robustness checklist — `createTestData` returns `Seq[DataFrame]`, one DataFrame per advertised type. Every DataFrame MUST cover:**
- Nulls in every nullable input column.
- Empty / zero-length values where meaningful (empty strings, empty arrays).
- Boundary values for each input type (min/max, zero, 1, -1, NaN/Infinity for floats).
- Realistic "typical" values.
- Unicode / multi-byte content for strings.
- Inputs that hit every branch the extracted cuDF code can reach (and **only** those — do not include inputs that would take a code path you did not extract).
- Stable ordering via a leading `id: Int` column.
- Correct Scala types when building `Row(...)`s for nested columns: **`Seq[T]`** for `ArrayType` (NOT `java.util.List` — Spark's encoder throws `ClassCastException`), **`Row(...)`** for `StructType`, **`new java.math.BigDecimal(...)`** for `DecimalType`.

**Across the list, you MUST cover every type the operator's `ExprChecks` advertises** — one DataFrame per type. The per-DataFrame test loop makes this structurally enforced rather than aspirational; skipping a type here is the same gap that lets primitive-only optimizations silently break on nested types.

## Step 4: Extract the cuDF Logic

Create `src/main/scala/com/udf/<OperatorName>RapidsUDF.scala`.

### 4a. File skeleton

Use Java `UDF<N>` (from `org.apache.spark.sql.api.java`). Because `UDF<N>[T1, ..., R]` binds a single `(T1, ..., R)` triple, one subclass cannot cover multiple advertised types. Put the cuDF work on a shared trait and create one thin `UDF<N>` subclass per advertised type — cuDF's `evaluateColumnar` dispatches on the column's runtime `DType`, not on the Java generics.

Use **boxed** Java types (`java.lang.Long`, not `Long`) so nulls survive Spark ↔ Java boundary. Mapping for the common advertised types:

| Spark type     | UDF type parameter          |
| -------------- | --------------------------- |
| `LongType`     | `java.lang.Long`            |
| `IntegerType`  | `java.lang.Integer`         |
| `StringType`   | `String`                    |
| `DecimalType`  | `java.math.BigDecimal`      |
| `ArrayType(T)` | `java.util.List[<boxed T>]` |
| `StructType`   | `org.apache.spark.sql.Row`  |

```scala
/*
 * SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
 */
package com.udf

import ai.rapids.cudf._
import com.nvidia.spark.RapidsUDF
import com.udf.Arm.{withResource, closeOnExcept}
import org.apache.spark.sql.Row
import org.apache.spark.sql.api.java.UDF<N>

// evaluateColumnar is polymorphic over cuDF type — put it on a shared trait.
trait <OperatorName>RapidsUDFBase extends RapidsUDF with Serializable {
  override def evaluateColumnar(numRows: Int, args: ColumnVector*): ColumnVector = {
    require(args.length == <N>, s"expects <N> columns, got ${args.length}")
    // TODO: Paste the extracted cuDF API call sequence here, verbatim.
    ???
  }
}

// One thin subclass per advertised type (only for the Java UDF<N> typing —
// the `call` body is never actually invoked). See operator's ExprChecks.
class <OperatorName>LongUDF
    extends UDF<N>[java.lang.Long, ..., java.lang.Long]
    with <OperatorName>RapidsUDFBase {
  override def call(<args>): java.lang.Long =
    throw new UnsupportedOperationException("CPU row-by-row path not used")
}
// ... one subclass per advertised type (String, Decimal, ArrayLong, Struct, ...)
```

### 4b. Extraction rules

- **Copy the cuDF API call sequence exactly as it appears in the plugin source.** Do not refactor or re-order operations during extraction — the point is to produce a faithful reproduction so later optimization measurements are attributable.
- Pull in any private helpers the extracted code calls (re-home them under `com.udf` and strip unused imports).
- Replace plugin-specific wrappers with their cuDF equivalents:
  - `input.getBase` → the input `ColumnVector` directly (`args(i)`).
  - `GpuScalar.from(x, dt)` / `GpuColumnVector.from(...)` → plain `Scalar.from*` / `ColumnVector` constructors.
- Preserve the original arg-count / type `require` assertions (or add your own if the original lacked them).
- Use `withResource` / `closeOnExcept` from `com.udf.Arm` for all intermediate cuDF resources. **Never** put the input `ColumnVector`s from `args` into `withResource` — the framework (or, here, the test harness) closes them.
- **NEVER** call `copyToHost()`, `TableDebug.debug(...)`, or any GPU→CPU path in the final version.

Consult `.claude/skills/udf/convert-to-cudf/references/RAPIDS_UDF.md` for the full RapidsUDF contract (type mapping, memory rules, debugging tips) — the same rules apply here.

## Step 5: Fill in `SqlOperatorComparisonTest`

Implement the four TODO methods in `src/test/scala/com/udf/SqlOperatorComparisonTest.scala` — the in-file docstrings document each method's contract and show worked examples. The template auto-generates one `test(...)` per DataFrame returned by `createTestData()`.

Two cross-cutting invariants the docstrings don't restate:

- The CPU SQL run is the source of truth — never hardcode expected outputs.
- The body toggles `spark.rapids.sql.enabled` per-case in one `SparkSession`; the assertions pin the plan side (CPU plan free of `Gpu*` nodes, GPU plan contains them). If your `gpuSqlQuery` or `registerRapidsUDF` doesn't actually flip the plan, the plan assertion will fire before the value comparison.

## Step 6: Run the Test

```bash
cd opt/<OperatorName>
mvn test -q -Dsuites=com.udf.SqlOperatorComparisonTest
```

If it fails, analyze stdout/stderr and iterate on the extracted RapidsUDF or the test data. Common failures:

- **Schema mismatch** between CPU and GPU result — fix the UDF return type or adjust `gpuSqlQuery` casts.
- **Null-handling divergence** — Spark's SQL null semantics must be matched inside `evaluateColumnar`.
- **Value mismatch on a specific branch** — narrow the extraction scope (Step 1) or expand the extraction to include a helper you missed.
- **Plan assertion fails** — the SQL you wrote doesn't actually flip plans with the toggle. Check `gpuSqlQuery` references the registered UDF (which forces the GPU path) and `cpuSqlQuery` uses only pure Spark functions.

## Step 7: Verify Outputs

After the test passes:

1. `createTestData` covers the full robustness checklist from Step 3.
2. `evaluateColumnar` is a faithful copy of the plugin's cuDF logic — no drive-by changes.
3. No `copyToHost()`, `TableDebug`, or stray debug statements.
4. No GPU resources leaked (run once with `-Ddebug.memory.leaks=true` as a sanity check; see Step 4 of `convert-to-cudf`).

## Output

Upon successful completion:
- Project directory: `<project_root>/opt/<OperatorName>/`
- Extracted RapidsUDF: `src/main/scala/com/udf/<OperatorName>RapidsUDF.scala`
- Comparison test: `src/test/scala/com/udf/SqlOperatorComparisonTest.scala`

These outputs are required for **Step 2: Benchmark**.
