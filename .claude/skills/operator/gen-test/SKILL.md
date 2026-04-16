---
name: operator-gen-test
description: Extracts a Spark RAPIDS plugin operator's cuDF implementation into a standalone RapidsUDF and generates a comparison test against Spark's CPU SQL baseline. This is step 1 of 3 in the operator optimization workflow (extract+test -> benchmark -> optimize). Use this skill when you want to isolate and later optimize an existing `Gpu*` expression (or a specific branch / sequence of cuDF API calls within one) from the sql-plugin codebase.
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
- **Optional scope narrowing** — the user may target:
  - A specific `doColumnar` overload (unary vs. binary form).
  - A specific input-type branch (e.g. only the `StringType` branch of `GpuCast`).
  - A specific sequence of cuDF API calls inside a helper method.

If it is unclear what to extract, **ask the user for clarification.**

Derive `<OperatorName>` (CamelCase, e.g. `GpuUpper`) and `<snake_name>` (e.g. `gpu_upper`) from the target class name.

> **Note:** Commands require access to `/tmp` (Spark temp storage) and `/dev` (GPU device). If commands fail due to sandbox restrictions, re-run them unsandboxed.

> **Scope note:** Work stays entirely inside the skill project. The extracted code is a self-contained `RapidsUDF` reproduction intended for isolated testing and optimization — it is **not** back-integrated into the plugin tree. No shim updates are required.

## Step 1: Identify and Scope

1. Locate the target class in `sql-plugin/src/main/scala/` or `sql-plugin/src/main/spark<ver>/scala/`. If multiple shim versions exist, **use the latest unshimmed / highest-version copy**.
2. Identify the exact cuDF call sequence to extract. Common cases:
   - Entire `doColumnar` body (simple operators).
   - A single input-type branch of a `match`/`if` chain.
   - A private helper method used inside `doColumnar`.
3. Identify the operator's Catalyst arity and Spark input/output types. Needed to:
   - Pick the right `args: ColumnVector*` indexing in the RapidsUDF.
   - Pass the correct return `DataType` to `spark.udf.register`.

## Step 2: Set Up the Project

Copy the template project:
```bash
cp -r .claude/skills/operator/gen-test/templates <project_root>/<OperatorName>/
```

This provides a complete Maven project with all test and benchmark infrastructure.

## Step 3: Seed Test Data

Robust test data is essential — a weak test won't catch regressions during optimization.

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

**Robustness checklist — the `createTestData` DataFrame MUST cover:**
- Nulls in every nullable input column.
- Empty / zero-length values where meaningful (empty strings, empty arrays).
- Boundary values for each input type (min/max, zero, 1, -1, NaN/Infinity for floats).
- Realistic "typical" values.
- Unicode / multi-byte content for strings.
- Inputs that hit every branch the extracted cuDF code can reach (and **only** those — do not include inputs that would take a code path you did not extract).
- Stable ordering via a leading `id: Int` column.

## Step 4: Extract the cuDF Logic

Create `src/main/scala/com/udf/<OperatorName>RapidsUDF.scala`.

### 4a. File skeleton

```scala
/*
 * SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
 */
package com.udf

import ai.rapids.cudf._
import com.nvidia.spark.RapidsUDF
import com.udf.Arm.{withResource, closeOnExcept}

class <OperatorName>RapidsUDF extends RapidsUDF with Serializable {

  // Row-by-row CPU path is not used (Spark SQL is the CPU baseline).
  // Kept only because RapidsUDF is typically paired with a Function*.
  // Throw if Spark ever falls back to row-at-a-time execution.
  def apply(args: Any*): Any =
    throw new UnsupportedOperationException(
      "Extracted operator RapidsUDF does not implement CPU row-by-row execution")

  override def evaluateColumnar(numRows: Int, args: ColumnVector*): ColumnVector = {
    // TODO: Paste the extracted cuDF API call sequence here, verbatim.
    ???
  }
}
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

Open `src/test/scala/com/udf/SqlOperatorComparisonTest.scala` and implement the four TODO methods:

- `createTestData()` — return the DataFrame from Step 3.
- `cpuSqlQuery()` — the SQL expression under test, over `test_table` (e.g. `"SELECT id, upper(s) AS result FROM test_table"`).
- `gpuSqlQuery(udfName)` — the mirror query invoking the registered UDF (e.g. `s"SELECT id, $udfName(s) AS result FROM test_table"`). The output schema must match `cpuSqlQuery()`.
- `registerRapidsUDF(udfName)` — call `spark.udf.register(udfName, new <OperatorName>RapidsUDF(), <SparkReturnType>)`.

**Critical:**
- The test toggles `spark.rapids.sql.enabled` at runtime in a single SparkSession. The skill asserts both that the CPU plan contains no `Gpu*` nodes and that the GPU plan does.
- Do not hardcode expected output values — the CPU SQL run is the source of truth.

## Step 6: Run the Test

```bash
cd <OperatorName>
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
- Project directory: `<project_root>/<OperatorName>/`
- Extracted RapidsUDF: `src/main/scala/com/udf/<OperatorName>RapidsUDF.scala`
- Comparison test: `src/test/scala/com/udf/SqlOperatorComparisonTest.scala`

These outputs are required for **Step 2: Benchmark**.
