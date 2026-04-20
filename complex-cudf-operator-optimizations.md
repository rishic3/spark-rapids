# Spark-RAPIDS: Complex-cuDF Operator Optimization Candidates

This document catalogues **specific, isolated operator optimization targets** identified by surveying the Spark-RAPIDS source code (prioritizing files with high `withResource` counts as a proxy for multi-kernel complexity) and drilling into the cuDF call graphs.

These are all optimizations **not covered in detail** in `synthesized-optimization-report.md`, and each is workload-agnostic — the inefficiency is structural to the operator's implementation, not data-dependent.

Candidates are grouped by subsystem and tagged with the specific kernels executed today, the proposed reduction, and NDS relevance.

---

## Tier A: Math / Arithmetic — kernel-count blow-ups from Spark semantics

### A1. `GpuHypot` — hypot(x, y) runs ~8 kernels to handle NaN/Inf/0 edge cases
`sql-plugin/src/main/scala/org/apache/spark/sql/rapids/mathExpressions.scala` (≈ lines 548–621)

**What it does today**: Spark's semantics require `hypot(±Inf, NaN) = Inf` (differing from libm). The implementation stabilizes `sqrt(a²+b²)` as `max·sqrt(1+(min/max)²)` and then patches the result with `ifElse` branches for `(0,0)`, `(NaN,*)`, `(Inf,*)`, null:

```
abs(a), abs(b), greaterThan, ifElse(max), ifElse(min), div(min/max),
mul(ratio²), add(1+ratio²), sqrt, mul(max·sqrt), ifElse(Inf cases), ifElse(0 cases)
```

That's ~8–10 kernel launches and 6+ intermediate FP columns per row pair.

**Opportunity**: Fuse the entire `hypot` into a single CUDA kernel in spark-rapids-jni that does the FMA-stabilized formula with the edge-case branches in registers, identical to how `CastStrings` is implemented. A single-pass kernel will be bandwidth-bound (~2 reads, 1 write) rather than kernel-launch-bound.

**NDS relevance**: Low directly (`hypot` uncommon), but the *pattern* (multi-branch patch-up of a mathematical formula for Spark semantics) also governs `GpuAcoshCompat`, `GpuAsinhCompat`, `GpuRemainder`, `GpuExpm1`, `GpuLog1p` — so a reusable JNI macro/template would amortize across all of them.

---

### A2. `GpuAcoshCompat` / `GpuAsinhCompat` — 5 kernels for a single unary op
`sql-plugin/src/main/scala/org/apache/spark/sql/rapids/mathExpressions.scala` (lines 105–189)

**What it does today**: Reimplements `acosh(x)` as `log(x + sqrt(x·x − 1))` on GPU using:
```
mul(x, x), sub(x² − 1), sqrt, add(x + sqrt), log
```
cuDF has a native `UnaryOp.ARCCOSH` (and `ARCSINH`) which the Spark-RAPIDS code explicitly avoids because of precision differences in corner cases.

**Opportunity**: Either (a) benchmark the native unary op against the formula — for bulk float data, the corner-case deltas may be below Spark's tolerance for specific configurations; or (b) write a fused JNI kernel that dispatches to libm's `acoshf`/`acosh` (which matches Spark). Either path collapses 5 kernels into 1.

---

### A3. `GpuLeast` / `GpuGreatest` with FloatType/DoubleType — O(4N) kernels
`sql-plugin/src/main/scala/org/apache/spark/sql/rapids/arithmetic.scala` (lines 1359–1420, specifically `combineButNoCloseFp`)

**What it does today**: For N float/double inputs, the current implementation folds pairwise:
```
for each pair (a, b):
  isNan(a), isNan(b), or, ifElse(nan vs non-nan), NULL_MIN/NULL_MAX  ≈ 4 kernels
```
giving **~4·(N−1) kernels** just to honor Spark's "NaN > everything except null" rule that cuDF's `NULL_MIN/NULL_MAX` doesn't respect.

**Opportunity**: Two independent transforms:

1. **Compute a single "anyNaN" mask upfront**: `anyNan = orReduce(isNan(col_i) for all i)` — this is an N-wide OR (can be streamed), ~N kernels total.
2. **Do a single N-ary reduction** with `NULL_MIN`/`NULL_MAX` (if cuDF exposes row-wise min over a Table, or emulate via pairwise `NULL_MIN` which drops the extra NaN-handling → 2 kernels per pair).
3. **Post-process once**: For `shouldNanWin`, overwrite the final result with `Float.NaN` where `anyNan`; for `shouldNanLose`, the `NULL_MIN` already does the right thing except when *all* inputs are NaN (one extra `ifElse`).

This brings the kernel count from ~4N to ~2N + 3, a ~2× reduction in launch overhead. For N=10+ columns (common in `greatest(v1,v2,...v10)` expressions) the savings are significant.

**NDS relevance**: Moderate — `greatest`/`least` appear in date-window calculations in NDS queries.

---

## Tier B: Decimal pipeline — every arithmetic result pays a fixed tax

### B1. `GpuCheckOverflow` — runs after every decimal arithmetic; 4+ kernels
`sql-plugin/src/main/scala/com/nvidia/spark/rapids/decimalExpressions.scala` (lines 34–65)

**What it does today**: Every `GpuAdd`/`GpuSubtract`/`GpuMultiply`/`GpuDivide` on `DecimalType` is wrapped with `GpuCheckOverflow`, which chains:
```
round(toScale), castTo(targetDType), DecimalUtil.outOfBounds(greaterThan(max), lessThan(min), or),
[ifElse(nullify)] OR [throw on any(true)]
```
That's **4 kernels after every decimal op** (5 in non-nullOnOverflow mode because of the `any().getBoolean` host sync).

**Opportunity**: Fold bounds-checking into the arithmetic itself via a JNI kernel variant that performs `add_decimal_with_overflow_check` in a single pass, writing either NULL or throwing via a device-side flag buffer (read once per batch). This halves the kernel count for every decimal expression in NDS queries, which are decimal-heavy in `ss_ext_sales_price`, `ss_list_price`, `ss_sales_price` calculations.

**Related**: `GpuMakeDecimal.doColumnar` (lines 128–167) does its own 4-kernel bounds check (`greaterThan`, `lessThan`, `or`, `ifElse`/`copyToColumnVector`). Same optimization template — single JNI kernel.

**NDS relevance**: **Very high**. Every TPC-DS revenue computation produces decimals and hits `GpuCheckOverflow`.

---

### B2. `GpuMakeDecimal` — 4 kernels for an unchecked integer→decimal bit cast
`sql-plugin/src/main/scala/com/nvidia/spark/rapids/decimalExpressions.scala` (lines 128–167)

**What it does today**: Converts an integer column to a decimal with a runtime bounds check:
```
greaterThan(hi), lessThan(lo), or, ifElse(null, bitCastToDecimal)
```
When `nullOnOverflow = true`, 4 kernels; plus intermediate columns. When the caller has already established that the values fit (e.g., a freshly-computed `unscaledLong`), all 4 are waste.

**Opportunity**: Add a fast-path `GpuMakeDecimal.unsafe(intCol, scale)` that only does the view reinterpret (`bitCastTo(DType.decimal(...))`, zero-copy) and use it from call sites that have already checked bounds. For the checked path, fuse check+cast into a single JNI kernel.

---

## Tier C: Complex-type operators — redundant passes and copies

### C1. `GpuArrayContains.orNotContainsNull` — 4 kernels of null-policy patch-up
`sql-plugin/src/main/scala/org/apache/spark/sql/rapids/complexTypeExtractors.scala` (lines 293–305, invoked from every overload)

**What it does today**: cuDF's `listContains` returns `false` (not `null`) when a list row has some nulls and the search key isn't found. Spark wants `null`. The fix currently runs:
```
listContainsNulls, not, or(contains, notContainsNulls), ifElse(result, nullScalar)
```
= 4 kernels after every `array_contains`.

**Opportunity**: Use `mergeAndSetValidity(BITWISE_AND, containsResult, orColumn)` or request a cuDF API `listContainsWithNullPropagate` that handles Spark semantics natively. Alternatively, express as:
```
validityMask = containsResult OR NOT(listContainsNulls)
output = containsResult.mergeAndSetValidity(validityMask)
```
= 3 kernels (instead of 4), and `mergeAndSetValidity` is a validity-buffer-only update, cheaper than `ifElse`.

---

### C2. `GpuMapFromEntries` (EXCEPTION policy) — full value dedup just to check keys
`sql-plugin/src/main/scala/org/apache/spark/sql/rapids/collectionOperations.scala` (lines 787–799)

**What it does today**: With the default Spark policy, the code calls `dropListDuplicatesWithKeysValues()` **solely to compare row counts** and detect duplicates — the deduped output is discarded. `dropListDuplicatesWithKeysValues` is strictly more expensive than `dropListDuplicates` because it also gathers the value side.

**Opportunity**: For the check-only path, call `dropListDuplicates()` (or expose a new cuDF API `listContainsDuplicates()` that returns a single boolean) and only invoke the full dedup on the `LAST_WIN` branch. This removes a full list-gather worth of data movement.

**Parallel**: This is the same class of optimization as Item #6 in the existing report (`GpuMapFromArrays.nullSanitize`), but for a different operator.

---

### C3. `GpuArrayExists.threeValueExists` — double reduction on the same data
`sql-plugin/src/main/scala/com/nvidia/spark/rapids/higherOrderFunctions.scala` (lines 392–398)

**What it does today**: Implements Spark's three-valued `EXISTS` semantics by running `existsReduce` twice on the same child column — once with `NullPolicy.EXCLUDE`, once with `NullPolicy.INCLUDE` — then combines the results.

**Opportunity**: A single pass over the child can compute both `hasTrue` and `hasAnyNull` per row (OR-reduce of `isTrue`, OR-reduce of `isNull`) and combine with an `ifElse` or `mergeAndSetValidity`. cuDF's segmented reductions can emit multiple aggregates per row in a single kernel — if not, a JNI-level fused kernel would suffice. Saves one full pass over the exploded child data.

---

### C4. `GpuGetArrayStructFields` — `copyToColumnVector` breaks the zero-copy chain
`sql-plugin/src/main/scala/org/apache/spark/sql/rapids/complexTypeExtractors.scala` (lines 435–446)

**What it does today**:
```scala
val listView = GpuListUtils.replaceListDataColumnAsView(base, fieldView) // zero-copy view
listView.copyToColumnVector()                                             // FULL COPY
```
The final `copyToColumnVector()` materializes the entire list (offsets + validity + extracted field) because Spark's result contract needs an owned `ColumnVector`.

**Opportunity**: Return a `ColumnView`-backed wrapper for expressions that are consumed by another GPU expression without crossing a batch boundary (e.g., a `GpuGetArrayStructFields` feeding into a `GpuExplode` within the same batch). Broader fix: plumb views (not owned vectors) through the expression tree's internal calls. This is a well-known speedup for struct-field extraction, which appears in **every JSON or Parquet struct column access**.

---

### C5. `GpuSequence.checkSequenceInputs` — 3 filter+comparison passes for one validity check
`sql-plugin/src/main/scala/org/apache/spark/sql/rapids/collectionOperations.scala` (lines 1838–1891)

**What it does today**: For each row, the function checks three separate rules (`step>0 && start≤stop`, `step<0 && start≥stop`, `step==0 && start==stop`). Each rule is implemented by:
```
filter(start,stop) by predicate  →  compare filtered columns  →  isAllValidTrue
```
That's **9 kernels + 3 table-filter allocations** for a per-row boolean predicate.

**Opportunity**: A single all-in-one predicate column:
```
valid = ((step>0) & (start<=stop)) | ((step<0) & (start>=stop)) | ((step==0) & (start==stop))
```
is ~7 kernels and **zero filter allocations**. Then a single `isAllValidTrue(valid)`. On top of that, the error message needs the offending row — which we can get from `argmin(valid).getInt` only in the failure branch.

---

## Tier D: String operators — chain length is the killer

### D1. `GpuFormatNumber.addCommas` — loops `substring` in chunks of 3 digits
`sql-plugin/src/main/scala/org/apache/spark/sql/rapids/stringFunctions.scala` (lines 2386–2410)

**What it does today**: To insert thousands-separators, the code walks the integer part in steps of 3 characters, calling `substring` on the entire column for each chunk then concatenating. A 19-digit Long needs **7 substring calls + 7 concats**, each of which is a full-column pass.

**Opportunity**: A single JNI CUDA kernel that inserts commas based on digit count, similar to `CastStrings.toDecimal`. One pass over the input strings → one output string column. This collapses 14+ kernels into 1.

**NDS relevance**: `format_number` is used in reporting queries and in any `format_string` / `DECIMAL_FORMAT` flows.

---

### D2. `GpuConv` ANSI-mode paths — double-scan (check, then convert)
`sql-plugin/src/main/scala/org/apache/spark/sql/rapids/stringFunctions.scala` (lines 2142–2264)

**What it does today**: In ANSI mode, every overload (`(col, scalar, scalar)`, `(col, scalar, col)`, `(col, col, scalar)`, `(col, col, col)`) makes **two JNI calls**: `isConvertOverflow(...)` to detect overflow, then `convert(...)` to produce the result. Both kernels re-parse each input string.

**Opportunity**: Extend the JNI `convert` kernel to emit both the result column and a sidecar boolean (or single-bit scalar) indicating whether any row overflowed. One pass instead of two — roughly **2× speedup** in ANSI mode with zero behavioral change. Analogous to `GpuCast` which combines `isFixedPoint + parse` into `CastStrings`.

---

### D3. `GpuStringLPad`/`RPad` — pad, then substring to truncate
`sql-plugin/src/main/scala/org/apache/spark/sql/rapids/stringFunctions.scala` (lines 1761–1775)

**What it does today**:
```scala
str.pad(l, direction, padStr) // pads strings shorter than l
  .substring(0, l)             // truncates strings longer than l
```
Two kernels + full intermediate column every call.

**Opportunity**: Fused `padOrTruncate(l, direction, padStr)` JNI kernel — one pass, one allocation. Common in ETL pipelines that normalize key widths.

---

## Tier E: Conditional expressions — fusion-eligible paths underused

### E1. `GpuCaseWhen` non-fusion path — O(N) intermediate columns
`sql-plugin/src/main/scala/com/nvidia/spark/rapids/conditionalExpressions.scala` (lines 412–424)

**What it does today**: The generic fallback recursively builds N−1 `ifElse` intermediate columns when `useFusion` is false. Fusion is only enabled when:
- `branches.size > 2`
- `isCaseWhenFusionSupportedType(type)` — excludes `ArrayType`, `StructType`, `MapType`, `BinaryType`, `TimestampType`, `DateType`
- **All THEN/ELSE are `GpuLiteral`** (very restrictive)

In practice, most NDS `CASE WHEN` expressions have *column* THEN branches (e.g., `CASE WHEN x > 0 THEN y ELSE z`) and fall into the slow recursive path.

**Opportunity** (already sketched as Item #18 in the existing report but worth concrete restatement): Generalize the fusion path to:
1. Compute all WHEN booleans once.
2. Call `CaseWhen.selectFirstTrueIndex(whenBoolCols)` to get a per-row branch index.
3. Evaluate each THEN expression (still needed for non-literal branches).
4. `Table(thenCol1, thenCol2, …, elseCol).gather(branchIndex)` → single output, single allocation.

For N branches of column THEN, this drops from N−1 intermediate `ifElse` result columns to a single gather — kernel count drops from O(N) to O(1) after the THEN evaluations. Supports all data types that cuDF can gather (including ArrayType/StructType — trivially extending the supported-type set).

**NDS relevance**: **Very high**. `CASE WHEN` appears **127 times** in `sf1k_query_0.sql` alone.

---

### E2. `GpuIf` eager branch evaluation without side-effects
`sql-plugin/src/main/scala/com/nvidia/spark/rapids/conditionalExpressions.scala` (lines 213–251)

**What it does today**: When neither branch has side effects, the code eagerly evaluates *both* the TRUE and FALSE expressions on the full batch, then merges with `ifElse`. If one branch's computation is expensive (e.g., contains a string regex, a decimal divide, a cast-to-string), we pay full cost for rows that never use it.

**Opportunity**: Use `conditionalWithSideEffects`-style filter/gather *also* in the side-effect-free path **when the branch cost exceeds a threshold** (an approximation based on expression complexity metadata). The existing filter+gather code is already written and tested — it just needs a cost heuristic to route more expressions through it.

---

## Tier F: Datetime — wasteful type casts

### F1. `GpuDateDiff` — `asInts()` full copy instead of zero-copy bit-cast
`sql-plugin/src/main/scala/org/apache/spark/sql/rapids/datetimeExpressions.scala` (lines 274–320)

**What it does today**:
```scala
dateColumn.asInts()   // TIMESTAMP_DAYS → INT32, ALLOCATES A NEW COLUMN
```
even though the underlying storage of `TIMESTAMP_DAYS` **is already** `INT32` (a day offset since epoch).

**Opportunity**: Replace with `bitCastTo(DType.INT32)` which returns a zero-copy view. Saves one full allocation per `datediff` call and per `GpuDateSub`/`GpuDateAdd` that uses the same idiom. Look for all `asInts()` / `asTimestampSeconds()` chains on timestamp columns — many are already-bit-equivalent representations.

**NDS relevance**: Moderate (datediff appears in date-range queries).

---

## Tier G: Regex / parsing

### G1. `GpuSubstringIndex` — no fast-path for empty/null inputs
`sql-plugin/src/main/scala/org/apache/spark/sql/rapids/stringFunctions.scala` (lines 1727–1742)

**What it does today**: Always calls `GpuSubstringIndexUtils.substringIndex` which internally does regex-like splitting. For `count == 0` (empty-string result for all rows), `count == Int.MaxValue` (identity), or `delim` not present in the column, we could short-circuit.

**Opportunity**: Pre-check `count == 0` → return empty-string scalar-expanded column; `delim` literal contains zero occurrences in the whole column (cheap `stringContains().any()`) → return the input unchanged (incRefCount). These are single-scalar tests that avoid a full JNI call.

Impact per-call is modest, but `substring_index` is used in schema parsers and path extractors, so the saved kernels add up.

---

## Summary table

| Operator | Current kernels | Proposed kernels | NDS impact |
|---|---|---|---|
| `GpuHypot` | ~8–10 | 1 (JNI) | Low |
| `GpuAcoshCompat`/`GpuAsinhCompat` | 5 | 1 (JNI) | Low |
| `GpuGreatest/Least` FP, N cols | ~4N | ~2N + 3 | Moderate |
| `GpuCheckOverflow` (decimal) | 4–5 per op | 1 (JNI fused) | **Very high** |
| `GpuMakeDecimal` | 4 | 1 (JNI fused) or 0 (unsafe path) | Moderate |
| `GpuArrayContains` | 5 (incl. listContains) | 3 | Moderate |
| `GpuMapFromEntries` (EXCEPTION) | full value dedup | key-only dedup | Moderate |
| `GpuArrayExists.threeValueExists` | 2 reductions | 1 fused reduction | Moderate |
| `GpuGetArrayStructFields` | zero-copy + full copy | zero-copy only | **High** (all struct/array access) |
| `GpuSequence.checkSequenceInputs` | 9 + 3 filter allocs | ~7, no filters | Low |
| `GpuFormatNumber.addCommas` | 14+ | 1 (JNI) | Moderate |
| `GpuConv` ANSI | 2 JNI calls | 1 JNI call | Moderate |
| `GpuStringLPad/RPad` | 2 | 1 (JNI fused) | Moderate |
| `GpuCaseWhen` non-fusion | N−1 `ifElse` | 1 gather | **Very high** |
| `GpuIf` eager eval | full-batch both sides | filter/gather above threshold | High |
| `GpuDateDiff` `asInts` | 1 allocation | 0 (bitCast view) | Moderate |
| `GpuSubstringIndex` edge cases | always JNI | short-circuit | Low |

---

## Recommended top-3 to pursue first

Given the NDS query shape and effort vs. payoff:

1. **`GpuCheckOverflow` fused decimal arithmetic** (Tier B1) — pays off on *every* decimal expression in TPC-DS; one JNI kernel template benefits `+`, `−`, `×`, `÷`.
2. **`GpuCaseWhen` generalized gather-based fusion** (Tier E1) — 127× per NDS query × many queries × significant kernel savings per case; involves only Scala changes (uses existing `CaseWhen.selectFirstTrueIndex`).
3. **`GpuGetArrayStructFields` zero-copy** (Tier C4) — every struct-field access today re-materializes the entire list; saves an allocation × row count on each access, and is particularly heavy for Parquet-backed nested schemas.
