# Spark RAPIDS: Synthesized Operator Optimization Report

This report merges and deduplicates optimization candidates from four independent
analyses of the Spark RAPIDS plugin codebase, grounded in NDS benchmark profiling
(102 queries, 1-GPU) and targeted code review. The focus is on **specific, isolated
operator-level optimizations** that are workload-independent and broadly applicable.

**Source analyses:**

- Claude 4.6 Opus (detailed operator-level code review)
- Codex GPT-5.4 (profile-driven prioritization)
- Cursor GPT-5.3/Codex (profile-driven prioritization)
- Cursor GPT-5.4 (operator and kernel review)

**Profiling context (summed across 102 NDS queries):**


| Category                 | NVTX Time (s) | % of Total |
| ------------------------ | ------------- | ---------- |
| I/O (Parquet read/write) | 8741          | 45.8%      |
| Shuffle                  | 3444          | 18.0%      |
| Coalesce/Concat          | 1945          | 10.2%      |
| Joins                    | 1834          | 9.6%       |
| GPU scheduling           | 1297          | 6.8%       |
| Aggregation              | 198           | 1.0%       |
| Project                  | 164           | 0.9%       |


**Ordering principle:** Items are ordered by ROI (importance / effort). Specific,
isolated operator fixes with clear implementations rank highest. Broader heuristic
or infrastructure changes are included at the end for completeness but are
deprioritized per the stated preference for targeted, workload-independent changes.

---

## Tier 1 — Quick Wins: High ROI, Very Low Effort

These are drop-in API substitutions or small control-flow changes within a single
method. Each has a well-defined fix using existing cuDF Java APIs and minimal risk
of behavioral regression.

---

### 1. GpuNvl (NVL / COALESCE): Replace `isNotNull` + `ifElse` with `replaceNulls`

**Identified by:** Claude 4.6 Opus

**Importance:** High — COALESCE is one of the most pervasive expressions in SQL
(~344 occurrences in NDS query streams). It is emitted by Spark for null handling
in joins, aggregations, and projections. `GpuCoalesce` chains `GpuNvl` N-1 times
for a `COALESCE(a, b, ..., n)` with N children, making this an extreme hot-path
multiplier.

**Effort:** Very Low — 2-line change per overload.

#### Current Inefficiency

`nullExpressions.scala:30-41` — both overloads of `GpuNvl.apply` create an
intermediate boolean column via `isNotNull`, then select between the two inputs
using `ifElse`:

```scala
def apply(lhs: ColumnVector, rhs: ColumnVector): ColumnVector = {
  withResource(lhs.isNotNull) { isLhsNotNull =>  // GPU kernel #1: allocates N booleans
    isLhsNotNull.ifElse(lhs, rhs)                // GPU kernel #2: copies N values
  }
}
```

This is 2 GPU kernel launches and 1 intermediate boolean allocation per call.
For `COALESCE(a, b, c, d)`, that totals 6 kernels and 3 boolean columns.

cuDF provides `replaceNulls(ColumnView)` and `replaceNulls(Scalar)` at
`ColumnView.java:536-554`, which perform the identical operation in a single
kernel with zero intermediate allocations.

#### Potential Optimization

```scala
object GpuNvl {
  def apply(lhs: ColumnVector, rhs: ColumnVector): ColumnVector =
    lhs.replaceNulls(rhs)

  def apply(lhs: ColumnVector, rhs: Scalar): ColumnVector =
    lhs.replaceNulls(rhs)
}
```

#### High-Level Change & Risks

**Change:** Replace both overloads with direct `replaceNulls` calls.

**Risks:**

- **Null semantics on nested types.** `replaceNulls` on LIST/STRUCT/MAP columns
replaces the top-level null with the corresponding row from the replacement
column. This matches `isNotNull + ifElse` semantics, but if cuDF's
`replaceNulls` and `ifElse` diverge on nested-type validity propagation,
edge cases would surface. Targeted unit tests for `COALESCE` on nested types
should be verified before merging.
- **No functional risk for scalar overload.** `replaceNulls(Scalar)` directly
replaces all null entries with the scalar value, matching `isNotNull.ifElse`
exactly.

---

### 2. NullUtilities.mergeNulls: Replace Multi-Kernel Pipeline with `mergeAndSetValidity`

**Identified by:** Claude 4.6 Opus

**Importance:** Medium-High — `mergeNulls` is called from arithmetic expressions,
collection operations, and other expression evaluators whenever null propagation
across multiple columns is needed.

**Effort:** Very Low — single-method replacement for each overload.

#### Current Inefficiency

`nullExpressions.scala:322-365` — the single-flag overload creates 1 intermediate
boolean column (`isNull`), checks `any()`, then does `ifElse(nullScalar, dataCol)`.
The two-flag overload creates 3 intermediate boolean columns (`isNull1`, `isNull2`,
`nullFlag = or`), checks `any()`, then does `ifElse`. The two-flag path is up to
5 GPU kernels and 3 intermediate boolean allocations. The `ifElse` with a null
scalar fundamentally just "sets validity bits to 0" — but it copies the entire
column data to update the bitmask.

cuDF's `mergeAndSetValidity(BinaryOp.BITWISE_AND, col1, col2)` at
`ColumnView.java:891` merges validity bitmasks directly in a single operation.
No boolean columns, no data copy.

#### Potential Optimization

```scala
// Single-flag overload
def mergeNulls(dataCol: ColumnVector, nullFlagCol: ColumnView): ColumnVector =
  dataCol.mergeAndSetValidity(BinaryOp.BITWISE_AND, nullFlagCol)

// Two-flag overload
def mergeNulls(dataCol: ColumnVector, nullFlagCol1: ColumnView,
               nullFlagCol2: ColumnView): ColumnVector =
  dataCol.mergeAndSetValidity(BinaryOp.BITWISE_AND, nullFlagCol1, nullFlagCol2)
```

#### High-Level Change & Risks

**Change:** Replace both overloads. The existing `needsToChange` fast-path
(`any()` reduction) is no longer needed since `mergeAndSetValidity` on all-valid
inputs is effectively a no-op internally.

**Risks:**

- **Validity semantics.** `mergeAndSetValidity(BITWISE_AND, ...)` ANDs the validity
bitmasks of all provided columns with the data column's own validity. This means
"null if any input is null", matching the current `isNull.or + ifElse` semantics.
- **Data type compatibility.** `mergeAndSetValidity` works on all column types
including nested types because it only touches the validity buffer.

---

### 3. GpuRegExpExtractAll: Use `extractListElement(int)` Instead of Materializing Index Column

**Identified by:** Claude 4.6 Opus

**Importance:** Medium — regex extraction appears in NDS queries on string columns.

**Effort:** Very Low — 3-line change.

#### Current Inefficiency

`stringFunctions.scala:1657-1663` — for each index `i`, the code creates a
`Scalar.fromInt(i)`, expands it to a full-size `ColumnVector` via
`fromScalar(scalarIndex, rowCount)`, then calls the column-index overload of
`extractListElement(ColumnView)`:

```scala
val stringCols = Range(0, maxSizeInt, 1).safeMap { i =>
  withResource(Scalar.fromInt(i)) { scalarIndex =>
    withResource(ColumnVector.fromScalar(scalarIndex, rowCount.toInt)) {
      index => allExtracted.extractListElement(index)
    }
  }
}
```

This creates 2 GPU allocations per loop iteration to pass a constant integer index.
cuDF provides `extractListElement(int)` at `ColumnView.java:2813` which accepts a
plain integer directly.

#### Potential Optimization

```scala
val stringCols = Range(0, maxSizeInt, 1).safeMap { i =>
  allExtracted.extractListElement(i)
}
```

#### High-Level Change & Risks

**Change:** Replace the scalar-expansion loop with the `int` overload.

**Risks:**

- **Negative index semantics.** The range starts at 0 and goes upward, so indices
are always non-negative. `extractListElement(int)` with non-negative indices
behaves identically to the column overload when every row contains the same
index value.
- **No risk of off-by-one.** The original code used `Scalar.fromInt(i)` which is
exactly what the `int` overload receives.

---

### 4. GpuDivModLike.replaceZeroWithNull: Replace `findAndReplaceAll` with `equalTo` + `ifElse`

**Identified by:** Claude 4.6 Opus

**Importance:** Medium — division and modulo operations appear in date arithmetic,
bucket calculations, and aggregation expressions across NDS queries.

**Effort:** Low — localized rewrite of one method.

#### Current Inefficiency

`arithmetic.scala:640-667` — creates 2 single-element `ColumnVector`s (a zero and
a null) via `fromScalar(..., 1)`, then calls `findAndReplaceAll` — a heavyweight
API designed for multi-value lookup tables:

```scala
def replaceZeroWithNull(v: ColumnVector): ColumnVector = {
  // ... imperative try/finally with 4 nullable vars ...
  zeroVec = ColumnVector.fromScalar(zeroScalar, 1)   // 1-element column
  nullVec = ColumnVector.fromScalar(nullScalar, 1)   // 1-element column
  v.findAndReplaceAll(zeroVec, nullVec)
}
```

This allocates 2 unnecessary GPU columns and uses a heavyweight API for a
single-value replacement.

#### Potential Optimization

```scala
def replaceZeroWithNull(v: ColumnVector): ColumnVector = {
  withResource(makeZeroScalar(v.getType)) { zero =>
    withResource(v.equalTo(zero)) { isZero =>
      withResource(Scalar.fromNull(v.getType)) { nullVal =>
        isZero.ifElse(nullVal, v)
      }
    }
  }
}
```

#### High-Level Change & Risks

**Change:** Replace `findAndReplaceAll` with `equalTo` + `ifElse`. Also
modernizes resource management from imperative try/finally to idiomatic
`withResource`.

**Risks:**

- **Floating-point NaN.** Both `findAndReplaceAll` and `equalTo` treat NaN as
not-equal-to anything (IEEE 754), so NaN divisors are not replaced — matching
current behavior.
- **Null input handling.** `equalTo(zero)` produces null for null rows; `ifElse`
maps null predicates to the false branch. `findAndReplaceAll` also does not
match nulls against the search column. Behavior is identical.

---

### 5. GpuSize with legacySizeOfNull: Replace `isNull` + `ifElse` with `replaceNulls`

**Identified by:** Claude 4.6 Opus

**Importance:** Low-Medium — `size()` is used in queries that inspect array/map
lengths.

**Effort:** Very Low — 3-line change.

#### Current Inefficiency

`collectionOperations.scala:648-663` — when `legacySizeOfNull` is true, creates an
`isNull` boolean column, then uses `ifElse` to replace nulls with -1:

```scala
withResource(input.getBase.countElements()) { collectionSize =>
  if (legacySizeOfNull) {
    withResource(Scalar.fromInt(-1)) { nullScalar =>
      withResource(input.getBase.isNull) { inputIsNull =>
        inputIsNull.ifElse(nullScalar, collectionSize)
      }
    }
  } ...
}
```

`countElements()` already returns null for null list/map inputs. The entire
operation is just "replace null sizes with -1" — exactly what `replaceNulls(Scalar)`
does.

#### Potential Optimization

```scala
withResource(input.getBase.countElements()) { collectionSize =>
  if (legacySizeOfNull) {
    withResource(Scalar.fromInt(-1)) { minusOne =>
      collectionSize.replaceNulls(minusOne)
    }
  } else {
    collectionSize.incRefCount()
  }
}
```

#### High-Level Change & Risks

**Change:** Replace `isNull` + `ifElse` with `replaceNulls(Scalar)`.

**Risks:** None. `replaceNulls(Scalar)` on an INT32 column is a direct,
well-tested operation with identical semantics to `isNull.ifElse(scalar, col)`.

---

### 6. GpuMapFromArrays: Eliminate Double Duplicate-Detection Kernel

**Identified by:** Claude 4.6 Opus

**Importance:** Medium — `map_from_arrays` is used in queries that construct maps
from parallel key/value arrays. The `LAST_WIN` dedup policy is the default in
Spark 3.4+.

**Effort:** Very Low — control flow change within one method.

#### Current Inefficiency

`collectionOperations.scala:1587-1654` — when `mapKeyDedupPolicy` is `"LAST_WIN"`,
the code first calls `rowContainsDuplicates(sanitizedLhsBase)` (which internally
runs `dropListDuplicates` to check), and if duplicates exist, calls
`mapCol.dropListDuplicatesWithKeysValues` — running the deduplication kernel a
second time:

```scala
val result = withResource(mapCol) { mapCol =>
  mapKeyDedupPolicy.toString match {
    case "LAST_WIN" if rowContainsDuplicates(sanitizedLhsBase) =>  // dedup kernel #1
        mapCol.dropListDuplicatesWithKeysValues                     // dedup kernel #2
    case _ =>
      mapCol.incRefCount()
  }
}
```

The `EXCEPTION` path has the same issue — `rowContainsDuplicates` detects
duplicates, then throws instead of using the dedup result.

#### Potential Optimization

For `LAST_WIN`, always call `dropListDuplicatesWithKeysValues` unconditionally.
Deduplication on already-unique data is marginally more expensive than
`incRefCount` but far cheaper than running the kernel twice when duplicates exist.

For `EXCEPTION`, use the dedup result to detect duplicates by comparing child
row counts:

```scala
case "EXCEPTION" =>
  withResource(sanitizedLhsBase.dropListDuplicates) { deduped =>
    val origRows = sanitizedLhsBase.getChildColumnView(0).getRowCount
    val dedupRows = deduped.getChildColumnView(0).getRowCount
    require(origRows == dedupRows, "[DUPLICATED_MAP_KEY] Duplicate map key was found")
  }
```

#### High-Level Change & Risks

**Change:** Remove the `rowContainsDuplicates` guard call and always dedup
directly.

**Risks:**

- **Performance regression in no-duplicate case.** For `LAST_WIN` when no
duplicates exist, the proposed code does `dropListDuplicatesWithKeysValues`
(full dedup) instead of `dropListDuplicates` (check) + `incRefCount`. The
overhead of a single dedup kernel scan that finds nothing is modest and still
cheaper than 2 kernels when duplicates exist.

---

### 7. GpuMonthsBetween: Deduplicate `.day()` Extractions

**Identified by:** Claude 4.6 Opus

**Importance:** Low-Medium — `months_between` appears in date-heavy NDS queries.

**Effort:** Very Low — parameter threading through helper methods.

#### Current Inefficiency

`datetimeExpressions.scala:1355-1407` — `calcJustMonth` calls `converted1.day()`
and `converted2.day()` at lines 1357-1358. Then `calcSecondsDiff` calls them again
at lines 1388-1389. Each `.day()` is a separate GPU kernel. Total: 4
day-extraction kernels when 2 suffice.

#### Potential Optimization

Extract `day1` and `day2` once in the calling `columnarEval` method and pass as
parameters to both helpers:

```scala
withResource(converted1.day()) { day1 =>
  withResource(converted2.day()) { day2 =>
    val justMonth = calcJustMonth(converted1, converted2, day1, day2)
    val secondsDiff = calcSecondsDiff(converted1, converted2, day1, day2)
    ...
  }
}
```

#### High-Level Change & Risks

**Change:** Add `day1`/`day2` parameters to `calcJustMonth` and `calcSecondsDiff`.

**Risks:**

- **Lifetime management.** The day columns must remain alive through both helpers.
They are managed by `withResource` in the calling scope, making this
straightforward.
- **No functional risk.** `.day()` is deterministic — calling it once vs. twice
produces identical results.

---

### 8. GpuDayOfWeek: Eliminate Type Cast Chain

**Identified by:** Claude 4.6 Opus

**Importance:** Low-Medium — `DAYOFWEEK` / `dow` appears in date-partitioned NDS
queries.

**Effort:** Low — arithmetic rearrangement.

#### Current Inefficiency

`datetimeExpressions.scala:71-87` — to shift from cuDF's Monday=1 convention to
Spark's Sunday=1, the code casts date → INT32, adds 1, casts back to
TIMESTAMP_DAYS, then calls `weekDay()`. That is 4 GPU kernels with 2 type casts.

Compare with `GpuWeekDay` (lines 60-68) which simply does `weekDay()` → `sub(one)`
— 2 kernels, no casts.

#### Potential Optimization

cuDF `weekDay()` returns Monday=1..Sunday=7. Spark `DayOfWeek` wants
Sunday=1..Saturday=7. The mapping is `(weekDay % 7) + 1`:

```scala
override protected def doColumnar(input: GpuColumnVector): ColumnVector = {
  withResource(input.getBase.weekDay()) { weekday =>           // kernel #1
    withResource(Scalar.fromShort(7.toShort)) { seven =>
      withResource(weekday.mod(seven)) { modded =>             // kernel #2
        withResource(Scalar.fromShort(1.toShort)) { one =>
          modded.add(one)                                       // kernel #3
        }
      }
    }
  }
}
```

3 kernels (no casts) vs. current 4 kernels (with 2 casts).

#### High-Level Change & Risks

**Change:** Replace cast-based approach with arithmetic modular mapping.

**Risks:**

- **Correctness.** The formula `(weekDay % 7) + 1` maps all 7 days correctly
(verified: Monday→2, Tuesday→3, ..., Saturday→7, Sunday→1).
- **Null handling.** `weekDay()` returns null for null inputs. `mod` and `add`
propagate nulls. Same as current behavior.

---

## Tier 2 — High-Value Operator Optimizations: Low to Medium Effort

These offer meaningful performance improvements with moderate implementation
complexity, still focused on specific operator paths.

---

### 9. Hash Partitioning: Use Fused `Table.hashPartition()` Instead of 3-Step Pipeline

**Identified by:** Claude 4.6 Opus, Codex GPT-5.4, Cursor GPT-5.3/Codex

**Importance:** High — Shuffle accounts for 18.0% of total NVTX time across NDS.
`Hash partition` alone is ~143.7s. Combined partition-related spend is ~472.5s
across all NDS queries. Every shuffle write passes through `hashPartitionAndClose`.

**Effort:** Low to Medium — API substitution, but requires verifying hash-seed
compatibility and preserving output contracts. (One model rated this Low effort,
two rated Medium.)

#### Current Inefficiency

`GpuHashPartitioningBase.scala:36-52` — the current code performs 3 separate GPU
operations to partition a batch:

1. `hashFunc.columnarEval(cb)` — compute hash column (full INT32 allocation)
2. `hash.pmod(partsLit, DType.INT32)` — compute partition IDs (another full INT32
  allocation)
3. `table.partition(parts, numPartitions)` — scatter to partitions

For a 10M-row batch, steps 1-2 produce ~80MB of temporary GPU memory, plus 3
separate kernel launches. The map-based partition path in cuDF also uses an
atomic-heavy scatter-map build internally.

cuDF provides `Table.onColumns(...).hashPartition(HashType.MURMUR3, numPartitions)`
at `Table.java:4572-4606` which fuses all three steps into a single native call.
The intermediate hash and partition-ID columns never exist as separate GPU
allocations.

#### Potential Optimization

Replace the 3-step pipeline with a single call:

```scala
withResource(batch) { cb =>
  withResource(GpuColumnVector.from(cb)) { table =>
    nvtxId {
      table.onColumns(keyColumnIndices: _*)
        .hashPartition(HashType.MURMUR3, numPartitions)
    }
  }
}
```

This requires mapping the `hashFunc` expression's bound key column references to
integer column indices within the table. Non-Murmur hash modes (e.g., `GpuHiveHash`)
would remain on the current fallback path.

#### High-Level Change & Risks

**Change:** In `GpuHashPartitioningBase` and join sub-partitioning flows, branch
on hash mode: Murmur3 → direct `hashPartition`, other modes → current fallback.
Preserve the `PartitionedTable` offsets contract and downstream slicing
expectations.

**Risks:**

- **Hash seed compatibility.** Spark's `Murmur3Hash` uses seed 42 by default.
cuDF's `Table.hashPartition(MURMUR3, n)` uses seed 0. There is an overload
`hashPartition(HashType, int, int)` with a seed parameter. The seed **must**
match whatever `GpuMurmur3Hash` currently produces — a mismatch would route
rows to wrong partitions mid-query.
- **Float normalization.** `GpuMurmur3Hash` normalizes -0.0 to +0.0 before hashing
(Spark requirement). It is unclear whether cuDF's `hashPartition(MURMUR3, ...)`
performs the same normalization. Must be verified in cuDF source or tested
explicitly.
- **Expression-based keys.** If hash key expressions involve computation beyond
simple column references (e.g., `hash(col1 + col2)`), the column-ordinal
approach won't work directly. Expressions would need to be evaluated first and
appended as temporary columns.
- **Hive hash support.** `Table.hashPartition` may not support Hive hashing. The
Hive hash path would need to remain on the current 3-step pipeline.
- **Shuffle compatibility.** Behavior under extreme skew or tiny-partition
workloads can regress if partition-offset semantics change. Must verify across
all supported Spark shim versions.

---

### 10. GpuArraysZip.normalizeNulls / GpuMapFromArrays.nullSanitize: Use `mergeAndSetValidity` Instead of Deep Copies

**Identified by:** Claude 4.6 Opus

**Importance:** Medium — these affect `arrays_zip` and `map_from_arrays` operations
on columns with nested (LIST/STRUCT) types.

**Effort:** Low — localized change in two methods.

#### Current Inefficiency

`**collectionOperations.scala:1179-1213` (`normalizeNulls`):** For each input
column, applies `nullOutput.ifElse(nullScalar, cv)` where `nullOutput` is a
combined boolean null mask. `ifElse` on a LIST column performs a deep copy of the
entire column (children, offsets, validity buffers) just to update the top-level
validity mask. For `arrays_zip(a, b, c)`, this is 3 full LIST-column deep copies.

`**collectionOperations.scala:1555-1581` (`nullSanitize`):** Same pattern — two
`ifElse(nullScalar, col)` calls for key and value columns.

#### Potential Optimization

Replace with `mergeAndSetValidity`, which only manipulates the validity bitmask:

```scala
// normalizeNulls — invert once (true="null" → false="invalid"), reuse N times
withResource(nullOutput.not()) { combinedValid =>
  inputs.safeMap { cv =>
    cv.mergeAndSetValidity(BinaryOp.BITWISE_AND, combinedValid, cv)
  }
}
```

#### High-Level Change & Risks

**Change:** Replace `ifElse(null, col)` patterns with `mergeAndSetValidity` in
both methods.

**Risks:**

- **Child data sharing.** `mergeAndSetValidity` creates a new column that shares
child data/offsets buffers with the original column (only validity buffer is
new). The original must remain alive while the new column is in use. Current
`withResource` scopes outlive the new columns, so this should be safe. However,
premature closure of the original could invalidate child data.
- **Boolean inversion.** The `combinedNulls` mask has `true` = "should be null",
but `mergeAndSetValidity(BITWISE_AND)` treats `1` bits as "valid". The `not()`
inversion adds one kernel, but it is shared across all inputs (computed once)
and is still far cheaper than N deep copies.

---

### 11. GpuCoalesce: Short-Circuit on Non-Nullable Children

**Identified by:** Claude 4.6 Opus

**Importance:** Medium — COALESCE is pervasive, and many real queries use patterns
like `COALESCE(nullable_col, 'default')` where the second argument is a non-null
literal.

**Effort:** Low-Medium — control flow changes in `GpuCoalesce.columnarEval`.

#### Current Inefficiency

`nullExpressions.scala:47-104` — `GpuCoalesce` iterates over all children in
reverse order, evaluating every single one. For `COALESCE(a, b, 42, c, d)` where
`42` is a non-null literal, children `c` and `d` are evaluated (potentially
expensive GPU operations) and then immediately discarded when the non-null scalar
`42` is encountered. Even after the scalar, `b` is evaluated and GpuNvl'd against
`a` when the non-null scalar already guarantees no null rows.

#### Potential Optimization

Two optimizations:

1. **Plan-time pruning:** Before evaluating, scan children right-to-left and
  truncate at the first non-nullable child (any `GpuLiteral` with `value != null`,
   or any expression with `nullable == false`). Everything after that position is
   dead code.
2. **Runtime early-exit:** After each `GpuNvl` call, check `runningResult.hasNulls`
  — if false (no nulls remain), skip remaining children. `hasNulls` is a cheap
   metadata check (reads a cached null count, no GPU kernel).

#### High-Level Change & Risks

**Change:** Add plan-time child pruning and runtime `hasNulls` early-exit to
`GpuCoalesce.columnarEval`.

**Risks:**

- **Side effects.** If any child expression has side effects (unlikely for typical
SQL, possible with UDFs), skipping evaluation changes observable behavior.
Plan-time pruning should respect `Expression.deterministic`.
- **Nullable metadata accuracy.** `Expression.nullable` can be conservative
(returning true when the expression never actually produces nulls). Plan-time
pruning based on `nullable == false` is safe but may miss opportunities. The
runtime `hasNulls` check catches these dynamically.

---

### 12. `array_join` with Null Replacement: Remove cuDF Workaround

**Identified by:** Cursor GPT-5.4

**Importance:** Medium — `array_join` with a non-null replacement is a concrete
low-hanging fruit with an explicit workaround comment in the code.

**Effort:** Low-Medium — tightly scoped change, depending on upstream cuDF fix
availability.

#### Current Inefficiency

`collectionOperations.scala` — the `array_join` path with a non-null replacement
uses an explicit workaround for a cuDF issue. That workaround materializes a null
mask over the list child, replaces null children with `ifElse`, swaps the list
child data column view, then forces a `copyToColumnVector()` — an extra full
list-data rewrite and copy.

#### Potential Optimization

- Remove the workaround when the relevant cuDF bug is fixed and available in the
dependency stack.
- If not yet fixed upstream, add a more direct JNI helper for replacing null list
children without forcing a full copy of the list column.

#### High-Level Change & Risks

**Change:** Check cuDF issue status. If fixed: switch to native path, remove
workaround. If not: evaluate whether a small JNI helper can eliminate the extra
copy.

**Risks:**

- **Nested list/string ownership.** This path depends on nested list/string
ownership and null-mask behavior. A naive rewrite could break correctness for
nested null cases.
- **Dependency alignment.** If the optimization depends on a newer cuDF behavior,
rollout timing depends on dependency alignment.

---

### 13. `cast(timestamp as string)`: Reduce Multi-Pass String Cleanup

**Identified by:** Cursor GPT-5.4

**Importance:** Medium — timestamps are common in NDS and the pattern is a clear
multi-pass string pipeline.

**Effort:** Low-Medium — contained to one cast path.

#### Current Inefficiency

`GpuCast.scala` — the current implementation does at least three logical
string-processing stages:

1. Format the timestamp as `%Y-%m-%d %H:%M:%S.%6f`
2. Remove `.000000` via string replace
3. Run a regex-based backreference replacement to trim other trailing fractional
  zeros

This is a multi-pass string cleanup pipeline over a full column.

#### Potential Optimization

- A one-pass formatter or trimming helper that emits Spark-compatible timestamp
strings directly.
- Short of that, a targeted non-regex fractional-trimming helper would reduce work
compared to the "format, replace, regex-rewrite" sequence.

#### High-Level Change & Risks

**Change:** Replace multi-pass format-then-cleanup with a more direct formatter.
May require a JNI helper or cuDF enhancement.

**Risks:**

- **Precision trimming rules.** Timestamp formatting is sensitive to precision
semantics. A faster path that misses corner cases (all-zero fractions, mixed
precision) would be hard to justify.
- **Cross-layer coordination.** If the best implementation lives below the Java
layer, this may require JNI or cuDF coordination.

---

### 14. boolInverted / boolToInt: Use Native `not()` and `castTo(INT32)`

**Identified by:** Claude 4.6 Opus

**Importance:** Low — these helpers are called from `GpuIf`'s gather path.

**Effort:** Very Low — single-line replacements.

#### Current Inefficiency

`conditionalExpressions.scala:84-101` — `boolToInt` creates two `GpuScalar`
wrappers to convert boolean to 0/1 via `ifElse`. cuDF's `castTo(DType.INT32)` does
this natively. `boolInverted` creates two `GpuScalar` wrappers for boolean
inversion when `cv.not()` exists natively.

#### Potential Optimization

```scala
private def boolToInt(cv: ColumnVector): ColumnVector = cv.castTo(DType.INT32)

def boolInverted(cv: ColumnVector): ColumnVector = {
  withResource(cv.not()) { notCv =>
    withResource(Scalar.fromBool(true)) { trueScalar =>
      notCv.replaceNulls(trueScalar)
    }
  }
}
```

#### High-Level Change & Risks

**Change:** Replace `ifElse`-based conversions with native cuDF operations.

**Risks:**

- **boolToInt null semantics.** The current `ifElse` maps null predicates to the
false branch (0), while `castTo(INT32)` maps null to null. The downstream
`scan(SUM, EXCLUSIVE, INCLUDE)` may propagate nulls differently. Verify that
callers handle null INT32 values correctly, or add
`replaceNulls(Scalar.fromInt(0))` after the cast.

---

### 15. GpuFloatArrayMin/Max: Add Early-Exit for Common All-Valid Case

**Identified by:** Claude 4.6 Opus

**Importance:** Low-Medium — affects `array_min` / `array_max` on float/double
arrays.

**Effort:** Low — add branch checks around existing code.

#### Current Inefficiency

`collectionOperations.scala:925-971` — always computes both the "all-NaN-or-null"
result (`trueOption`, 3 GPU operations) and the "normal min/max" result
(`falseOption`, 3 GPU operations), then selects between them. In the common case
where no list is entirely NaN/null, `trueOption` is computed for nothing — wasting
3 GPU kernel launches per batch.

#### Potential Optimization

Check `allNanOrNull.any()` (a cheap scalar reduction) before computing
`trueOption`:

```scala
withResource(allNanOrNull) { allNanOrNull =>
  val anyAllNan = withResource(allNanOrNull.any()) { a => a.isValid && a.getBoolean }
  if (!anyAllNan) {
    // Fast path: just compute falseOption directly
    ...
  } else {
    // Slow path: existing code (both branches)
    ...
  }
}
```

#### High-Level Change & Risks

**Change:** Add `any()` guard before computing the NaN-handling branch.

**Risks:**

- **Correctness.** The `any()` reduction correctly identifies whether any row
needs the NaN-handling path. When false, skipping `trueOption` is safe.
- **Marginal overhead in rare case.** When `anyAllNan` is true, the `any()`
reduction adds one extra kernel — negligible vs. the 7 that follow.

---

## Tier 3 — Significant Optimizations: Medium to High Effort

These items offer high potential performance improvement but involve cross-repo
changes, kernel-level work, or more complex implementation. They remain specific
to identifiable operator paths.

---

### 16. Hash Join: Reuse cuDF `HashJoin` Hash Table Across Stream Batches

**Identified by:** Claude 4.6 Opus, Codex GPT-5.4, Cursor GPT-5.3/Codex

**Importance:** High — Joins account for 9.6% of total NVTX time. `hash Inner gather` alone is ~94.3s. Join gather size estimation (`calc gather size`) adds
~11.8s. Combined join-streaming and join-materialization families total ~3174s.
Broadcast hash joins iterate over many stream batches with the same build side.

**Effort:** Medium — requires cross-repo changes (spark-rapids-jni +
spark-rapids). (All three identifying models agreed on Medium effort.)

#### Current Inefficiency

**Hash table rebuild:** `GpuHashJoin.scala:1263-1305` — every call to
`createGatherer` (once per stream batch) re-projects the build keys and calls
`JoinPrimitives.hashInnerJoin(leftKeys, rightKeys, compareNullsEqual)`. Under
the hood (`join_primitives.cu:142`), this constructs a `cudf::hash_join` object,
builds the hash table, probes, and destroys it. For broadcast hash joins, the
build side is identical across all stream batches — the hash table is rebuilt
from scratch each time.

**Row count heuristic:** `computeNumJoinRows` at line 1252-1261 uses a heuristic
estimate (`ceil(cb.numRows * buildStats.streamMagnificationFactor)`) instead of
an exact count. Over-estimates waste memory; under-estimates trigger OOM-retry-split
cycles visible at lines 1289-1304. The code has a TODO referencing cudf issue
#9053.

**Gather size estimation overhead:** The `JoinGatherer` repeatedly recomputes
expensive intermediates (row bit counts, prefix sums) for gather sizing. `calc gather size` is ~11.8s of pure estimation overhead across NDS.

#### Potential Optimization

The cuDF Java API provides `HashJoin(Table buildKeys, boolean compareNulls)` at
`HashJoin.java:72`, which builds the hash table once and supports reuse.
`Table.innerJoinRowCount(HashJoin)`, `Table.leftJoinRowCount(HashJoin)`, etc. at
`Table.java:2968-3488` return exact output sizes with negligible cost when reusing
a pre-built hash table.

Additionally, caching row-bit-count vectors per underlying spillable batch for
the `JoinGatherer` can eliminate repeated estimation work.

#### High-Level Change & Risks

**Change (3 parts):**

1. **spark-rapids-jni:** Extend `JoinPrimitives` to expose a persistent
  `cudf::hash_join` handle. Add `createHashJoin`, `hashInnerJoin(handle, ...)`,
   and `destroyHashJoin(handle)`.
2. **spark-rapids:** In `BaseHashJoinIterator`, build the `HashJoin` object
  lazily on first `createGatherer` call and cache it for the iterator's
   lifetime. Use handle-based probe APIs for subsequent batches.
3. **Exact row counts + gather estimation caching:** Replace
  `computeNumJoinRows` heuristic with
   `probeKeys.innerJoinRowCount(cachedHashJoin)`. Cache row-bit-count vectors
   in gatherer implementations for reusable size-estimation artifacts.

**Risks:**

- **Memory lifetime.** The cached `HashJoin` holds a reference to build-side key
data in GPU memory for the iterator's entire lifetime. For large build tables,
this increases peak memory. The spill framework should be evaluated for
spill/rebuild-on-demand support.
- **Build-side mutations.** Should not occur for broadcast joins, but defensive
code should assert immutability.
- **JNI lifecycle complexity.** Persistent native handles across JNI require
careful create/destroy pairing. The existing `HashJoin.java` in cuDF already
handles this pattern.
- **Shuffled hash joins.** Build side may change between iterations for non-
broadcast joins. Hash table reuse only applies when the build side is fixed —
primarily broadcast hash joins and `GpuShuffledSizedHashJoinExec`.
- **Cache invalidation for gather estimation.** Incorrect invalidation can
produce wrong gather row estimates (perf regressions or OOM retries). Caching
extra columns could increase memory pressure if lifecycle is not tightly
bounded.

---

### 17. Conditional Join Filtering: Fuse AST Predicate Evaluation with Compaction

**Identified by:** Cursor GPT-5.4, Codex GPT-5.4

**Importance:** High — Join work is central in NDS (~97/99 queries), and the
current conditional join filtering implementation is a clear multi-pass pipeline.
The `dispatch_map_type` join-mapping kernel alone is ~5.70% of total kernel time.

**Effort:** Medium — requires C++/CUDA changes in `spark-rapids-jni` and JNI
contract stability.

#### Current Inefficiency

`spark-rapids-jni/src/main/cpp/src/join_primitives.cu` —
`filterGatherMapsByAST` materializes candidate pairs and then runs a separate
filtering pipeline:

1. Evaluate AST predicate for every pair into a `keep_mask`
2. Count true entries with `thrust::count`
3. Allocate exact-sized output vectors
4. Compact matching pairs with `thrust::copy_if`

This is a multi-pass design with an intermediate boolean vector and at least one
extra full pass over the candidate set. For large joins with many candidate pairs
that are mostly eliminated by the non-equi predicate, that extra work is expensive
in both bandwidth and temporary allocation pressure.

Additional related post-processing inefficiencies: outer/semi/anti conversion does
`for_each + count + copy_if/fill` sequences — also multi-pass over the same data.

#### Potential Optimization

- Block-local or warp-local compaction while evaluating the predicate
- A two-stage compacting filter that avoids materializing a full boolean vector
- Specialized fast paths when the join predicate shape is simple enough to fuse
more aggressively
- Reuse temporary buffers for mask/index transformations

#### High-Level Change & Risks

**Change:** Keep the current external API in `GpuHashJoin`. Replace the internal
filtering implementation in `join_primitives.cu` to produce compacted left/right
gather maps more directly from predicate evaluation.

**Risks:**

- **Join predicate semantics.** Null handling and AST evaluation order are subtle.
Some join types swap left/right inputs to match compiled AST expectations; any
new fused path must preserve that behavior.
- **Output ordering.** Must remain compatible with the rest of the join stack.
- **Register pressure.** Reducing passes can increase kernel complexity and
register pressure, so final design needs benchmarking across both selective and
non-selective predicates.
- **Maintainability.** Additional kernel complexity reduces maintainability and
portability.
- **Correctness matrix.** Null markers, left/right map alignment, and ordering
across all join types (inner, left, right, full, semi, anti) must be preserved.

---

### 18. Conditional Expressions: `GpuIf` and `GpuCaseWhen` Lazy Evaluation

**Identified by:** Cursor GPT-5.4

**Importance:** High — NDS uses `CASE` heavily (~~1160 `GROUP BY`, ~1052 `CASE`
expressions, ~728 `CAST`s). Kernel pivot shows non-trivial time in logical
condition kernels such as `NullLogicalAnd` (~~2.46% of total kernel time).

**Effort:** Medium — incremental extension of existing fused path, but must
preserve Spark conditional semantics.

#### Current Inefficiency

`**GpuIf`:** Eagerly evaluates both branches in the side-effect-free path, then
merges with `ifElse`. This does nearly twice the expression work that CPU Spark
would do when one branch is substantially more expensive than the other.

`**GpuCaseWhen`:** Only uses its fused JNI path (`case_when.cu`) when all
`THEN`/`ELSE` values are scalars. The generic path still recursively builds the
result using repeated conditional merges, creating multiple temporary columns for
deep `CASE` expressions.

#### Potential Optimization

Move toward row-subset or selection-based execution rather than full-column eager
evaluation:

- Generalize the existing fused `CaseWhen.selectFirstTrueIndex` approach to support
vector-valued `THEN` branches, not just scalar ones
- Use predicate-derived row subsets to evaluate only the rows that need each branch
- Add specialized fast paths for common binary `IF` cases where one branch is cheap
or scalar

#### High-Level Change & Risks

**Change (incremental):**

1. Extend `GpuCaseWhen` first — the plugin already has a fused scalar path and
  clear selection abstraction
2. Reuse the same selection machinery to reduce eager evaluation in `GpuIf`
3. Keep the current side-effect-aware path separate and unchanged

**Risks:**

- **Spark conditional semantics.** Sensitive to short-circuit behavior, exceptions,
null behavior, and side effects. Making `GpuIf` lazier must not change ANSI
failure behavior.
- **Gather/scatter tradeoff.** Row-subset execution can trade full-column work for
more gather/scatter work, so some expression shapes may regress if the heuristic
is poorly chosen.

---

### 19. `substring`: Reduce Preparatory Column Operations

**Identified by:** Cursor GPT-5.4

**Importance:** Medium — substring and string-gather kernels appear in the profile
pivot (substring kernel: ~1.14% of total kernel time). The current implementation
is operationally busy before issuing the final substring.

**Effort:** Medium — either tighten the Java/cuDF pipeline or implement a
lower-level JNI helper.

#### Current Inefficiency

`stringFunctions.scala` — the current `substring` implementation does significant
preparatory work:

1. Compute character lengths
2. Derive Spark-compatible start positions
3. Derive and clamp end positions in `INT64`
4. Cast ends back to `INT32`
5. Rebuild validity handling before calling `substring`

This is semantically correct but creates several full-column intermediates for
what is conceptually a single operation.

#### Potential Optimization

- Tighten the pipeline to reduce temporary columns and repeated passes
- Implement a fused lower-level helper that directly applies Spark substring
semantics (1-based indexing, negative positions, overflow clamping)

#### High-Level Change & Risks

**Change:** Prototype a fused helper, keep the current multi-step code as fallback.

**Risks:**

- **UTF-8 character indexing.** Easy to get wrong if the implementation drifts
from Spark semantics (1-based, negative positions, character-not-byte indexing).
- **Null propagation.** A fused path that ignores validity semantics could break
null propagation.

---

## Tier 4 — Lower Priority or Higher Effort

These are still valid optimizations but affect less critical code paths, require
deep kernel work, or yield smaller improvements.

---

### 20. Window SumBinaryFixer.fixUpDecimal: Avoid All-False Overflow Columns in No-Op Cases

**Identified by:** Claude 4.6 Opus

**Importance:** Low | **Effort:** Low-Medium

`GpuWindowExpression.scala:1505-1611` — in no-op cases (no previous result or
mask is all-false), allocates a full-size boolean column of `false` values which
is then OR'd with other overflow indicators. OR-ing with all-false is a no-op.

**Change:** Return a `None` sentinel for the overflow column in no-op cases and
skip the OR. **Risk:** Requires updating all pattern-match sites that consume the
overflow column to handle `None`.

---

### 21. GpuUnboundedToUnbounded: Eliminate Materialized Ones Column

**Identified by:** Claude 4.6 Opus

**Importance:** Low | **Effort:** Low

`GpuUnboundedToUnboundedAggWindowExec.scala:165-179` — creates a full N-row INT32
column of 1s to feed into `GpuCount(1)`. Could use
`GroupByAggregation.count(NullPolicy.INCLUDE)` on any existing column instead,
eliminating ~40MB per 10M-row batch.

**Change:** Modify `GpuCount(literal(1))` handling to use `count(INCLUDE)`.
**Risk:** Verify `GpuCount(1)` is the only aggregate triggering this path and
that count-with-include matches count-of-ones semantics.

---

### 22. SecondPassIterator.concat: Fast Path for Single-Batch Case

**Identified by:** Claude 4.6 Opus

**Importance:** Low | **Effort:** Very Low

`GpuUnboundedToUnboundedAggWindowExec.scala:384-406` — when `tables.length == 1`,
still does `Table → ColumnarBatch → SpillableColumnarBatch`. Could return the
input directly with `incRefCount()`.

**Change:** Add single-batch early return. **Risk:** Verify caller ownership
expectations — returning the input SCB means shared underlying data.

---

### 23. GpuNaNvl Scalar+Scalar: Host-Side Evaluation

**Identified by:** Claude 4.6 Opus

**Importance:** Low | **Effort:** Very Low

`nullExpressions.scala:304-308` — when both arguments are scalars, expands `lhs`
into a full column, then calls the column+scalar overload. The result is
deterministic from the two scalar values alone.

**Change:** Evaluate on the host: check `isValid` and `isNan`, then
`ColumnVector.fromScalar(result, numRows)`. **Risk:** `Scalar` may not have a
direct `isNan` method — may need to extract and check with Java's
`Float.isNaN()`/`Double.isNaN()`.

---

### 24. Decimal128 Divide and Precision Helpers

**Identified by:** Cursor GPT-5.4

**Importance:** Medium-High | **Effort:** High

**Primary code:** `spark-rapids-jni/src/main/cpp/src/decimal_utils.cu`

The decimal path contains two explicit algorithmic warning signs in the JNI layer:
`divide_unsigned` performs bit-at-a-time long division over 256 bits, and
`precision10` linearly scans powers of ten up to 76 digits. Source comments
explicitly call these out as poor implementations. These helpers sit underneath
decimal division, integer division, and several decimal cast/rounding paths.

**Potential optimization:** Replace linear `precision10` with a leading-zero-based
estimate plus short local correction. Add specialized fast paths for common
power-of-ten divisors. Reduce repeated `pow_ten` work in scale adjustment.

**Risks:** Decimal correctness is unforgiving (rounding mode, overflow, ANSI,
Spark precision semantics). This is JNI-level infrastructure, so regressions would
affect several operators at once.

---

### 25. `str_to_map`: Dedicated Lower-Level Parser

**Identified by:** Cursor GPT-5.4

**Importance:** Low-Medium | **Effort:** High

`stringFunctions.scala` — structurally heavy: split records → split entries →
create structs → rewrite list data → create map. Lots of intermediate nested-
column reshaping for what is conceptually a parser.

**Potential optimization:** A dedicated lower-level parser (JNI helper) that emits
the final nested structure more directly. **Risk:** Parsing semantics around
delimiters, empty tokens, nulls, and duplicate keys are easy to get wrong.
Workload importance is less clear than top-ranked items.

---

### 26. `concat_ws`: Avoid Unnecessary Scalar Broadcasting

**Identified by:** Cursor GPT-5.4

**Importance:** Low-Medium | **Effort:** Medium

`stringFunctions.scala` — can expand scalar or array inputs into full columns
before the final concatenation. Avoidable bandwidth when many inputs are scalar.

**Potential optimization:** Avoid broadcasting scalars into full columns; add fast
paths for common argument patterns. **Risk:** Input-shape variation makes it easy
to optimize the common case while complicating maintenance. Workload evidence is
relatively weak.

---

## Tier 5 — Broader Infrastructure Optimizations (Deprioritized)

These items emerged from multiple models' analyses and have high profiling impact,
but they are broader heuristic-based or system-level changes rather than specific,
isolated operator fixes. They are included for completeness.

---

### 27. Aggregate Pipeline Projection Overhead

**Identified by:** Codex GPT-5.4, Cursor GPT-5.3/Codex

**Importance:** High | **Effort:** Medium

Combined aggregate + projection spend is ~439s across NDS (`computeAggregate`,
`groupby`, `finalize agg`, `post-process`, projection tiers). GROUP BY appears in
~82/99 NDS queries.

**Inefficiency:** Multiple projection passes around aggregate stages (pre-process,
post-process, final reorder). Additional avoidable per-agg overhead in reduction
path (repeated column extraction in loops, rebuilding invariant objects per batch).

**Potential optimization:** Fuse or bypass no-op/simple projection tiers. Hoist
repeated column extraction outside tight loops. Add fast-paths for passthrough or
fixed-width expressions.

**Risks:** Projection/order semantics are subtle — easy to introduce schema/order
regressions. Retryability split between deterministic/non-deterministic expressions
must remain correct. Spill ownership boundaries must stay unchanged.

---

### 28. Filter No-Op Detection Overhead

**Identified by:** Codex GPT-5.4

**Importance:** High | **Effort:** Low

`Java:filter batch` ~65.9s. Current flow computes a full boolean mask and runs
`all()` to detect no-op filters. For selective filters, this no-op check is pure
overhead before the real filter.

**Potential optimization:** Gate or skip `all()` precheck under conditions likely
to be selective. Use lightweight selectivity hinting.

**Risks:** Poor heuristics can regress workloads dominated by true-noop filters.
Needs balanced defaults and measurable guardrails.

---

### 29. Top-N Iterator Algorithm

**Identified by:** Codex GPT-5.4

**Importance:** Medium | **Effort:** Medium

ORDER BY is common in NDS (~90/99 queries). The current iterative flow can
repeatedly sort and concat interim results, especially under large inputs.

**Potential optimization:** Use bounded top-k merge strategy with incremental
heap/selection semantics. Reduce full re-sort frequency as partitions stream in.

**Risks:** Ordering/tie-break semantics must remain bit-for-bit compatible with
Spark expectations. Offset handling must remain correct.

---

### 30. Parquet `filterBlocks` Setup Caching

**Identified by:** Cursor GPT-5.3/Codex

**Importance:** Medium | **Effort:** Low-Medium

`Java:filterBlocks` ~97.0s, appearing consistently across all 102 queries.
Repeated construction/compilation of Parquet filter machinery per file/block.
Some setup is invariant for a scan (schema, options, pushdown flags).

**Potential optimization:** Cache reusable filter compilation artifacts keyed by
relevant schema/options context. Reuse immutable portions across files in the
same scan task.

**Risks:** Incorrect cache key design can apply stale/incompatible filters. Must
ensure no leakage across scans/tasks with different schemas or confs.

---

### 31. Parquet Decode/Decompression Kernels

**Identified by:** Codex GPT-5.4, Cursor GPT-5.3/Codex

**Importance:** Very High | **Effort:** Very High

Parquet decode/decompression is the single largest kernel family (~29.86% of total
GPU kernel time). `decode_page_data_generic`* and `nvcomp::unsnap_kernel` are
dominant. These are deep cuDF C++/CUDA kernel optimizations with a broad format
correctness matrix (encodings, page sizes, nested types, nulls).

**Potential optimization:** Kernel-level improvements in page decode and
decode/decompress overlap. Stream usage, staging, and memory-access tuning for
high-frequency encodings. Likely a multi-PR campaign.

**Risks:** Architecture-specific regressions (A100 vs H100 vs other GPUs).
Potential correctness edge cases in nested/null/encoding combinations. Broad
CI/perf validation burden. Changes affect many downstream consumers beyond
Spark RAPIDS.

---

## Appendix A: Recurring Optimization Patterns

The operator-level optimizations above follow recurring anti-patterns in the
codebase. These can be used to find additional instances:


| Pattern                                          | Search Query                                                  | What to Look For                                                                             |
| ------------------------------------------------ | ------------------------------------------------------------- | -------------------------------------------------------------------------------------------- |
| `fromScalar` where scalar API exists             | `ColumnVector.fromScalar` near `ifElse`, `extractListElement` | A full column is created from a scalar only to pass to an API that has a scalar/int overload |
| `isNull` + `ifElse(null, col)`                   | `isNull` followed by `ifElse` with `Scalar.fromNull`          | Could be `mergeAndSetValidity` or `replaceNulls`                                             |
| `isNotNull` + `ifElse(col, replacement)`         | `isNotNull` followed by `ifElse`                              | Could be `replaceNulls`                                                                      |
| `findAndReplaceAll` for single-value replacement | `findAndReplaceAll` with 1-element columns                    | `equalTo` + `ifElse` or `replaceNulls`                                                       |
| Both branches of `ifElse` fully materialized     | Two expensive column operations followed by `ifElse`          | Early-exit or branch elimination                                                             |
| Double dedup / detect-then-act                   | `rowContainsDuplicates` followed by dedup                     | Unconditional dedup or check-via-result                                                      |


## Appendix B: Source Document Cross-Reference


| #   | Optimization                                    | Claude 4.6 | Codex GPT-5.4 | Cursor GPT-5.3 | Cursor GPT-5.4 |
| --- | ----------------------------------------------- | ---------- | ------------- | -------------- | -------------- |
| 1   | GpuNvl replaceNulls                             | 1.1        | —             | —              | —              |
| 2   | mergeNulls mergeAndSetValidity                  | 1.2        | —             | —              | —              |
| 3   | GpuRegExpExtractAll extractListElement(int)     | 1.3        | —             | —              | —              |
| 4   | GpuDivModLike replaceZeroWithNull               | 1.5        | —             | —              | —              |
| 5   | GpuSize replaceNulls                            | 2.8        | —             | —              | —              |
| 6   | GpuMapFromArrays double dedup                   | 2.2        | —             | —              | —              |
| 7   | GpuMonthsBetween .day() dedup                   | 2.5        | —             | —              | —              |
| 8   | GpuDayOfWeek cast elimination                   | 2.6        | —             | —              | —              |
| 9   | Hash Partitioning fused API                     | 1.4        | Item 2        | P0             | —              |
| 10  | normalizeNulls/nullSanitize mergeAndSetValidity | 2.3        | —             | —              | —              |
| 11  | GpuCoalesce short-circuit                       | 2.7        | —             | —              | —              |
| 12  | array_join workaround removal                   | —          | —             | —              | Item 5         |
| 13  | cast(timestamp as string)                       | —          | —             | —              | Item 6         |
| 14  | boolInverted/boolToInt native ops               | 3.1        | —             | —              | —              |
| 15  | GpuFloatArrayMin/Max early exit                 | 2.4        | —             | —              | —              |
| 16  | Hash Join reuse + gather estimation             | 2.1        | Item 1        | P1             | —              |
| 17  | Conditional join filtering                      | —          | Item 5        | —              | Item 1         |
| 18  | Conditional expressions lazy eval               | —          | —             | —              | Item 2         |
| 19  | substring optimization                          | —          | —             | —              | Item 4         |
| 20  | Window SumBinaryFixer                           | 3.2        | —             | —              | —              |
| 21  | Materialized ones column                        | 3.3        | —             | —              | —              |
| 22  | SecondPassIterator single-batch                 | 3.4        | —             | —              | —              |
| 23  | GpuNaNvl scalar+scalar                          | 3.5        | —             | —              | —              |
| 24  | Decimal128 divide/precision                     | —          | —             | —              | Item 3         |
| 25  | str_to_map parser                               | —          | —             | —              | Item 7         |
| 26  | concat_ws scalar broadcasting                   | —          | —             | —              | Item 8         |
| 27  | Aggregate projection overhead                   | —          | Item 3        | P3             | —              |
| 28  | Filter no-op detection                          | —          | Item 4        | —              | —              |
| 29  | Top-N iterator                                  | —          | Item 6        | —              | —              |
| 30  | Parquet filterBlocks caching                    | —          | —             | P2             | —              |
| 31  | Parquet decode/decompression                    | —          | Item 7        | P4             | —              |


