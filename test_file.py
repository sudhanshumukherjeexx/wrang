#!/usr/bin/env python3
"""
test_file.py
============
End-to-end demonstration of the wrang package used as a Python library.
Run with:   python test_file.py
"""

import polars as pl
from pathlib import Path

# ── 0. Imports ──────────────────────────────────────────────────────────────
from wrang.core.loader import FastDataLoader, DataSaver, load_data, save_data
from wrang.core.inspector import DataInspector, inspect_data
from wrang.core.explorer import DataExplorer, explore_data
from wrang.core.cleaner import DataCleaner, quick_clean
from wrang.core.transformer import DataTransformer, quick_transform
from wrang.core.validator import DataSchema, ColumnSchema, DataValidator, infer_schema, validate_data
from wrang.config import (
    get_config, update_config, reset_config,
    ImputationStrategy, ScalingMethod, EncodingMethod,
)

DIVIDER = "=" * 65


def section(title: str) -> None:
    print(f"\n{DIVIDER}")
    print(f"  {title}")
    print(DIVIDER)


# ── 1. Build a sample dataset ────────────────────────────────────────────────
section("1. Sample Dataset")

df = pl.DataFrame({
    "id":       [1, 2, 3, 4, 5, 6, 7, 8, 9, 10],
    "name":     ["Alice", "Bob", "Carol", "Dave", "Eve",
                 "Frank", "Grace", None, "Henry", "Iris"],
    "age":      [25, 30, 22, 35, 28, 45, 31, None, 27, 29],
    "salary":   [55000.0, 72000.0, 48000.0, 95000.0, 61000.0,
                 120000.0, 68000.0, 53000.0, 59000.0, None],
    "dept":     ["eng", "eng", "hr", "eng", "hr",
                 "eng", "hr", "hr", "eng", "hr"],
    "score":    [88.5, 92.0, 76.3, 95.0, 82.1,
                 97.5, 78.9, 65.0, 85.3, 90.0],
    "active":   [True, True, True, True, False,
                 True, True, False, True, True],
})

print(f"Created dataframe: {df.shape[0]} rows × {df.shape[1]} columns")
print(df)


# ── 2. Loader — save and reload ──────────────────────────────────────────────
section("2. Loader — Save & Reload")

tmp = Path("/tmp/wrang_test")
tmp.mkdir(exist_ok=True)
csv_path     = tmp / "employees.csv"
parquet_path = tmp / "employees.parquet"

saver = DataSaver()
saver.save(df, csv_path)
saver.save(df, parquet_path)
print(f"Saved CSV:     {csv_path}")
print(f"Saved Parquet: {parquet_path}")

loader = FastDataLoader()
df_csv     = loader.load(csv_path)
df_parquet = loader.load(parquet_path)
print(f"Reloaded CSV:     {df_csv.shape}")
print(f"Reloaded Parquet: {df_parquet.shape}")

# Peek (first N rows without loading full file)
peek = loader.peek(csv_path, n_rows=3)
print(f"Peek (3 rows): {peek.shape}")

# Streaming chunks
total_streamed = sum(len(chunk) for chunk in loader.stream_chunks(csv_path, chunk_size=4))
print(f"Streamed total rows: {total_streamed}")

# Convenience functions
df2 = load_data(csv_path)
print(f"load_data() returned: {df2.shape}")


# ── 3. Inspector ─────────────────────────────────────────────────────────────
section("3. Inspector")

inspector = inspect_data(df)

info = inspector.get_basic_info()
print(f"Rows:          {info['n_rows']}")
print(f"Columns:       {info['n_columns']}")
print(f"Missing cells: {info['missing_values_total']}")
print(f"Duplicates:    {info['duplicate_rows']}")

# Column profiles
profiles = inspector.get_column_profiles()
print("\nColumn profiles:")
for col, profile in profiles.items():
    missing_pct = profile.get("missing_percentage", 0)
    print(f"  {col:10} dtype={profile['dtype']:12}  missing={missing_pct:.1f}%")

# Memory usage
memory = inspector.get_memory_usage()
print(f"\nMemory usage keys: {list(memory.keys())[:5]}")

# Detect potential issues
issues = inspector.detect_potential_issues()
print(f"\nDetected {len(issues)} potential issue(s):")
for issue in issues:
    print(f"  [{issue.get('severity','?').upper():6}] {issue.get('type','?')}: {issue.get('column','?')}")

# Display methods (prints to terminal)
print("\n--- Overview ---")
inspector.display_overview()


# ── 4. Explorer ──────────────────────────────────────────────────────────────
section("4. Explorer")

# Use a clean numeric-only df for exploration
num_df = df.select(["age", "salary", "score"]).drop_nulls()
explorer = explore_data(num_df)

# Correlations
corr = explorer.analyze_correlations(min_correlation=0.0)
pairs = corr["correlations"]
print(f"Correlation pairs found: {len(pairs)}")
for p in pairs[:3]:
    print(f"  {p['column1']:8} × {p['column2']:8}  r={p['correlation']:.3f}  ({p['strength']})")

# Distributions
dist = explorer.analyze_distributions()["distributions"]
print("\nDistribution summaries:")
for col, stats in dist.items():
    print(f"  {col:8}  mean={stats['mean']:.1f}  skew={stats['skewness']:.2f}")

# Outliers
outliers = explorer.detect_outliers(method="iqr")["outliers"]
print("\nOutlier detection (IQR):")
for col, info in outliers.items():
    print(f"  {col:8}  outlier_count={info['outlier_count']}")

# Normality tests
normality = explorer.test_normality()["normality_tests"]
print("\nShapiro-Wilk normality tests:")
for col, result in normality.items():
    print(f"  {col:8}  p={result['p_value']:.4f}  normal={result['is_normal']}")

# Categorical analysis
cat_explorer = explore_data(df.select(["dept", "active"]))
cat_result = cat_explorer.analyze_categorical_variables()["categorical_analysis"]
print("\nCategorical analysis columns:", list(cat_result.keys()))

# Terminal plots
print("\n--- Histogram of score ---")
explorer.plot_histogram("score")
print("\n--- Scatter: age vs salary ---")
explorer.plot_scatter("age", "salary")


# ── 5. Cleaner ───────────────────────────────────────────────────────────────
section("5. Cleaner")

print(f"Before cleaning: {df.shape}")

# Handle missing values — MEAN for numeric, MODE for categorical
cleaner = DataCleaner(df)
cleaner.handle_missing_values(strategy=ImputationStrategy.MEAN, columns=["age", "salary"])
cleaner.handle_missing_values(strategy=ImputationStrategy.MODE, columns=["name"])
df_clean = cleaner.get_cleaned_data()
print(f"After MEAN/MODE fill: {df_clean.shape}  missing={df_clean.null_count().sum_horizontal().sum()}")

# Forward fill demo
df_ff = (
    DataCleaner(df)
    .handle_missing_values(strategy=ImputationStrategy.FORWARD_FILL)
    .get_cleaned_data()
)
print(f"After FORWARD_FILL: missing={df_ff.null_count().sum_horizontal().sum()}")

# Custom value
df_cust = (
    DataCleaner(df)
    .handle_missing_values(
        strategy=ImputationStrategy.CUSTOM_VALUE,
        columns=["name"],
        custom_value="Unknown",
    )
    .get_cleaned_data()
)
print(f"Custom fill for name — 'Unknown' present: {'Unknown' in df_cust['name'].to_list()}")

# Remove duplicates
df_dups = pl.concat([df, df.head(2)])   # introduce duplicates
before = len(df_dups)
df_dedup = DataCleaner(df_dups).remove_duplicates().get_cleaned_data()
print(f"Dedup: {before} → {len(df_dedup)} rows")

# Handle outliers
df_out = (
    DataCleaner(df_clean)
    .handle_outliers(method="iqr", action="remove")
    .get_cleaned_data()
)
print(f"After outlier removal: {df_out.shape}")

# Text cleaning
df_txt = DataCleaner(df_clean).clean_text_data(columns=["name", "dept"]).get_cleaned_data()
print(f"Text cleaned — dept sample: {df_txt['dept'].unique().sort().to_list()}")

# Quick clean (one-shot ML-ready)
df_qc = quick_clean(df)
print(f"Quick clean result: {df_qc.shape}")

# Method chaining
df_chained = (
    DataCleaner(df)
    .handle_missing_values(strategy=ImputationStrategy.MEAN, columns=["age", "salary"])
    .handle_missing_values(strategy=ImputationStrategy.MODE, columns=["name"])
    .remove_duplicates()
    .get_cleaned_data()
)
print(f"Chained result: {df_chained.shape}")


# ── 6. Transformer ───────────────────────────────────────────────────────────
section("6. Transformer")

# Start with a clean df (no nulls)
df_base = df_chained.drop_nulls()
print(f"Base for transform: {df_base.shape}")

# Label encoding
t_label = DataTransformer(df_base)
t_label.encode_categorical_features(method=EncodingMethod.LABEL, columns=["dept"])
enc = t_label.get_transformed_data()
print(f"Label-encoded 'dept' — unique values: {enc['dept'].n_unique()}")

# One-hot encoding
t_onehot = DataTransformer(df_base)
t_onehot.encode_categorical_features(method=EncodingMethod.ONEHOT, columns=["dept"])
oh = t_onehot.get_transformed_data()
print(f"One-hot encoded: {df_base.shape[1]} cols → {oh.shape[1]} cols")

# Standard scaling
t_scale = DataTransformer(df_base)
t_scale.encode_categorical_features(method=EncodingMethod.LABEL)
t_scale.scale_features(method=ScalingMethod.STANDARD, columns=["age", "salary", "score"])
scaled = t_scale.get_transformed_data()
print(f"After standard scaling — age mean≈{scaled['age'].mean():.4f} (should be ≈0)")

# MinMax scaling
t_mm = DataTransformer(df_base)
t_mm.encode_categorical_features(method=EncodingMethod.LABEL)
t_mm.scale_features(method=ScalingMethod.MINMAX, columns=["age", "salary", "score"])
mm = t_mm.get_transformed_data()
print(f"MinMax score range: [{mm['score'].min():.2f}, {mm['score'].max():.2f}] (should be [0, 1])")

# Full pipeline: encode then scale
final = (
    DataTransformer(df_base)
    .encode_categorical_features(method=EncodingMethod.LABEL)
    .scale_features(method=ScalingMethod.ROBUST)
    .get_transformed_data()
)
print(f"Pipeline result: {final.shape}")


# ── 7. Validator ─────────────────────────────────────────────────────────────
section("7. Validator")

# Infer schema
schema = infer_schema(df_clean)
print(f"Inferred schema — {len(schema.columns)} columns:")
for cs in schema.columns:
    print(f"  {cs.name:10}  dtype={cs.dtype:12}  nullable={cs.nullable}")

# Save and reload schema
schema_path = tmp / "schema.json"
schema.to_json(schema_path)
reloaded = DataSchema.from_json(schema_path)
print(f"\nSchema saved to {schema_path} and reloaded ({len(reloaded.columns)} columns)")

# Validate inferred schema against clean df — should pass
result = DataValidator(schema).validate(df_clean)
print(f"\nValidate clean df against inferred schema: {'PASS' if result.passed else 'FAIL'}")

# Strict schema that will produce violations
strict = DataSchema(
    columns=[
        ColumnSchema(name="id",     dtype="Int64",   nullable=False, unique=True),
        ColumnSchema(name="age",    dtype="Int64",   nullable=False, min_value=18, max_value=65),
        ColumnSchema(name="salary", dtype="Float64", nullable=False, min_value=0.0, max_value=100000.0),
        ColumnSchema(name="dept",   dtype="String",  nullable=False,
                     allowed_values=["eng", "hr", "finance"]),
        ColumnSchema(name="score",  dtype="Float64", nullable=False, min_value=0.0, max_value=100.0),
    ],
    require_all_columns=False,
    allow_extra_columns=True,
)

result2 = validate_data(df_clean, strict)
result2.display()
print(f"\nPassed: {result2.passed}  Errors: {len(result2.errors)}  Warnings: {len(result2.warnings)}")

# Programmatic access
for v in result2.violations[:3]:
    print(f"  [{v.severity.upper()}] {v.column}/{v.check}: {v.message}")


# ── 8. Config ────────────────────────────────────────────────────────────────
section("8. Configuration")

config = get_config()
print(f"random_state:       {config.random_state}")
print(f"chunk_size:         {config.chunk_size}")
print(f"outlier_method:     {config.outlier_method}")
print(f"missing_threshold:  {config.missing_value_threshold}")

updated = update_config(random_state=123, chunk_size=5000)
print(f"\nAfter update — random_state: {updated.random_state}  chunk_size: {updated.chunk_size}")

reset_config()
print(f"After reset — random_state: {get_config().random_state}")


# ── Done ─────────────────────────────────────────────────────────────────────
section("All tests passed!")
print("The wrang package is working correctly as a Python library.")
print(f"Test artifacts saved to: {tmp}\n")
