# wrang — Manual Testing Guide

Run each command from the repo root after installing the package locally:

```bash
cd /path/to/ride_cli
pip install -e .
```

The sample datasets are in `datasets/`. All commands use `datasets/titanic.csv` unless noted.

---

## 1. Installation Verification

```bash
# Confirm the CLI is installed
wrang --version

# System info (shows all installed dependencies)
wrang --system-info

# Built-in help
wrang --help

# Help topics
wrang --help-topic usage
wrang --help-topic examples
wrang --help-topic formats
wrang --help-topic config
```

**Expected:** Version string, no import errors, help text printed.

---

## 2. Interactive Mode

```bash
# Launch interactive mode (no file)
ride

# Launch with a file pre-loaded
wrang datasets/titanic.csv
```

**Expected:** Welcome banner, main menu with 11 options (1–11 + $ + q).

### 2.1 Load Dataset (option 1)

1. Press `1`
2. Enter path: `datasets/titanic.csv`
3. **Expected:** "Loaded N rows × M columns", data summary panel

Repeat with other formats:
```
datasets/AmesHousing.csv
datasets/Fish.csv
datasets/camera_dataset.csv
```

### 2.2 Inspect Data (option 2)

Load titanic.csv first, then press `2`. Walk through each sub-option:

| Sub-option | What to verify |
|---|---|
| Dataset Overview | Rows, columns, memory usage, type breakdown |
| Column Details | All columns listed with dtype, null %, unique % |
| Statistical Summary | Mean/std/quartiles for numeric columns |
| Data Quality Report | Missing values, duplicates, quality warnings |
| Memory Usage Analysis | Per-column memory breakdown |
| Detect Data Issues | Lists constant columns, high-missing columns, potential ID cols |
| Change Column Data Type | Select a column → change dtype → verify in next overview |

### 2.3 Explore Data (option 3)

| Sub-option | What to verify |
|---|---|
| Correlations | Pearson/Spearman table with strength labels |
| Distributions | Skewness, kurtosis, symmetric/skewed label per column |
| Outliers | IQR / Z-score detection with counts |
| Categorical Analysis | Value counts for string columns |
| Histogram | ASCII histogram renders in terminal |
| Scatter Plot | ASCII scatter renders with correlation coefficient |
| Normality Test | Shapiro-Wilk p-values for numeric columns |

### 2.4 Clean Data (option 4)

| Sub-option | What to try |
|---|---|
| Handle Missing Values | Try each strategy: Drop, Mean, Median, Mode, Forward Fill, Backward Fill, Custom Value |
| Remove Duplicates | Choose "all columns" then "subset" |
| Handle Outliers | IQR Remove → verify row count drops; IQR Cap → verify row count stays same |
| Validate Data Types | Auto-detect mismatches |
| Quick Clean (ML-ready) | One-shot clean for modeling |

> After each clean operation, press `2 → Dataset Overview` to verify the change took effect.

### 2.5 Transform Data (option 5)

| Sub-option | What to try |
|---|---|
| Encode Categorical | Label → verify string columns become codes; OneHot → verify column count increases |
| Scale Features | Standard → verify near-zero mean; MinMax → verify [0,1] range |
| Feature Selection | Choose a target column, select K features |

### 2.6 Visualize Data (option 6)

| Sub-option | What to verify |
|---|---|
| Histogram | Select a numeric column — ASCII plot renders |
| Scatter Plot | Select two numeric columns — plot + correlation |
| Correlation Heatmap | Renders color-coded matrix in terminal |

### 2.7 Export Data (option 7)

1. Load titanic.csv, make one change (e.g. fill missing values)
2. Press `7`
3. Export as each format: `.csv`, `.xlsx`, `.parquet`
4. Verify file appears on disk
5. Reload the file with option `1` to confirm it round-trips correctly

### 2.8 Settings (option 8)

| Sub-option | What to try |
|---|---|
| View Current Settings | All config fields shown |
| Memory Settings | Change max_memory_usage_mb |
| Performance Settings | Change chunk_size |
| Reset to Defaults | Resets all values |

### 2.9 SQL Query — DuckDB (option 9)

```
# After loading titanic.csv, press 9 and try:

SELECT * FROM data LIMIT 10

SELECT Sex, AVG(Age) as avg_age, COUNT(*) as count
FROM data
GROUP BY Sex

SELECT * FROM data WHERE Age > 50 ORDER BY Fare DESC

SELECT Pclass, Survived, COUNT(*) as n
FROM data
GROUP BY Pclass, Survived
ORDER BY Pclass, Survived

exit
```

**Expected:** Each query returns a Rich table. "Replace current dataset?" prompt works.

### 2.10 HTML Profile (option 10)

1. Load titanic.csv
2. Press `10`
3. Confirm default filename or enter custom path
4. **Expected:** File saved to disk, open in browser → full HTML report with overview, column cards, correlation matrix, sample data

### 2.11 Validate Data (option 11) ← New Feature

1. Load any dataset
2. Press `11`

| Sub-option | What to try |
|---|---|
| Infer schema from current data | Table shows all columns with dtype, nullable, missing %, unique flag |
| Validate with inferred schema | Should PASS (data matches its own schema) |
| Save inferred schema to JSON | Creates `<name>_schema.json`; open file and tighten a rule |
| Validate from JSON schema file | Point to the saved schema → see violations for tightened rules |

**To produce a violation:**
1. Save schema as `titanic_schema.json`
2. Edit the file: find `"Cabin"` column, change `"max_missing_pct"` from ~80 to `5.0`
3. Run "Validate from JSON schema file" → pointing at the edited file
4. **Expected:** WARNING for Cabin exceeding missing % threshold

### 2.12 Quick Export ($)

1. Load a dataset, make a change
2. Press `$`
3. **Expected:** File saved as `<original_name>_processed.csv` in current directory

---

## 3. Non-interactive CLI Commands

### 3.1 Quick Inspect

```bash
# Text output (default)
wrang datasets/titanic.csv --inspect

# JSON output (CI-friendly)
wrang datasets/titanic.csv --inspect --output-format json

# Verify JSON is valid
wrang datasets/titanic.csv --inspect --output-format json | python -c "import sys,json; d=json.load(sys.stdin); print('rows:', d['rows'])"
```

### 3.2 DuckDB SQL (non-interactive)

```bash
# Single query from command line
wrang datasets/titanic.csv --sql "SELECT Sex, AVG(Age) FROM data GROUP BY Sex"

wrang datasets/AmesHousing.csv --sql "SELECT Neighborhood, AVG(SalePrice) as avg_price FROM data GROUP BY Neighborhood ORDER BY avg_price DESC LIMIT 10"

# JSON output
wrang datasets/titanic.csv --sql "SELECT COUNT(*) as n FROM data WHERE Survived=1" --output-format json
```

### 3.3 Dataset Comparison

```bash
# Compare two identical files (should show schema_match: true)
wrang --compare datasets/titanic.csv datasets/titanic.csv --output-format json

# Compare different files (should show schema_match: false)
wrang --compare datasets/titanic.csv datasets/Fish.csv --output-format json

# Verify output structure
wrang --compare datasets/titanic.csv datasets/Fish.csv --output-format json \
  | python -c "import sys,json; d=json.load(sys.stdin); print('schema_match:', d['schema_match'])"
```

### 3.4 Streaming Large Files

```bash
# Stream in chunks of 100 rows
wrang datasets/AmesHousing.csv --chunk-size 100

# Combine with inspect
wrang datasets/AmesHousing.csv --inspect --chunk-size 500
```

### 3.5 HTML Profile

```bash
# Generate and save to custom path
wrang datasets/titanic.csv --profile --export /tmp/titanic_profile.html

# Open in browser
xdg-open /tmp/titanic_profile.html   # Linux
open /tmp/titanic_profile.html        # macOS
```

### 3.6 Export

```bash
# Convert CSV to Parquet
wrang datasets/titanic.csv --export /tmp/titanic.parquet --format parquet

# Verify the parquet file
python -c "import polars as pl; df = pl.read_parquet('/tmp/titanic.parquet'); print(df.shape)"

# Convert to Excel
wrang datasets/Fish.csv --export /tmp/fish.xlsx --format excel
```

---

## 4. Edge Cases to Test

```bash
# Non-existent file — should print a friendly error, not a stack trace
wrang /nonexistent/path/data.csv

# Unsupported extension — should print supported formats
wrang datasets/titanic.csv.bak --inspect

# Empty SQL query — interactive mode should show a prompt
# (Enter empty string when asked for SQL query)

# Very small dataset
echo "a,b\n1,2" > /tmp/tiny.csv
wrang /tmp/tiny.csv --inspect
```

---

## 5. python -m wrang (module entrypoint)

```bash
python -m wrang
python -m wrang datasets/titanic.csv --inspect
python -m wrang --version
```

**Expected:** Same behaviour as the `ride` command.

---

## 6. Checklist Summary

Mark each as ✅ after verifying:

- [ ] `wrang --version` prints version
- [ ] `wrang` launches interactive menu
- [ ] Load CSV, Excel, Parquet files
- [ ] All 7 Inspect sub-options work
- [ ] All Explore sub-options work including terminal plots
- [ ] Missing value imputation (all 7 strategies)
- [ ] Duplicate removal
- [ ] Outlier detection + cap/remove
- [ ] Categorical encoding (Label, OneHot)
- [ ] Feature scaling (Standard, MinMax)
- [ ] Export to CSV, Parquet, Excel
- [ ] SQL query mode (interactive + `--sql` flag)
- [ ] Dataset comparison (`--compare`)
- [ ] HTML profile generation
- [ ] Validate Data menu (infer, validate, save, load from file)
- [ ] `--inspect --output-format json` produces valid JSON
- [ ] Quick Export (`$`) saves `_processed.csv`
- [ ] Settings reset works
