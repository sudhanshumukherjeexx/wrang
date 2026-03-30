# wrang

**Lightning-fast data wrangling for the terminal.**

[![PyPI version](https://img.shields.io/pypi/v/wrang.svg?color=blue)](https://pypi.org/project/wrang/)
[![Python](https://img.shields.io/badge/python-3.10%2B-blue)](https://pypi.org/project/wrang/)
[![License](https://img.shields.io/badge/license-MIT-green)](LICENSE)
[![Tests](https://img.shields.io/badge/tests-217%20passing-brightgreen)](#testing)

`wrang` is a terminal-native data analysis toolkit. Load, inspect, clean, transform, and export datasets without writing a single line of boilerplate. Use it interactively, script it as a Python library, or wire it into CI pipelines.

---

## Install

```bash
pip install wrang
```

Optional extras:

```bash
pip install wrang[viz]       # matplotlib + seaborn
pip install wrang[advanced]  # duckdb + connectorx
pip install wrang[full]      # everything
```

Requires Python 3.10+.

---

## Quick start

```bash
# Interactive mode
wrang

# Load a file directly
wrang data.csv

# Quick inspect (CI-friendly JSON output)
wrang data.csv --inspect --output-format json

# SQL query via DuckDB
wrang data.csv --sql "SELECT dept, AVG(salary) FROM data GROUP BY dept"

# Generate self-contained HTML profile report
wrang data.csv --profile

# Compare two datasets
wrang --compare before.csv after.csv

# Stream large files in chunks
wrang large.csv --chunk-size 50000

# Export to Parquet
wrang data.csv --export clean.parquet
```

---

## Interactive menu

```
wrang
```

The full interactive session gives you a menu-driven workflow:

| Option | Action |
|--------|--------|
| `1` | Load dataset (CSV / Excel / Parquet / JSON) |
| `2` | Inspect — shape, types, missing values, quality report |
| `3` | Explore — correlations, distributions, outliers, plots |
| `4` | Clean — impute, deduplicate, handle outliers, fix types |
| `5` | Transform — encode, scale, polynomial features, binning |
| `6` | Visualize — terminal histograms, scatter, heatmap |
| `7` | Export — save to any supported format |
| `8` | Settings — configure wrang preferences |
| `9` | SQL Query — run DuckDB SQL against the current dataset |
| `10` | HTML Profile — generate a full standalone HTML report |
| `11` | Validate — check data against a JSON/YAML schema |
| `$` | Quick export — save current dataset instantly |
| `q` | Exit |

---

## Python API

`wrang` is also a full Python library. Every module is independently importable.

### Load & save

```python
from wrang import FastDataLoader, DataSaver

loader = FastDataLoader()
df = loader.load("sales.csv")           # auto-detects format
df_lazy = loader.scan_lazy("big.parquet")   # lazy frame for large files

saver = DataSaver()
saver.save(df, "output.parquet")
```

### Inspect

```python
from wrang import DataInspector

inspector = DataInspector(df)
info = inspector.get_basic_info()
print(info["n_rows"], info["missing_values_total"])

inspector.display_overview()         # rich terminal output
inspector.display_data_quality()
```

### Explore

```python
from wrang import DataExplorer

explorer = DataExplorer(df)

corr = explorer.analyze_correlations(method="pearson")
outliers = explorer.detect_outliers(method="iqr")
normality = explorer.test_normality()

explorer.plot_histogram("age")
explorer.plot_scatter("age", "salary")
explorer.plot_correlation_heatmap()
```

### Clean

```python
from wrang import DataCleaner
from wrang.config import ImputationStrategy

cleaned = (
    DataCleaner(df)
    .handle_missing_values(ImputationStrategy.MEDIAN, columns=["age", "salary"])
    .handle_missing_values(ImputationStrategy.MODE,   columns=["dept"])
    .remove_duplicates()
    .remove_outliers(method="iqr", factor=1.5)
    .get_result()
)
```

Supported imputation strategies: `DROP`, `MEAN`, `MEDIAN`, `MODE`, `FORWARD_FILL`, `BACKWARD_FILL`, `CUSTOM_VALUE`, `DISTRIBUTION`, `KNN`.

### Transform

```python
from wrang import DataTransformer, create_pipeline
from wrang.config import EncodingMethod, ScalingMethod

result = (
    DataTransformer(df)
    .encode_categorical_features(method=EncodingMethod.ONEHOT, columns=["dept"])
    .scale_features(method=ScalingMethod.STANDARD, columns=["age", "salary"])
    .get_result()
)

# Or use the pipeline builder
result = (
    create_pipeline(df)
    .encode_categorical_features(method=EncodingMethod.LABEL)
    .scale_features(method=ScalingMethod.ROBUST)
    .create_polynomial_features(degree=2)
    .get_result()
)
```

### Validate

```python
from wrang import DataSchema, ColumnSchema, DataValidator

schema = DataSchema(columns=[
    ColumnSchema(name="id",     dtype="Int64",   nullable=False, unique=True),
    ColumnSchema(name="salary", dtype="Float64", nullable=False, min_value=0.0),
    ColumnSchema(name="dept",   dtype="String",  allowed_values=["eng", "hr"]),
])

result = DataValidator(schema).validate(df)
print(result.passed)           # True / False
for v in result.violations:
    print(v.severity, v.message)

# Infer schema from data and save to file
from wrang import infer_schema
infer_schema(df).to_json("schema.json")
```

### Configuration

```python
from wrang.config import get_config, update_config, reset_config

config = get_config()
print(config.outlier_factor)   # 1.5

update_config(outlier_factor=2.0, chunk_size=5000)
reset_config()
```

User config is persisted at `~/.wrang/config.json`.

---

## Notebook usage

```python
import polars as pl
from wrang import DataInspector, DataCleaner, DataExplorer
from wrang.config import ImputationStrategy

df = pl.read_csv("titanic.csv")

# Profile the data
DataInspector(df).display_overview()

# Clean
df_clean = (
    DataCleaner(df)
    .handle_missing_values(ImputationStrategy.MEDIAN)
    .remove_duplicates()
    .get_result()
)

# Explore
explorer = DataExplorer(df_clean.select(["Age", "Fare", "Pclass"]))
explorer.plot_histogram("Age")
explorer.plot_scatter("Age", "Fare")
```

---

## Supported file formats

| Format | Read | Write | Notes |
|--------|------|-------|-------|
| CSV | ✓ | ✓ | Auto delimiter detection |
| Excel (`.xlsx`) | ✓ | ✓ | via openpyxl |
| Excel (`.xls`) | ✓ | — | via xlrd |
| Parquet | ✓ | ✓ | Columnar, fast |
| JSON / JSON Lines | ✓ | ✓ | Auto schema inference |

---

## Non-interactive CLI reference

```
wrang [FILE] [OPTIONS]

Options:
  --inspect                  Print dataset overview and exit
  --output-format {text,json}  Output format (default: text)
  --profile                  Generate standalone HTML report
  --sql QUERY                Run DuckDB SQL against FILE (table: "data")
  --compare FILE_A FILE_B    Diff two datasets
  --chunk-size N             Stream FILE in N-row chunks
  --export PATH              Export dataset to PATH
  --format {csv,excel,parquet,json}  Export format
  --version                  Show version and exit
  --help-topic {usage,examples,formats,config}
  --debug / --verbose
```

---

## Testing

```bash
pip install wrang[dev]
pytest tests/ -v
# 217 passed, 2 xfailed
```

---

## Contributing

1. Fork the repo
2. Create a feature branch
3. Run the test suite — all tests must pass
4. Open a pull request

Bug reports and feature requests → [GitHub Issues](https://github.com/sudhanshumukherjeexx/wrang/issues).

---

## License

MIT — see [LICENSE](LICENSE).

---

*Built with [Polars](https://pola.rs) and [Rich](https://github.com/Textualize/rich).*
