# wrang — Project Details

> **data wrangling toolkit** — A fast, interactive command-line toolkit for data analysis, cleaning, and feature engineering.

---

## Table of Contents

1. [Overview](#1-overview)
2. [Project Structure](#2-project-structure)
3. [Architecture](#3-architecture)
4. [Module Reference](#4-module-reference)
5. [CLI Interface](#5-cli-interface)
6. [Configuration System](#6-configuration-system)
7. [Data Flow](#7-data-flow)
8. [Public API](#8-public-api)
9. [Test Suite](#9-test-suite)
10. [Dependencies](#10-dependencies)
11. [Entry Points](#11-entry-points)
12. [Design Decisions](#12-design-decisions)
13. [Error Handling](#13-error-handling)
14. [Known Issues](#14-known-issues)

---

## 1. Overview

| Field | Value |
|---|---|
| **Version** | 0.1.0 |
| **Python** | 3.10+ |
| **License** | MIT |
| **Install** | `pip install wrang` |
| **Commands** | `wrang`, `python -m wrang` |
| **Previous name** | prepup-linux (fully rebranded) |

wrang replaces heavy notebook-based EDA workflows with a fast, terminal-native toolkit. Core goals:

- **Performance** — Uses [Polars](https://pola.rs) instead of Pandas for ~10x speed
- **Accessibility** — Fully interactive menu-driven CLI (no code required)
- **Modularity** — Composable Python API: `FastDataLoader`, `DataCleaner`, `DataTransformer`, etc.
- **Portability** — No CUDA, no heavy ML runtime; ~90% smaller footprint than alternatives
- **Scriptability** — Non-interactive command-line modes for CI pipelines

---

## 2. Project Structure

```
ride_cli/
├── wrang/                         # Main package
│   ├── __init__.py               # Lazy-loading public API gateway
│   ├── __main__.py               # Entry for `python -m wrang`
│   ├── main.py                   # cli_entry_point() — top-level entry
│   ├── config.py                 # Global RideConfig dataclass + helpers
│   ├── _public_api.py            # Explicit public symbol exports
│   │
│   ├── cli/                      # User-facing CLI layer
│   │   ├── __init__.py
│   │   ├── interface.py          # RideCLI — argument parsing, mode dispatch
│   │   ├── formatters.py         # RideFormatter — Rich-based terminal output
│   │   └── menus.py              # MenuHandler — interactive session state
│   │
│   ├── core/                     # Data processing layer
│   │   ├── __init__.py
│   │   ├── loader.py             # FastDataLoader, DataSaver
│   │   ├── inspector.py          # DataInspector
│   │   ├── explorer.py           # DataExplorer
│   │   ├── cleaner.py            # DataCleaner, BatchCleaner
│   │   ├── transformer.py        # DataTransformer, TransformationPipeline
│   │   └── validator.py          # DataValidator, DataSchema, ColumnSchema
│   │
│   ├── utils/                    # Shared utilities
│   │   ├── __init__.py
│   │   ├── exceptions.py         # Custom exception hierarchy
│   │   └── constants.py          # Type helpers, formatters, constants
│   │
│   └── viz/                      # Visualization / export
│       ├── __init__.py
│       └── export_utils.py       # generate_html_report()
│
├── tests/
│   ├── test_loader.py
│   ├── test_inspector.py
│   ├── test_explorer.py
│   ├── test_cleaner.py
│   ├── test_transformer.py
│   ├── test_config.py
│   ├── test_cli.py
│   ├── test_automl.py
│   ├── test_basic.py
│   └── test_utils.py
│
├── pyproject.toml                # Poetry/setuptools config, entry points
├── requirements.txt              # Pinned dependencies
├── CHANGELOG.md
└── README.md
```

---

## 3. Architecture

### 3.1 Layered Design

```
┌──────────────────────────────────────────┐
│  User Layer (CLI)                        │
│  RideCLI · MenuHandler · RideFormatter   │
└─────────────────┬────────────────────────┘
                  │
┌─────────────────▼────────────────────────┐
│  Processing Layer (Core)                 │
│  FastDataLoader · DataInspector          │
│  DataExplorer   · DataCleaner            │
│  DataTransformer· DataValidator          │
│  DataSaver                               │
└─────────────────┬────────────────────────┘
                  │
┌─────────────────▼────────────────────────┐
│  Engine Layer (Libraries)                │
│  Polars · NumPy · scikit-learn · SciPy   │
│  Rich   · DuckDB (optional)              │
└──────────────────────────────────────────┘
```

### 3.2 Key Design Patterns

| Pattern | Where Used | Purpose |
|---|---|---|
| Lazy Initialization | `wrang/__init__.py` | Fast import; graceful missing-dep handling |
| Singleton | `get_config()`, `get_formatter()` | Single global config/formatter instance |
| Method Chaining | `DataCleaner`, `DataTransformer` | Fluent API for pipeline construction |
| Strategy (Enum) | `ImputationStrategy`, `ScalingMethod`, `EncodingMethod` | Type-safe, extensible operation selection |
| Builder | `TransformationPipeline` | Step-by-step transformation assembly |
| Factory | `FastDataLoader` format detection | Auto-detect file type from extension |
| Snapshot Undo | `DataCleaner._undo_stack` | Revert individual or all operations |

---

## 4. Module Reference

### 4.1 `wrang/config.py` — Configuration

**Primary class:** `RideConfig` (dataclass)

```python
@dataclass
class RideConfig:
    # Performance
    random_state: int = 42
    max_memory_usage_mb: int = 1024
    chunk_size: int = 10000
    sample_size: int = 1000

    # File Handling
    supported_formats: List[str]  # csv, xlsx, xls, parquet, json
    default_encoding: str = "utf-8"
    csv_delimiter: str = ","

    # Data Processing
    missing_value_threshold: float = 0.9
    correlation_threshold: float = 0.90
    outlier_method: str = "iqr"
    outlier_factor: float = 1.5

    # Visualization
    plot_width: int = 80
    plot_height: int = 20

    # Feature Engineering
    max_features_for_onehot: int = 30
    max_features_for_label: int = 300
    default_test_size: float = 0.2
    cross_validation_folds: int = 5
```

**Key Enums:**

| Enum | Values |
|---|---|
| `FileFormat` | `CSV`, `EXCEL`, `PARQUET`, `JSON` |
| `ImputationStrategy` | `DROP`, `MEAN`, `MEDIAN`, `MODE`, `FORWARD_FILL`, `BACKWARD_FILL`, `CUSTOM_VALUE`, `DISTRIBUTION`, `KNN` |
| `ScalingMethod` | `STANDARD`, `MINMAX`, `ROBUST`, `MAXABS`, `QUANTILE_UNIFORM`, `QUANTILE_NORMAL` |
| `EncodingMethod` | `LABEL`, `ONEHOT`, `ORDINAL`, `TARGET` |

**Helper functions:**

```python
get_config() -> RideConfig           # Get/create global singleton
update_config(**kwargs)              # Modify fields on global config
reset_config()                       # Restore all defaults
get_config_dir() -> Path             # ~/.wrang/
load_user_config()                   # Load from ~/.wrang/config.json
save_user_config(config)             # Persist to ~/.wrang/config.json
```

---

### 4.2 `wrang/core/loader.py` — Data I/O

**`FastDataLoader`**

```python
class FastDataLoader:
    SUPPORTED_EXTENSIONS = {
        '.csv', '.xlsx', '.xls', '.parquet', '.json', '.jsonl'
    }

    load(file_path, **kwargs) -> pl.DataFrame
    scan_lazy(file_path, **kwargs) -> pl.LazyFrame   # For large files
    stream_chunks(file_path, chunk_size, **kwargs) -> Iterator[pl.DataFrame]
    infer_schema(file_path) -> Dict
    get_memory_requirements(file_path) -> float      # Estimated MB
```

**`DataSaver`**

```python
class DataSaver:
    save(df, file_path, format='csv', **kwargs) -> Path
    save_csv(df, file_path, **kwargs)
    save_excel(df, file_path, **kwargs)
    save_parquet(df, file_path, **kwargs)
    save_json(df, file_path, **kwargs)
```

**Convenience functions:** `load_data(file_path)`, `save_data(df, file_path)`

---

### 4.3 `wrang/core/inspector.py` — Data Profiling

**`DataInspector`**

```python
class DataInspector:
    def __init__(self, df: pl.DataFrame)

    # Analysis — return dicts
    get_basic_info() -> Dict
    get_column_profiles() -> Dict
    detect_potential_issues() -> List[str]
    get_memory_usage() -> Dict
    analyze_data_quality() -> Dict

    # Display — print to terminal
    display_overview()
    display_column_summary()
    display_data_quality()
    display_statistical_summary()
    detect_data_issues()
```

Metrics covered: shape, memory usage, missing values, data types, duplicate rows, potential outliers.

**Convenience:** `inspect_data(df)`

---

### 4.4 `wrang/core/explorer.py` — Statistical Analysis

**`DataExplorer`**

```python
class DataExplorer:
    def __init__(self, df: pl.DataFrame)

    analyze_correlations(method='pearson', min_correlation=0.1) -> Dict
    analyze_distributions(columns=None) -> Dict
    detect_outliers(method='iqr', columns=None) -> Dict
    analyze_categorical_variables(max_categories=20) -> Dict
    test_normality(columns=None, alpha=0.05) -> Dict

    plot_histogram(column, bins=20)
    plot_scatter(col1, col2, sample_size=1000)
    plot_correlation_heatmap(method='pearson')
```

**Supported methods:**

| Category | Methods |
|---|---|
| Correlation | Pearson, Spearman, Kendall |
| Outlier Detection | IQR, Z-score, Modified Z-score |
| Normality Tests | Shapiro-Wilk, Kolmogorov-Smirnov, Anderson-Darling |

**Convenience:** `explore_data(df)`

---

### 4.5 `wrang/core/cleaner.py` — Data Cleaning

**`DataCleaner`** (supports method chaining + undo)

```python
class DataCleaner:
    def __init__(self, df: pl.DataFrame)

    # Chainable operations (each returns self)
    handle_missing_values(strategy: ImputationStrategy,
                          columns=None,
                          custom_value=None) -> DataCleaner
    remove_duplicates(columns=None, keep='first') -> DataCleaner
    remove_outliers(method='iqr', columns=None, factor=1.5) -> DataCleaner
    handle_data_type_issues(auto_fix=True) -> DataCleaner
    clean_text_columns(lowercase=False, strip_whitespace=True) -> DataCleaner

    # Result / history
    get_result() -> pl.DataFrame
    undo() -> DataCleaner            # Revert last operation
    undo_all() -> DataCleaner        # Revert all operations
    get_cleaning_report() -> Dict
```

**`BatchCleaner`** — Apply same cleaning to multiple DataFrames.

**Convenience functions:**
```python
clean_data(df, strategy='auto') -> pl.DataFrame
quick_clean(df, mode='basic'|'aggressive'|'conservative'|'ml_ready') -> pl.DataFrame
```

---

### 4.6 `wrang/core/transformer.py` — Feature Engineering

**`DataTransformer`** (supports method chaining)

```python
class DataTransformer:
    def __init__(self, df: pl.DataFrame)

    # Encoding
    encode_categorical_features(method='label',
                                 columns=None,
                                 drop_original=True) -> DataTransformer

    # Scaling
    scale_features(method='standard',
                   columns=None,
                   feature_range=(0, 1)) -> DataTransformer

    # Feature Creation
    create_polynomial_features(columns=None, degree=2) -> DataTransformer
    create_interaction_features(columns=None) -> DataTransformer
    create_binned_features(column, n_bins=5, bin_names=None) -> DataTransformer

    # Feature Selection
    select_best_features(n_features=10,
                         score_func='f_classif',
                         target=None) -> DataTransformer

    get_result() -> pl.DataFrame
    get_transformation_report() -> Dict
```

**`TransformationPipeline`** — Builder-style chaining:

```python
result = create_pipeline(df) \
    .encode_categorical_features(method='onehot') \
    .scale_features(method='standard') \
    .create_polynomial_features(degree=2) \
    .get_result()
```

**Convenience functions:** `transform_data(df, **kwargs)`, `create_pipeline(df)`, `quick_transform(df, mode='ml_ready')`

---

### 4.7 `wrang/core/validator.py` — Schema Validation

**`ColumnSchema`**

```python
@dataclass
class ColumnSchema:
    name: str
    dtype: Optional[str] = None
    nullable: bool = True
    max_missing_pct: float = 100.0
    unique: bool = False
    min_value: Optional[float] = None
    max_value: Optional[float] = None
    allowed_values: Optional[List[Any]] = None
```

**`DataSchema`** — Collection of `ColumnSchema` with I/O support:

```python
DataSchema.from_json(path)   # Load from JSON
DataSchema.from_yaml(path)   # Load from YAML
DataSchema.load(path)        # Auto-detect format
schema.to_json(path)
schema.to_dict()
```

**`DataValidator`**

```python
class DataValidator:
    validate(df, schema) -> ValidationResult
    infer_schema(df, strict=False) -> DataSchema
```

**`ValidationResult`**

```python
class ValidationResult:
    passed: bool
    violations: List[ValidationViolation]
    to_dict() -> Dict
```

**Convenience functions:** `validate_data(df, schema)`, `infer_schema(df)`

---

### 4.8 `wrang/cli/interface.py` — CLI Application

**`RideCLI`**

```python
class RideCLI:
    def __init__(self)          # Initializes config, formatter, loader, menu_handler
    def run(args=None) -> int   # Parse args, dispatch to correct mode, return exit code

    # Non-interactive modes
    _run_interactive_mode() -> int
    _run_with_file(file_path, args) -> int
    _run_sql(file_path, query, args) -> int
    _run_compare(files, args) -> int
    _run_html_profile(df, file_path, args) -> int
    _run_streaming(file_path, chunk_size, args) -> int
    _run_command_line_inspect(df, output_format) -> int
    _run_command_line_export(df, output_path, format_type) -> int

    _show_version()
    _show_help(topic)
    _show_debug_info()
```

---

### 4.9 `wrang/cli/formatters.py` — Terminal Output

**`RideFormatter`** — Wraps Rich for consistent styling.

```python
class RideFormatter:
    # Output
    display_welcome_banner()
    display_main_menu(current_dataset=None)
    display_section_header(title, subtitle=None, icon='🔧')
    display_data_summary(df, filename=None)
    display_column_info(df, detailed=False)
    display_operation_result(operation, success, details=None, duration=None)
    display_error(error_title, error_message, suggestions=None)
    display_progress_operation(operation_name, items, operation_func)

    # Input
    prompt_user_choice(question, choices, default=None) -> str
    prompt_confirmation(message, default=False) -> bool
    prompt_file_path(message) -> str
```

**Color scheme:** `primary=cyan`, `success=green`, `warning=yellow`, `error=red`, `accent=magenta`

**Convenience:** `get_formatter()`, `clear_screen()`, `wait_for_input(message)`

---

### 4.10 `wrang/cli/menus.py` — Interactive Session

**`MenuHandler`** — Maintains state across menu operations.

```python
class MenuHandler:
    current_df: Optional[pl.DataFrame]   # Active dataset
    current_file: Optional[str]          # Loaded file path
    operation_history: List[Dict]        # Log of all operations

    run_main_menu()                      # Main interactive loop

    handle_load_dataset()        # Option 1
    handle_inspect_data()        # Option 2
    handle_explore_data()        # Option 3
    handle_clean_data()          # Option 4
    handle_transform_data()      # Option 5
    handle_visualize_data()      # Option 6
    handle_export_data()         # Option 7
    handle_settings()            # Option 8
    handle_sql_query()           # Option 9
    handle_html_profile()        # Option 10
    handle_validate_data()       # Option 11
    handle_quick_export()        # Option $ (shortcut)
```

---

### 4.11 `wrang/utils/exceptions.py` — Error Hierarchy

```
RideError (base)
├── DataLoadError          — file I/O failures
├── DataValidationError    — schema / type violations
├── PreprocessingError     — cleaning/transformation failures
│     attributes: operation, affected_columns, suggestions
├── ExportError            — save/export failures
├── MemoryError            — insufficient memory
│     attributes: required_memory_mb, available_memory_mb
└── UnsupportedOperationError
```

**Helpers:**
- `handle_polars_error(error, context)` — Converts Polars exceptions to `RideError`
- `create_user_friendly_message(exception)` — Formats error for display

---

### 4.12 `wrang/utils/constants.py` — Shared Utilities

**Type classification (per-Series):**
```python
is_numeric(series) -> bool
is_categorical(series) -> bool
is_datetime(series) -> bool
is_boolean(series) -> bool
```

**Column selectors (per-DataFrame):**
```python
numeric_columns(df) -> List[str]
categorical_columns(df) -> List[str]
datetime_columns(df) -> List[str]
boolean_columns(df) -> List[str]
```

**Formatters:**
```python
format_memory(memory_mb) -> str           # "1.5 MB" / "2.3 GB"
format_file_size(size_bytes) -> str
classify_correlation(abs_corr) -> str     # "Very Strong" … "Very Weak"
interpret_skewness(skew) -> str           # "highly skewed" / "approximately symmetric"
```

---

### 4.13 `wrang/viz/export_utils.py` — HTML Reports

```python
generate_html_report(
    df,
    output_path="data_profile.html",
    title="wrang — Data Profile Report",
    sample_rows=10,
    max_cat_values=10
) -> Path
```

Produces a **self-contained** HTML file (inline CSS/JS, no CDN dependencies) containing:
- Dataset overview (shape, memory, dtype distribution)
- Per-column statistics and quality metrics
- Correlation heatmap for numeric columns
- Sample data rows table

---

## 5. CLI Interface

### 5.1 Interactive Mode

```bash
ride                     # Start with empty session
ride data.csv            # Start with file pre-loaded
```

Main menu options:

| Key | Action |
|---|---|
| 1 | Load Dataset |
| 2 | Inspect Data |
| 3 | Explore Data |
| 4 | Clean Data |
| 5 | Transform Data |
| 6 | Visualize Data |
| 7 | Export Data |
| 8 | Settings |
| 9 | SQL Query |
| 10 | HTML Profile |
| 11 | Validate Data |
| $ | Quick Export |
| q | Exit |

### 5.2 Non-Interactive Modes

```bash
# Inspect & profile
ride data.csv --inspect
ride data.csv --inspect --output-format json
ride data.csv --profile                        # Generate HTML report

# SQL query (requires duckdb)
ride data.csv --sql "SELECT col, COUNT(*) FROM data GROUP BY col"

# Compare datasets
ride --compare before.csv after.csv

# Streaming / chunked processing
ride data.csv --chunk-size 10000

# Export
ride data.csv --export output.parquet
ride data.csv --export output.xlsx --format excel

# Info
ride --version
ride --help-topic usage|examples|formats
ride --debug
ride --verbose
```

### 5.3 Output Formats

| Flag | Description |
|---|---|
| `--output-format text` | Human-readable (default) |
| `--output-format json` | Machine-readable JSON (CI-friendly) |

---

## 6. Configuration System

### 6.1 Resolution Order

1. Hard-coded defaults in `RideConfig`
2. User config at `~/.wrang/config.json` (loaded at startup)
3. Runtime overrides via `update_config(**kwargs)`

### 6.2 Usage

```python
from wrang.config import get_config, update_config, reset_config, save_user_config

config = get_config()
print(config.outlier_factor)          # 1.5

update_config(outlier_factor=2.0, chunk_size=5000)
save_user_config()                    # Persist to ~/.wrang/config.json

reset_config()                        # Restore all defaults
```

### 6.3 Per-File Config

```python
file_config = config.get_file_config(Path("data.csv"))
# {'delimiter': ',', 'encoding': 'utf-8', 'quoting': 1, ...}
```

---

## 7. Data Flow

### 7.1 Typical Pipeline

```
Input File
    │
    ▼
FastDataLoader.load()
    │  Detects format by extension
    │  Loads into Polars DataFrame
    ▼
[DataInspector]          ← optional: understand the data
    │
    ▼
DataCleaner
    ├── handle_missing_values(strategy)
    ├── remove_duplicates()
    └── remove_outliers()
    │  .get_result() → cleaned pl.DataFrame
    ▼
DataTransformer
    ├── encode_categorical_features()
    ├── scale_features()
    └── create_polynomial_features()
    │  .get_result() → transformed pl.DataFrame
    ▼
[DataExplorer]           ← optional: analyze final state
[DataValidator]          ← optional: enforce schema
    │
    ▼
DataSaver.save()
    │
    ▼
Output File (CSV / Excel / Parquet / JSON)
```

### 7.2 Interactive Session Data Flow

```
ride start
  → display_welcome_banner()
  → MenuHandler.run_main_menu()       ← infinite loop
      → User picks option
      → handler dispatches to handle_*()
      → Core module processes current_df
      → Result stored back in MenuHandler.current_df
      → Repeat
```

---

## 8. Public API

Import any public symbol lazily from the top-level package:

```python
# Loading
from wrang import FastDataLoader, DataSaver, load_data, save_data

# Inspection
from wrang import DataInspector, inspect_data

# Exploration
from wrang import DataExplorer, explore_data

# Cleaning
from wrang import DataCleaner, BatchCleaner, clean_data, quick_clean

# Transformation
from wrang import DataTransformer, TransformationPipeline
from wrang import transform_data, create_pipeline, quick_transform

# Validation
from wrang import DataValidator, DataSchema, ColumnSchema, ValidationResult
from wrang import validate_data, infer_schema

# Configuration
from wrang import get_config, update_config, reset_config

# Formatting
from wrang import get_formatter
```

### Deprecated / Removed

| Old Symbol | Replacement |
|---|---|
| `Prepup` class | `DataCleaner`, `DataTransformer`, etc. |
| `AutoMLProcessor` | Removed — use `DataTransformer` |
| `automl_processor` module | Removed |
| `utils` module | Moved to `wrang/utils/` |

---

## 9. Test Suite

### 9.1 Running Tests

```bash
python -m pytest tests/ -v
# Expected: 217 passed, 2 xfailed, 0 failed
```

### 9.2 Coverage by File

| File | Tests | Notes |
|---|---|---|
| test_loader.py | 22 | FastDataLoader, DataSaver, formats |
| test_inspector.py | 20 | Overview, quality, column profiles |
| test_explorer.py | 27 | Correlations, outliers, normality |
| test_cleaner.py | 27 | All imputation strategies, undo |
| test_transformer.py | 15 | Encoding, scaling, features (2 xfail) |
| test_config.py | 26 | Load/save/update/reset config |
| test_cli.py | 13 | Argument parsing, mode dispatch |
| test_automl.py | 21 | AutoML-related functionality |
| test_basic.py | 3 | Basic imports and smoke tests |
| test_utils.py | 23 | Type helpers, formatters, constants |

### 9.3 Known Expected Failures (xfail)

Both in `tests/test_transformer.py`:

- `test_polynomial_adds_columns` — `PreprocessingError` in polynomial path does not accept `affected_columns` kwarg
- `test_polynomial_preserves_row_count` — same root cause

These are pre-existing known issues, not regressions.

### 9.4 Test Markers

```ini
# pyproject.toml
markers = ["slow", "unit", "integration"]
```

### 9.5 Common Fixtures

- `sample_df()` — Small Polars DataFrame
- `csv_file()` / `parquet_file()` — Temp files for I/O tests
- `loader()` / `saver()` — Preconfigured instances

---

## 10. Dependencies

### 10.1 Core (Always Installed)

| Package | Purpose |
|---|---|
| `polars-lts-cpu >= 0.20` | DataFrame engine (10x Pandas) |
| `numpy >= 1.26` | Numerical ops |
| `pyarrow >= 14.0` | Arrow format I/O |
| `scikit-learn >= 1.3` | Encoding, scaling, feature selection |
| `scipy >= 1.11` | Statistics, normality tests |
| `joblib >= 1.2` | Parallel processing |
| `rich >= 13.0` | Terminal formatting |
| `click >= 8.0` | CLI framework |
| `pydantic >= 2.0` | Config validation |
| `pyfiglet == 0.8.post1` | ASCII art banners |
| `plotext == 5.2.8` | Terminal plots |
| `blessed == 1.19.1` | Terminal control |
| `termcolor >= 2.0` | Colored output |
| `fastparquet >= 2024.0` | Parquet I/O |
| `openpyxl >= 3.1` | Excel .xlsx |
| `xlrd >= 2.0` | Excel .xls |
| `nbformat >= 5.9` | Notebook format |

### 10.2 Optional Extras

```bash
pip install wrang[viz]       # matplotlib, seaborn
pip install wrang[advanced]  # duckdb, connectorx, polars[all]
pip install wrang[full]      # viz + advanced
```

### 10.3 Platform Support

- **Python:** 3.10, 3.11, 3.12
- **OS:** Linux, macOS, Windows

---

## 11. Entry Points

### 11.1 pyproject.toml Scripts

```toml
[project.scripts]
wrang = "wrang.main:cli_entry_point"
```

### 11.2 Call Stack

```
wrang  (shell)
  → wrang/main.py : cli_entry_point()
    → wrang/main.py : main()
      → wrang/cli/interface.py : main()
        → RideCLI.run(args)
          ├── Interactive: MenuHandler.run_main_menu()
          └── Command-line: RideCLI._run_*()
```

### 11.3 Module Execution

```bash
python -m wrang         # wrang/__main__.py → same as above
```

---

## 12. Design Decisions

### Lazy Imports in `__init__.py`

`wrang/__init__.py` defers all module loading to first access via `__getattr__`. This means:
- `import wrang` is near-instant regardless of heavy dependencies
- Missing optional dependencies raise `ImportError` only when the relevant symbol is accessed
- Startup performance is not penalized by large optional imports

### Polars over Pandas

All DataFrames are `polars.DataFrame`. Polars was chosen for:
- ~10x throughput on typical EDA workloads
- Native lazy evaluation (`LazyFrame`) for memory-efficient large-file processing
- Strong type system that catches dtype issues early

### Undo Stack in DataCleaner

Each chainable operation snapshots `(df, log_entry)` onto `_undo_stack` before modifying `self.df`. This allows:

```python
cleaner.remove_outliers()   # snapshot taken
cleaner.undo()              # restored to pre-outlier state
```

Tradeoff: Memory usage doubles per operation step. Acceptable for typical interactive EDA sizes.

### Strategy Enums

`ImputationStrategy`, `ScalingMethod`, and `EncodingMethod` are Python enums rather than string constants. This gives:
- IDE autocompletion
- Runtime type checking
- Self-documenting code

### Configuration Singleton

A single `RideConfig` instance is shared across all modules via `get_config()`. Settings like `random_state`, `outlier_factor`, and `chunk_size` propagate automatically without needing to be passed through every function call.

---

## 13. Error Handling

Custom exceptions carry structured metadata:

```python
# DataLoadError
raise DataLoadError(file_path=path, message="File not found")

# PreprocessingError (note: affected_columns and suggestions kwargs
# are not yet supported on all code paths — see Known Issues)
raise PreprocessingError(
    operation="imputation",
    affected_columns=["age", "salary"],
    suggestions=["Use median for skewed distributions"]
)

# MemoryError
raise MemoryError(required_memory_mb=4096, available_memory_mb=1024)
```

Pattern for callers:

```python
try:
    loader.load(path)
except DataLoadError as e:
    formatter.display_error("Load Failed", str(e))
except wrang.utils.exceptions.MemoryError as e:
    formatter.display_error("Out of Memory",
        f"Need {e.required_memory_mb}MB, have {e.available_memory_mb}MB")
```

---

## 14. Known Issues

| Issue | Location | Status |
|---|---|---|
| `PreprocessingError.__init__` does not accept `affected_columns`/`suggestions` in polynomial feature creation path | `wrang/core/transformer.py` | Known; tests marked `xfail` |
| `polars-lts-cpu` `popcnt` CPU warning on some machines | Runtime | Cosmetic; install `polars-lts-cpu` wheel resolves it |
| `pyfiglet` uses deprecated `pkg_resources` API | `pyfiglet 0.8.post1` | Upstream issue; no action needed |

---

*Generated: 2026-03-29 | wrang v0.1.0*
