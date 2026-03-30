# Changelog

All notable changes to this project will be documented in this file.

The format is based on [Keep a Changelog](https://keepachangelog.com/en/1.0.0/),
and this project adheres to [Semantic Versioning](https://semver.org/spec/v2.0.0.html).

---

## [0.1.0] - 2026-03-29

### Added

- **Rebranded** from `ride-cli` to `wrang` — new name, clean start, proper semver from `v0.1.0`
- `FastDataLoader` — load CSV, Excel, Parquet, JSON with automatic format detection; supports lazy frames and chunked streaming
- `DataSaver` — save to CSV, Excel, Parquet, JSON with a single unified API
- `DataInspector` — data profiling: shape, memory usage, missing values, duplicate detection, data quality report
- `DataExplorer` — statistical analysis: correlations (Pearson/Spearman/Kendall), distribution analysis, outlier detection (IQR/Z-score), normality tests (Shapiro-Wilk/KS/Anderson-Darling), terminal plots via `plotext`
- `DataCleaner` — chainable data cleaning with undo stack: missing value imputation (9 strategies), duplicate removal, outlier handling, type coercion, text cleaning
- `DataTransformer` — feature engineering pipeline: categorical encoding (Label/OneHot/Ordinal/Target), feature scaling (6 methods), polynomial/interaction features, binning, feature selection
- `DataValidator` + `DataSchema` + `ColumnSchema` — schema-based validation with JSON/YAML schema I/O; infer schema from data; produce structured `ValidationResult` with violations
- `BatchCleaner` — apply the same cleaning operations to multiple DataFrames
- `TransformationPipeline` — builder-style chaining of transformer operations
- `RideConfig` dataclass — centralized configuration with `~/.wrang/config.json` persistence; `get_config()`, `update_config()`, `reset_config()` helpers
- Interactive CLI (`wrang`) — full menu-driven terminal interface with Rich formatting: load, inspect, explore, clean, transform, visualize, export, settings, SQL, HTML profile, validate
- Non-interactive CLI modes: `--inspect`, `--profile`, `--sql`, `--compare`, `--export`, `--chunk-size`, `--output-format json`
- HTML report generation via `generate_html_report()` — self-contained, no external dependencies
- DuckDB SQL query mode (`--sql` / menu option 9)
- Dataset comparison mode (`--compare`)
- Streaming / chunked processing mode (`--chunk-size`)
- Lazy import system in `wrang/__init__.py` — fast import, graceful missing-dependency handling
- Full test suite — 217 passing tests across 10 modules

### Technical

- Engine: [Polars](https://pola.rs) for ~10x throughput vs Pandas
- Python 3.10 / 3.11 / 3.12 support
- Zero CUDA dependencies; ~90% smaller footprint than heavy ML toolkits
- Config dir: `~/.wrang/`
- Entry point: `wrang` command

---

> Previous history of the `ride-cli` / `prepup-linux` lineage is intentionally not carried forward. `wrang` starts clean at `v0.1.0`.
