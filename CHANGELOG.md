# Changelog

All notable changes to this project will be documented in this file.

The format is based on [Keep a Changelog](https://keepachangelog.com/en/1.0.0/),
and this project adheres to [Semantic Versioning](https://semver.org/spec/v2.0.0.html).

---

## [0.2.2] - 2026-04-13

### Fixed
- `plotext` unpinned from `==5.2.8` to `>=5.3.0` — older version imported `pkg_resources` which is not available on Python 3.12+ without `setuptools`
- `pyfiglet` unpinned from `==0.8.post1` to `>=1.0.2` — same root cause; 1.x dropped the `pkg_resources` dependency

---

## [0.2.0] - 2026-04-13

### Added
- `ride` command alias — `ride` and `wrang` are now interchangeable entry points (backward compatibility for users of the earlier `ride-cli` project)
- Automated PyPI publishing via GitHub Actions — push a `v*` tag to release

### Fixed
- MODE imputation now correctly handles columns where null is tied with the most frequent value (`mode().drop_nulls()` before selecting the fill value)
- `wrang/cli/interface.py` was importing `__version__` from the old `ride` package name — corrected to `wrang`
- `wrang/cli/formatters.py` had a hardcoded version string — now reads from `wrang.__version__` dynamically
- README code examples used non-existent methods `.get_result()` and `.remove_outliers()` — corrected to `.get_cleaned_data()`, `.get_transformed_data()`, and `.handle_outliers()`

### Changed
- Development status promoted from Alpha → Beta
- `tests/test_automl.py` renamed to `tests/test_validator.py` (file tests the validator module, not the removed AutoML feature)

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
