#!/usr/bin/env python3
"""
ride/core/validator.py
Schema-based validation for DataFrames.

Supports:
  - Inline schema definition (dict / Python objects)
  - JSON / YAML schema files
  - Per-column checks: dtype, nullable, missing-%, range, allowed values, uniqueness
  - Extra / missing column detection
  - Severity levels: error, warning
  - Rich-table output and dict serialisation
"""

from __future__ import annotations

import json
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Dict, List, Optional, Union

import polars as pl

# ---------------------------------------------------------------------------
# ColumnSchema
# ---------------------------------------------------------------------------

@dataclass
class ColumnSchema:
    """Definition of expected properties for one column."""

    name: str
    dtype: Optional[str] = None            # polars dtype name, e.g. "Float64", "Utf8"
    nullable: bool = True                  # may the column contain nulls?
    max_missing_pct: float = 100.0        # maximum allowed % of nulls (0-100)
    unique: bool = False                   # must all non-null values be unique?
    min_value: Optional[float] = None     # numeric lower bound (inclusive)
    max_value: Optional[float] = None     # numeric upper bound (inclusive)
    allowed_values: Optional[List[Any]] = None  # whitelist (for categoricals)

    # -----------------------------------------------------------------
    @classmethod
    def from_dict(cls, d: Dict[str, Any]) -> "ColumnSchema":
        return cls(
            name=d["name"],
            dtype=d.get("dtype"),
            nullable=d.get("nullable", True),
            max_missing_pct=float(d.get("max_missing_pct", 100.0)),
            unique=bool(d.get("unique", False)),
            min_value=d.get("min_value"),
            max_value=d.get("max_value"),
            allowed_values=d.get("allowed_values"),
        )

    def to_dict(self) -> Dict[str, Any]:
        return {
            "name": self.name,
            "dtype": self.dtype,
            "nullable": self.nullable,
            "max_missing_pct": self.max_missing_pct,
            "unique": self.unique,
            "min_value": self.min_value,
            "max_value": self.max_value,
            "allowed_values": self.allowed_values,
        }


# ---------------------------------------------------------------------------
# DataSchema
# ---------------------------------------------------------------------------

@dataclass
class DataSchema:
    """Collection of ColumnSchemas plus dataset-level settings."""

    columns: List[ColumnSchema] = field(default_factory=list)
    allow_extra_columns: bool = True       # True → extra cols are warnings only
    require_all_columns: bool = True       # True → missing cols are errors

    # -----------------------------------------------------------------
    @classmethod
    def from_dict(cls, d: Dict[str, Any]) -> "DataSchema":
        cols = [ColumnSchema.from_dict(c) for c in d.get("columns", [])]
        return cls(
            columns=cols,
            allow_extra_columns=d.get("allow_extra_columns", True),
            require_all_columns=d.get("require_all_columns", True),
        )

    @classmethod
    def from_json(cls, path: Union[str, Path]) -> "DataSchema":
        with open(path, "r", encoding="utf-8") as fh:
            return cls.from_dict(json.load(fh))

    @classmethod
    def from_yaml(cls, path: Union[str, Path]) -> "DataSchema":
        try:
            import yaml  # type: ignore
        except ImportError as exc:
            raise ImportError("PyYAML is required to load YAML schemas: pip install pyyaml") from exc
        with open(path, "r", encoding="utf-8") as fh:
            return cls.from_dict(yaml.safe_load(fh))

    @classmethod
    def load(cls, path: Union[str, Path]) -> "DataSchema":
        """Auto-detect JSON vs YAML by extension."""
        p = Path(path)
        if p.suffix in {".yaml", ".yml"}:
            return cls.from_yaml(p)
        return cls.from_json(p)

    def to_dict(self) -> Dict[str, Any]:
        return {
            "columns": [c.to_dict() for c in self.columns],
            "allow_extra_columns": self.allow_extra_columns,
            "require_all_columns": self.require_all_columns,
        }

    def to_json(self, path: Union[str, Path]) -> None:
        with open(path, "w", encoding="utf-8") as fh:
            json.dump(self.to_dict(), fh, indent=2)


# ---------------------------------------------------------------------------
# ValidationViolation / ValidationResult
# ---------------------------------------------------------------------------

@dataclass
class ValidationViolation:
    column: str
    check: str
    severity: str          # "error" | "warning"
    message: str
    actual: Optional[Any] = None
    expected: Optional[Any] = None

    def to_dict(self) -> Dict[str, Any]:
        return {
            "column": self.column,
            "check": self.check,
            "severity": self.severity,
            "message": self.message,
            "actual": self.actual,
            "expected": self.expected,
        }


class ValidationResult:
    """Holds all violations from a validation run."""

    def __init__(self) -> None:
        self.violations: List[ValidationViolation] = []

    # -----------------------------------------------------------------
    @property
    def passed(self) -> bool:
        return not any(v.severity == "error" for v in self.violations)

    @property
    def errors(self) -> List[ValidationViolation]:
        return [v for v in self.violations if v.severity == "error"]

    @property
    def warnings(self) -> List[ValidationViolation]:
        return [v for v in self.violations if v.severity == "warning"]

    def add(
        self,
        column: str,
        check: str,
        severity: str,
        message: str,
        actual: Any = None,
        expected: Any = None,
    ) -> None:
        self.violations.append(
            ValidationViolation(column, check, severity, message, actual, expected)
        )

    # -----------------------------------------------------------------
    def display(self) -> None:
        """Print a Rich table summarising all violations."""
        try:
            from rich.console import Console
            from rich.table import Table
            from rich import box

            console = Console()

            if not self.violations:
                console.print("[bold green]✅  All validation checks passed![/bold green]")
                return

            status = "[bold green]PASSED[/bold green]" if self.passed else "[bold red]FAILED[/bold red]"
            console.print(f"\n[bold]Validation Result:[/bold] {status}")
            console.print(
                f"  Errors: [red]{len(self.errors)}[/red]  "
                f"Warnings: [yellow]{len(self.warnings)}[/yellow]\n"
            )

            tbl = Table(box=box.SIMPLE_HEAD, show_lines=False)
            tbl.add_column("Column", style="cyan", no_wrap=True)
            tbl.add_column("Check", style="white")
            tbl.add_column("Severity", justify="center")
            tbl.add_column("Message", style="dim")
            tbl.add_column("Actual", style="magenta")
            tbl.add_column("Expected", style="green")

            for v in self.violations:
                sev_style = "[red]ERROR[/red]" if v.severity == "error" else "[yellow]WARN[/yellow]"
                tbl.add_row(
                    v.column,
                    v.check,
                    sev_style,
                    v.message,
                    str(v.actual) if v.actual is not None else "—",
                    str(v.expected) if v.expected is not None else "—",
                )

            console.print(tbl)

        except ImportError:
            # Fallback plain-text output
            print(f"Validation {'PASSED' if self.passed else 'FAILED'}")
            for v in self.violations:
                print(f"  [{v.severity.upper()}] {v.column} / {v.check}: {v.message}")

    def to_dict(self) -> Dict[str, Any]:
        return {
            "passed": self.passed,
            "error_count": len(self.errors),
            "warning_count": len(self.warnings),
            "violations": [v.to_dict() for v in self.violations],
        }


# ---------------------------------------------------------------------------
# DataValidator
# ---------------------------------------------------------------------------

class DataValidator:
    """Validates a Polars DataFrame against a DataSchema."""

    def __init__(self, schema: DataSchema) -> None:
        self.schema = schema

    # -----------------------------------------------------------------
    def validate(self, df: pl.DataFrame) -> ValidationResult:
        result = ValidationResult()
        schema_col_names = {cs.name for cs in self.schema.columns}
        df_col_names = set(df.columns)

        # --- dataset-level: missing columns ---
        if self.schema.require_all_columns:
            for name in schema_col_names - df_col_names:
                result.add(
                    column=name,
                    check="column_presence",
                    severity="error",
                    message=f"Required column '{name}' is missing from the DataFrame.",
                )

        # --- dataset-level: extra columns ---
        if not self.schema.allow_extra_columns:
            for name in df_col_names - schema_col_names:
                result.add(
                    column=name,
                    check="extra_column",
                    severity="warning",
                    message=f"Column '{name}' is not in the schema.",
                )

        # --- per-column checks ---
        for col_schema in self.schema.columns:
            if col_schema.name not in df_col_names:
                continue  # already flagged above
            series = df[col_schema.name]
            self._check_column(series, col_schema, result)

        return result

    # -----------------------------------------------------------------
    def _check_column(
        self,
        series: pl.Series,
        cs: ColumnSchema,
        result: ValidationResult,
    ) -> None:
        col = cs.name
        n = len(series)

        # dtype
        if cs.dtype is not None:
            actual_dtype = str(series.dtype)
            if actual_dtype != cs.dtype:
                result.add(
                    column=col,
                    check="dtype",
                    severity="error",
                    message=f"Dtype mismatch.",
                    actual=actual_dtype,
                    expected=cs.dtype,
                )

        null_count = series.null_count()
        null_pct = (null_count / n * 100) if n > 0 else 0.0

        # nullable
        if not cs.nullable and null_count > 0:
            result.add(
                column=col,
                check="nullable",
                severity="error",
                message=f"Column must not contain nulls but has {null_count} null(s).",
                actual=null_count,
                expected=0,
            )

        # max_missing_pct
        if null_pct > cs.max_missing_pct:
            result.add(
                column=col,
                check="max_missing_pct",
                severity="warning",
                message=f"Missing % ({null_pct:.1f}%) exceeds limit.",
                actual=f"{null_pct:.1f}%",
                expected=f"≤{cs.max_missing_pct:.1f}%",
            )

        # unique
        if cs.unique:
            non_null = series.drop_nulls()
            n_unique = non_null.n_unique()
            if n_unique < len(non_null):
                result.add(
                    column=col,
                    check="unique",
                    severity="error",
                    message=f"Values must be unique but {len(non_null) - n_unique} duplicate(s) found.",
                    actual=n_unique,
                    expected=len(non_null),
                )

        # numeric range checks
        numeric_dtypes = {
            pl.Int8, pl.Int16, pl.Int32, pl.Int64,
            pl.UInt8, pl.UInt16, pl.UInt32, pl.UInt64,
            pl.Float32, pl.Float64,
        }
        if series.dtype in numeric_dtypes:
            non_null = series.drop_nulls()
            if len(non_null) > 0:
                if cs.min_value is not None:
                    actual_min = non_null.min()
                    if actual_min < cs.min_value:
                        result.add(
                            column=col,
                            check="min_value",
                            severity="error",
                            message=f"Minimum value {actual_min} is below threshold.",
                            actual=actual_min,
                            expected=f"≥{cs.min_value}",
                        )
                if cs.max_value is not None:
                    actual_max = non_null.max()
                    if actual_max > cs.max_value:
                        result.add(
                            column=col,
                            check="max_value",
                            severity="error",
                            message=f"Maximum value {actual_max} exceeds threshold.",
                            actual=actual_max,
                            expected=f"≤{cs.max_value}",
                        )

        # allowed_values (whitelist)
        if cs.allowed_values is not None:
            non_null = series.drop_nulls()
            allowed = set(cs.allowed_values)
            actual_vals = set(non_null.to_list())
            bad = actual_vals - allowed
            if bad:
                result.add(
                    column=col,
                    check="allowed_values",
                    severity="error",
                    message=f"{len(bad)} disallowed value(s) found.",
                    actual=sorted(str(v) for v in list(bad)[:5]),
                    expected=f"one of {sorted(str(v) for v in list(allowed)[:10])}",
                )


# ---------------------------------------------------------------------------
# Convenience functions
# ---------------------------------------------------------------------------

def validate_data(
    df: pl.DataFrame,
    schema: Union[DataSchema, Dict[str, Any], str, Path],
) -> ValidationResult:
    """
    Validate *df* against *schema*.

    *schema* may be:
      - a DataSchema instance
      - a dict (parsed inline)
      - a file path string/Path (JSON or YAML)
    """
    if isinstance(schema, (str, Path)):
        schema = DataSchema.load(schema)
    elif isinstance(schema, dict):
        schema = DataSchema.from_dict(schema)

    return DataValidator(schema).validate(df)


def infer_schema(df: pl.DataFrame) -> DataSchema:
    """
    Build a DataSchema from an existing DataFrame (useful as a starting point).
    Infers dtype, nullable, and missing-% from the data.
    """
    cols: List[ColumnSchema] = []
    n = len(df)
    for col_name in df.columns:
        series = df[col_name]
        null_count = series.null_count()
        null_pct = (null_count / n * 100) if n > 0 else 0.0
        cols.append(
            ColumnSchema(
                name=col_name,
                dtype=str(series.dtype),
                nullable=null_count > 0,
                max_missing_pct=round(null_pct + 5.0, 1),  # +5% tolerance
                unique=False,
            )
        )
    return DataSchema(columns=cols)
