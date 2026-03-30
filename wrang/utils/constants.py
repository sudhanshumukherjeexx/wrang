#!/usr/bin/env python3
"""
ride/utils/constants.py
Shared constants and column-type helpers used across all core modules.
Centralises logic that was previously copy-pasted in inspector, explorer,
cleaner, and transformer.
"""

from typing import List
import polars as pl

# ---------------------------------------------------------------------------
# Polars dtype category sets
# ---------------------------------------------------------------------------

_NUMERIC_DTYPES = frozenset({
    pl.Int8, pl.Int16, pl.Int32, pl.Int64,
    pl.UInt8, pl.UInt16, pl.UInt32, pl.UInt64,
    pl.Float32, pl.Float64,
})

_CATEGORICAL_DTYPES = frozenset({
    pl.Utf8, pl.Categorical,
})

_DATETIME_DTYPES = frozenset({
    pl.Date, pl.Datetime, pl.Time, pl.Duration,
})

_BOOLEAN_DTYPES = frozenset({
    pl.Boolean,
})


# ---------------------------------------------------------------------------
# Column-classification helpers
# ---------------------------------------------------------------------------

def is_numeric(series: pl.Series) -> bool:
    """Return True if the series has a numeric dtype."""
    return series.dtype in _NUMERIC_DTYPES


def is_categorical(series: pl.Series) -> bool:
    """Return True if the series has a string/categorical dtype."""
    return series.dtype in _CATEGORICAL_DTYPES


def is_datetime(series: pl.Series) -> bool:
    """Return True if the series has a datetime-related dtype."""
    return series.dtype in _DATETIME_DTYPES


def is_boolean(series: pl.Series) -> bool:
    """Return True if the series is boolean."""
    return series.dtype in _BOOLEAN_DTYPES


# DataFrame-level helpers ---------------------------------------------------

def numeric_columns(df: pl.DataFrame) -> List[str]:
    """Return names of all numeric columns in *df*."""
    return [col for col in df.columns if is_numeric(df[col])]


def categorical_columns(df: pl.DataFrame) -> List[str]:
    """Return names of all string/categorical columns in *df*."""
    return [col for col in df.columns if is_categorical(df[col])]


def datetime_columns(df: pl.DataFrame) -> List[str]:
    """Return names of all datetime-related columns in *df*."""
    return [col for col in df.columns if is_datetime(df[col])]


def boolean_columns(df: pl.DataFrame) -> List[str]:
    """Return names of all boolean columns in *df*."""
    return [col for col in df.columns if is_boolean(df[col])]


# ---------------------------------------------------------------------------
# Shared thresholds / magic numbers
# ---------------------------------------------------------------------------

# Correlation strength classification
CORRELATION_THRESHOLDS = {
    "very_strong": 0.9,
    "strong": 0.7,
    "moderate": 0.5,
    "weak": 0.3,
}


def classify_correlation(abs_corr: float) -> str:
    """Return a human-readable label for a correlation magnitude."""
    if abs_corr >= CORRELATION_THRESHOLDS["very_strong"]:
        return "Very Strong"
    if abs_corr >= CORRELATION_THRESHOLDS["strong"]:
        return "Strong"
    if abs_corr >= CORRELATION_THRESHOLDS["moderate"]:
        return "Moderate"
    if abs_corr >= CORRELATION_THRESHOLDS["weak"]:
        return "Weak"
    return "Very Weak"


# Skewness interpretation
def interpret_skewness(skew: float) -> str:
    """Return a human-readable description of distribution skew."""
    abs_skew = abs(skew)
    if abs_skew > 1:
        return "highly skewed"
    if abs_skew > 0.5:
        return "moderately skewed"
    return "approximately symmetric"


# Memory formatting
def format_memory(memory_mb: float) -> str:
    """Format memory size in human-readable format."""
    if memory_mb < 1:
        return f"{memory_mb * 1024:.1f} KB"
    if memory_mb < 1024:
        return f"{memory_mb:.1f} MB"
    return f"{memory_mb / 1024:.1f} GB"


# File-size formatting
def format_file_size(size_bytes: int) -> str:
    """Format file size in human-readable format."""
    for unit in ('B', 'KB', 'MB', 'GB', 'TB'):
        if size_bytes < 1024:
            return f"{size_bytes:.1f} {unit}"
        size_bytes //= 1024
    return f"{size_bytes:.1f} PB"
