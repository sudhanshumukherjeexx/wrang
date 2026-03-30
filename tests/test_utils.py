"""Tests for ride/utils/constants.py helpers."""

import polars as pl
import pytest

from wrang.utils.constants import (
    is_numeric,
    is_categorical,
    is_boolean,
    numeric_columns,
    categorical_columns,
    boolean_columns,
    datetime_columns,
    classify_correlation,
    interpret_skewness,
    format_memory,
    format_file_size,
)


@pytest.fixture
def mixed_df():
    return pl.DataFrame(
        {
            "int_col": [1, 2, 3],
            "float_col": [1.1, 2.2, 3.3],
            "str_col": ["a", "b", "c"],
            "bool_col": [True, False, True],
        }
    )


# ---------------------------------------------------------------------------
# Series-level helpers
# ---------------------------------------------------------------------------

def test_is_numeric_int(mixed_df):
    assert is_numeric(mixed_df["int_col"]) is True


def test_is_numeric_float(mixed_df):
    assert is_numeric(mixed_df["float_col"]) is True


def test_is_numeric_str(mixed_df):
    assert is_numeric(mixed_df["str_col"]) is False


def test_is_categorical_str(mixed_df):
    assert is_categorical(mixed_df["str_col"]) is True


def test_is_categorical_int(mixed_df):
    assert is_categorical(mixed_df["int_col"]) is False


def test_is_boolean(mixed_df):
    assert is_boolean(mixed_df["bool_col"]) is True


def test_is_boolean_int(mixed_df):
    assert is_boolean(mixed_df["int_col"]) is False


# ---------------------------------------------------------------------------
# DataFrame-level helpers
# ---------------------------------------------------------------------------

def test_numeric_columns(mixed_df):
    cols = numeric_columns(mixed_df)
    assert "int_col" in cols
    assert "float_col" in cols
    assert "str_col" not in cols


def test_categorical_columns(mixed_df):
    cols = categorical_columns(mixed_df)
    assert "str_col" in cols
    assert "int_col" not in cols


def test_boolean_columns(mixed_df):
    cols = boolean_columns(mixed_df)
    assert "bool_col" in cols
    assert "int_col" not in cols


def test_datetime_columns_empty(mixed_df):
    # mixed_df has no datetime columns
    cols = datetime_columns(mixed_df)
    assert cols == []


def test_datetime_columns_with_date():
    import datetime
    df = pl.DataFrame({"dt": [datetime.date(2024, 1, 1), datetime.date(2024, 1, 2)]})
    cols = datetime_columns(df)
    assert "dt" in cols


# ---------------------------------------------------------------------------
# Utility functions
# ---------------------------------------------------------------------------

def test_classify_correlation_very_strong():
    assert classify_correlation(0.95) == "Very Strong"


def test_classify_correlation_strong():
    assert classify_correlation(0.75) == "Strong"


def test_classify_correlation_moderate():
    assert classify_correlation(0.55) == "Moderate"


def test_classify_correlation_weak():
    assert classify_correlation(0.35) == "Weak"


def test_classify_correlation_very_weak():
    assert classify_correlation(0.1) == "Very Weak"


def test_interpret_skewness_symmetric():
    assert "symmetric" in interpret_skewness(0.2)


def test_interpret_skewness_highly():
    assert "highly" in interpret_skewness(1.5)


def test_format_memory_kb():
    assert "KB" in format_memory(0.5)


def test_format_memory_mb():
    assert "MB" in format_memory(10.0)


def test_format_memory_gb():
    assert "GB" in format_memory(2048.0)


def test_format_file_size_bytes():
    assert "B" in format_file_size(500)


def test_format_file_size_mb():
    result = format_file_size(1_500_000)
    assert "MB" in result or "KB" in result  # depends on rounding path
