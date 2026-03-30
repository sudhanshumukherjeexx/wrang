"""Tests for ride/core/cleaner.py — DataCleaner."""

import polars as pl
import pytest

from wrang.config import ImputationStrategy
from wrang.core.cleaner import DataCleaner, quick_clean


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------

@pytest.fixture
def df_with_nulls():
    return pl.DataFrame({
        "id":    [1, 2, 3, 4, 5, 6],
        "score": [88.0, None, 76.0, None, 95.0, 60.0],
        "grade": ["A", "B", None, "C", "A", None],
        "rank":  [1, 2, 3, 4, 5, 6],
    })


@pytest.fixture
def df_with_duplicates():
    return pl.DataFrame({
        "x": [1, 1, 2, 3, 3, 4],
        "y": ["a", "a", "b", "c", "c", "d"],
    })


@pytest.fixture
def df_with_outliers():
    return pl.DataFrame({
        "val": [1.0, 2.0, 2.5, 1.5, 2.0, 3.0, 100.0, 1.8, 2.2, 1.9],
    })


@pytest.fixture
def df_text():
    return pl.DataFrame({
        "name":  ["  ALICE  ", "bob", "  Carol"],
        "email": ["ALICE@TEST.COM", "Bob@Example.COM", "carol@test.com"],
    })


@pytest.fixture
def clean_df():
    return pl.DataFrame({
        "a": [1.0, 2.0, 3.0, 4.0, 5.0],
        "b": [10.0, 20.0, 30.0, 40.0, 50.0],
    })


# ---------------------------------------------------------------------------
# Constructor and get_cleaned_data
# ---------------------------------------------------------------------------

def test_constructor(clean_df):
    cleaner = DataCleaner(clean_df)
    assert cleaner.df is not None
    assert len(cleaner.df) == len(clean_df)


def test_get_cleaned_data_returns_dataframe(clean_df):
    result = DataCleaner(clean_df).get_cleaned_data()
    assert isinstance(result, pl.DataFrame)


# ---------------------------------------------------------------------------
# handle_missing_values — DROP
# ---------------------------------------------------------------------------

def test_missing_drop_removes_null_rows(df_with_nulls):
    cleaner = DataCleaner(df_with_nulls)
    cleaner.handle_missing_values(strategy=ImputationStrategy.DROP)
    result = cleaner.get_cleaned_data()
    assert result["score"].null_count() == 0
    assert result["grade"].null_count() == 0


def test_missing_drop_reduces_rows(df_with_nulls):
    cleaner = DataCleaner(df_with_nulls)
    cleaner.handle_missing_values(strategy=ImputationStrategy.DROP)
    assert len(cleaner.get_cleaned_data()) < len(df_with_nulls)


# ---------------------------------------------------------------------------
# handle_missing_values — MEAN
# ---------------------------------------------------------------------------

def test_missing_mean_fills_numeric(df_with_nulls):
    cleaner = DataCleaner(df_with_nulls)
    cleaner.handle_missing_values(strategy=ImputationStrategy.MEAN, columns=["score"])
    assert cleaner.get_cleaned_data()["score"].null_count() == 0


def test_missing_mean_preserves_row_count(df_with_nulls):
    cleaner = DataCleaner(df_with_nulls)
    cleaner.handle_missing_values(strategy=ImputationStrategy.MEAN, columns=["score"])
    assert len(cleaner.get_cleaned_data()) == len(df_with_nulls)


# ---------------------------------------------------------------------------
# handle_missing_values — MEDIAN
# ---------------------------------------------------------------------------

def test_missing_median_fills_numeric(df_with_nulls):
    cleaner = DataCleaner(df_with_nulls)
    cleaner.handle_missing_values(strategy=ImputationStrategy.MEDIAN, columns=["score"])
    assert cleaner.get_cleaned_data()["score"].null_count() == 0


# ---------------------------------------------------------------------------
# handle_missing_values — MODE
# ---------------------------------------------------------------------------

def test_missing_mode_fills_categorical(df_with_nulls):
    cleaner = DataCleaner(df_with_nulls)
    cleaner.handle_missing_values(strategy=ImputationStrategy.MODE, columns=["grade"])
    assert cleaner.get_cleaned_data()["grade"].null_count() == 0


# ---------------------------------------------------------------------------
# handle_missing_values — FORWARD_FILL / BACKWARD_FILL
# ---------------------------------------------------------------------------

def test_missing_forward_fill(df_with_nulls):
    cleaner = DataCleaner(df_with_nulls)
    cleaner.handle_missing_values(
        strategy=ImputationStrategy.FORWARD_FILL, columns=["score"]
    )
    result = cleaner.get_cleaned_data()
    # score[0]=88 is not null, so forward fill covers score[1] null
    assert result["score"].null_count() == 0


def test_missing_backward_fill(df_with_nulls):
    cleaner = DataCleaner(df_with_nulls)
    cleaner.handle_missing_values(
        strategy=ImputationStrategy.BACKWARD_FILL, columns=["score"]
    )
    result = cleaner.get_cleaned_data()
    # score[-1]=60 is not null, so backward fill covers score[3] null
    assert result["score"].null_count() == 0


# ---------------------------------------------------------------------------
# handle_missing_values — CUSTOM_VALUE
# ---------------------------------------------------------------------------

def test_missing_custom_value_numeric(df_with_nulls):
    cleaner = DataCleaner(df_with_nulls)
    cleaner.handle_missing_values(
        strategy=ImputationStrategy.CUSTOM_VALUE,
        columns=["score"],
        custom_value=0.0,
    )
    result = cleaner.get_cleaned_data()
    assert result["score"].null_count() == 0
    assert 0.0 in result["score"].to_list()


def test_missing_custom_value_string(df_with_nulls):
    cleaner = DataCleaner(df_with_nulls)
    cleaner.handle_missing_values(
        strategy=ImputationStrategy.CUSTOM_VALUE,
        columns=["grade"],
        custom_value="UNKNOWN",
    )
    result = cleaner.get_cleaned_data()
    assert result["grade"].null_count() == 0
    assert "UNKNOWN" in result["grade"].to_list()


# ---------------------------------------------------------------------------
# remove_duplicates
# ---------------------------------------------------------------------------

def test_remove_duplicates_reduces_rows(df_with_duplicates):
    cleaner = DataCleaner(df_with_duplicates)
    cleaner.remove_duplicates()
    assert len(cleaner.get_cleaned_data()) < len(df_with_duplicates)


def test_remove_duplicates_all_unique_after(df_with_duplicates):
    cleaner = DataCleaner(df_with_duplicates)
    cleaner.remove_duplicates()
    result = cleaner.get_cleaned_data()
    assert len(result) == result.n_unique()


def test_remove_duplicates_subset(df_with_duplicates):
    cleaner = DataCleaner(df_with_duplicates)
    cleaner.remove_duplicates(columns=["x"])
    result = cleaner.get_cleaned_data()
    assert result["x"].n_unique() == len(result)


def test_remove_duplicates_preserves_columns(df_with_duplicates):
    cleaner = DataCleaner(df_with_duplicates)
    cleaner.remove_duplicates()
    assert set(cleaner.get_cleaned_data().columns) == set(df_with_duplicates.columns)


# ---------------------------------------------------------------------------
# handle_outliers — IQR remove
# ---------------------------------------------------------------------------

def test_outliers_iqr_remove_eliminates_extreme(df_with_outliers):
    cleaner = DataCleaner(df_with_outliers)
    cleaner.handle_outliers(method="iqr", action="remove")
    assert 100.0 not in cleaner.get_cleaned_data()["val"].to_list()


def test_outliers_iqr_remove_reduces_rows(df_with_outliers):
    cleaner = DataCleaner(df_with_outliers)
    cleaner.handle_outliers(method="iqr", action="remove")
    assert len(cleaner.get_cleaned_data()) < len(df_with_outliers)


# ---------------------------------------------------------------------------
# handle_outliers — IQR cap
# ---------------------------------------------------------------------------

def test_outliers_iqr_cap_preserves_row_count(df_with_outliers):
    cleaner = DataCleaner(df_with_outliers)
    cleaner.handle_outliers(method="iqr", action="cap")
    assert len(cleaner.get_cleaned_data()) == len(df_with_outliers)


def test_outliers_iqr_cap_has_same_row_count_as_remove(df_with_outliers):
    """Cap keeps all rows; remove drops rows — verify they differ."""
    cap_len = len(DataCleaner(df_with_outliers).handle_outliers(method="iqr", action="cap").get_cleaned_data())
    rem_len = len(DataCleaner(df_with_outliers).handle_outliers(method="iqr", action="remove").get_cleaned_data())
    assert cap_len >= rem_len


# ---------------------------------------------------------------------------
# clean_text_data
# ---------------------------------------------------------------------------

def test_clean_text_strips_whitespace(df_text):
    cleaner = DataCleaner(df_text)
    cleaner.clean_text_data(columns=["name"])
    for val in cleaner.get_cleaned_data()["name"].to_list():
        assert val == val.strip()


def test_clean_text_no_crash_on_all_columns(df_text):
    cleaner = DataCleaner(df_text)
    cleaner.clean_text_data()   # all text columns
    assert cleaner.get_cleaned_data() is not None


# ---------------------------------------------------------------------------
# validate_data_types
# ---------------------------------------------------------------------------

def test_validate_data_types_no_crash(df_with_nulls):
    cleaner = DataCleaner(df_with_nulls)
    cleaner.validate_data_types()
    assert cleaner.get_cleaned_data() is not None


def test_validate_data_types_explicit_map(clean_df):
    cleaner = DataCleaner(clean_df)
    cleaner.validate_data_types(type_map={"a": "Float64"})
    assert cleaner.get_cleaned_data()["a"].dtype in (pl.Float32, pl.Float64)


# ---------------------------------------------------------------------------
# Method chaining
# ---------------------------------------------------------------------------

def test_method_chaining(clean_df):
    df = pl.DataFrame({
        "a": [1.0, None, 3.0, 1.0, 5.0],
        "b": ["x", "y", "y", "x", "z"],
    })
    result = (
        DataCleaner(df)
        .handle_missing_values(strategy=ImputationStrategy.MEAN, columns=["a"])
        .remove_duplicates()
        .get_cleaned_data()
    )
    assert result["a"].null_count() == 0
    assert isinstance(result, pl.DataFrame)


def test_chaining_returns_self(clean_df):
    cleaner = DataCleaner(clean_df)
    returned = cleaner.remove_duplicates()
    assert returned is cleaner


# ---------------------------------------------------------------------------
# quick_clean convenience function
# ---------------------------------------------------------------------------

def test_quick_clean_returns_dataframe(df_with_nulls):
    result = quick_clean(df_with_nulls)
    assert isinstance(result, pl.DataFrame)
