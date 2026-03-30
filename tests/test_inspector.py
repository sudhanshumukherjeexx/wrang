"""Tests for ride/core/inspector.py — DataInspector."""

import polars as pl
import pytest

from wrang.core.inspector import DataInspector, inspect_data


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------

@pytest.fixture
def numeric_df():
    return pl.DataFrame({
        "a": [1, 2, 3, 4, 5],
        "b": [10.0, 20.0, 30.0, 40.0, 50.0],
        "c": [100, 200, 300, 400, 500],
    })


@pytest.fixture
def mixed_df():
    return pl.DataFrame({
        "id":     [1, 2, 3, 4, 5],
        "name":   ["Alice", "Bob", None, "Dave", "Eve"],
        "score":  [88.5, None, 76.3, 55.0, 95.1],
        "active": [True, True, False, None, True],
    })


@pytest.fixture
def duplicate_df():
    return pl.DataFrame({
        "x": [1, 1, 2, 3, 3],
        "y": ["a", "a", "b", "c", "c"],
    })


@pytest.fixture
def constant_df():
    return pl.DataFrame({
        "val":      [1, 2, 3],
        "constant": [42, 42, 42],
    })


@pytest.fixture
def high_missing_df():
    return pl.DataFrame({
        "ok":     [1, 2, 3, 4, 5],
        "sparse": [None, None, None, 1, None],
    })


# ---------------------------------------------------------------------------
# Constructor and convenience function
# ---------------------------------------------------------------------------

def test_constructor(numeric_df):
    inspector = DataInspector(numeric_df)
    assert inspector is not None


def test_inspect_data_convenience(numeric_df):
    inspector = inspect_data(numeric_df)
    assert isinstance(inspector, DataInspector)


# ---------------------------------------------------------------------------
# get_basic_info
# ---------------------------------------------------------------------------

def test_basic_info_n_rows(numeric_df):
    info = DataInspector(numeric_df).get_basic_info()
    assert info["n_rows"] == 5


def test_basic_info_n_columns(numeric_df):
    info = DataInspector(numeric_df).get_basic_info()
    assert info["n_columns"] == 3


def test_basic_info_has_required_keys(mixed_df):
    info = DataInspector(mixed_df).get_basic_info()
    for key in ("n_rows", "n_columns", "missing_values_total", "duplicate_rows"):
        assert key in info, f"Expected key '{key}' in basic_info"


def test_basic_info_missing_count(mixed_df):
    info = DataInspector(mixed_df).get_basic_info()
    # name:1 + score:1 + active:1 = 3 missing cells
    assert info["missing_values_total"] >= 1


def test_basic_info_duplicates(duplicate_df):
    info = DataInspector(duplicate_df).get_basic_info()
    assert info["duplicate_rows"] >= 1


def test_basic_info_zero_duplicates(numeric_df):
    info = DataInspector(numeric_df).get_basic_info()
    assert info["duplicate_rows"] == 0


# ---------------------------------------------------------------------------
# get_memory_usage
# ---------------------------------------------------------------------------

def test_memory_usage_returns_dict(numeric_df):
    result = DataInspector(numeric_df).get_memory_usage()
    assert isinstance(result, dict)


def test_memory_usage_non_empty(numeric_df):
    result = DataInspector(numeric_df).get_memory_usage()
    assert len(result) > 0


# ---------------------------------------------------------------------------
# get_column_profiles
# ---------------------------------------------------------------------------

def test_column_profiles_returns_dict(mixed_df):
    profiles = DataInspector(mixed_df).get_column_profiles()
    assert isinstance(profiles, dict)


def test_column_profiles_all_columns(mixed_df):
    profiles = DataInspector(mixed_df).get_column_profiles()
    for col in mixed_df.columns:
        assert col in profiles


def test_column_profiles_has_dtype(numeric_df):
    profiles = DataInspector(numeric_df).get_column_profiles()
    for col in numeric_df.columns:
        assert "dtype" in profiles[col]


def test_column_profiles_missing_pct_nonzero(mixed_df):
    profiles = DataInspector(mixed_df).get_column_profiles()
    name_pct = profiles["name"].get("missing_percentage", 0)
    assert name_pct > 0


# ---------------------------------------------------------------------------
# detect_potential_issues
# ---------------------------------------------------------------------------

def test_detect_issues_returns_list(numeric_df):
    issues = DataInspector(numeric_df).detect_potential_issues()
    assert isinstance(issues, list)


def test_detect_constant_column(constant_df):
    issues = DataInspector(constant_df).detect_potential_issues()
    types = [i.get("type", "").lower() for i in issues]
    assert any("constant" in t for t in types)


def test_detect_high_missing(high_missing_df):
    issues = DataInspector(high_missing_df).detect_potential_issues()
    types = [i.get("type", "").lower() for i in issues]
    assert any("missing" in t for t in types)


def test_detect_duplicates_in_issues(duplicate_df):
    issues = DataInspector(duplicate_df).detect_potential_issues()
    types = [i.get("type", "").lower() for i in issues]
    assert any("duplicate" in t for t in types)


def test_detect_issues_each_has_type(numeric_df):
    # Even on a clean df the method runs without crashing
    issues = DataInspector(numeric_df).detect_potential_issues()
    for issue in issues:
        assert "type" in issue


# ---------------------------------------------------------------------------
# Display methods (smoke tests — verify no exception)
# ---------------------------------------------------------------------------

def test_display_overview_no_crash(mixed_df):
    DataInspector(mixed_df).display_overview()


def test_display_data_quality_no_crash(mixed_df):
    DataInspector(mixed_df).display_data_quality()


def test_display_statistical_summary_no_crash(numeric_df):
    DataInspector(numeric_df).display_statistical_summary()


def test_display_column_summary_no_crash(mixed_df):
    DataInspector(mixed_df).display_column_summary()
