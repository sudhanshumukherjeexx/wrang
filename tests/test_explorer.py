"""Tests for ride/core/explorer.py — DataExplorer."""

import polars as pl
import pytest

from wrang.core.explorer import DataExplorer, explore_data


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------

@pytest.fixture
def numeric_df():
    """Two perfectly correlated columns plus an independent one."""
    import numpy as np
    rng = np.random.default_rng(42)
    x = rng.standard_normal(50).tolist()
    y = [v * 2.0 for v in x]          # perfect linear correlation with x
    z = rng.standard_normal(50).tolist()
    return pl.DataFrame({"x": x, "y": y, "z": z})


@pytest.fixture
def outlier_df():
    """One column with a clear outlier at 100.0."""
    return pl.DataFrame({
        "normal": [1.0, 2.0, 3.0, 2.5, 1.5, 2.0, 3.0, 1.0, 100.0],
        "clean":  [10.0, 11.0, 12.0, 10.5, 11.5, 12.0, 10.0, 11.0, 12.0],
    })


@pytest.fixture
def categorical_df():
    return pl.DataFrame({
        "status": ["active", "inactive", "active", "active", "pending", "inactive"],
        "value":  [100.0, 200.0, 150.0, 300.0, 250.0, 180.0],
    })


# ---------------------------------------------------------------------------
# Constructor and convenience function
# ---------------------------------------------------------------------------

def test_constructor(numeric_df):
    explorer = DataExplorer(numeric_df)
    assert explorer is not None


def test_explore_data_convenience(numeric_df):
    assert isinstance(explore_data(numeric_df), DataExplorer)


# ---------------------------------------------------------------------------
# analyze_correlations
# ---------------------------------------------------------------------------

def test_analyze_correlations_returns_dict(numeric_df):
    result = DataExplorer(numeric_df).analyze_correlations()
    assert isinstance(result, dict)


def test_analyze_correlations_has_correlations_key(numeric_df):
    result = DataExplorer(numeric_df).analyze_correlations()
    assert "correlations" in result


def test_analyze_correlations_pairs_are_dicts(numeric_df):
    pairs = DataExplorer(numeric_df).analyze_correlations()["correlations"]
    assert isinstance(pairs, list)
    if pairs:
        assert "correlation" in pairs[0]


def test_analyze_correlations_finds_strong_pair(numeric_df):
    """x and y are perfectly correlated — should appear in results."""
    pairs = DataExplorer(numeric_df).analyze_correlations(
        min_correlation=0.0
    )["correlations"]
    corr_values = [abs(p["correlation"]) for p in pairs]
    assert any(v > 0.99 for v in corr_values)


def test_analyze_correlations_spearman(numeric_df):
    result = DataExplorer(numeric_df).analyze_correlations(method="spearman")
    assert "correlations" in result


def test_analyze_correlations_min_threshold_filters(numeric_df):
    high = DataExplorer(numeric_df).analyze_correlations(min_correlation=0.99)
    low  = DataExplorer(numeric_df).analyze_correlations(min_correlation=0.0)
    assert len(high["correlations"]) <= len(low["correlations"])


# ---------------------------------------------------------------------------
# analyze_distributions
# ---------------------------------------------------------------------------

def test_analyze_distributions_returns_dict(numeric_df):
    result = DataExplorer(numeric_df).analyze_distributions()
    assert isinstance(result, dict)
    assert "distributions" in result


def test_analyze_distributions_covers_all_columns(numeric_df):
    inner = DataExplorer(numeric_df).analyze_distributions()["distributions"]
    for col in ["x", "y", "z"]:
        assert col in inner


def test_analyze_distributions_has_skewness(numeric_df):
    inner = DataExplorer(numeric_df).analyze_distributions()["distributions"]
    col_stats = inner["x"]
    assert "skewness" in col_stats


def test_analyze_distributions_has_kurtosis(numeric_df):
    inner = DataExplorer(numeric_df).analyze_distributions()["distributions"]
    assert "kurtosis" in inner["x"]


def test_analyze_distributions_subset(numeric_df):
    inner = DataExplorer(numeric_df).analyze_distributions(columns=["x"])["distributions"]
    assert "x" in inner
    assert "y" not in inner


# ---------------------------------------------------------------------------
# detect_outliers
# ---------------------------------------------------------------------------

def test_detect_outliers_returns_dict(outlier_df):
    result = DataExplorer(outlier_df).detect_outliers()
    assert isinstance(result, dict)
    assert "outliers" in result


def test_detect_outliers_finds_outlier_in_normal_col(outlier_df):
    result = DataExplorer(outlier_df).detect_outliers(method="iqr")
    outlier_info = result["outliers"]["normal"]
    assert outlier_info["outlier_count"] >= 1


def test_detect_outliers_clean_col_zero_count(outlier_df):
    result = DataExplorer(outlier_df).detect_outliers(method="iqr")
    assert result["outliers"]["clean"]["outlier_count"] == 0


def test_detect_outliers_zscore_method(outlier_df):
    result = DataExplorer(outlier_df).detect_outliers(method="zscore")
    assert "outliers" in result


def test_detect_outliers_column_subset(outlier_df):
    result = DataExplorer(outlier_df).detect_outliers(columns=["normal"])
    assert "normal" in result["outliers"]
    assert "clean" not in result["outliers"]


# ---------------------------------------------------------------------------
# analyze_categorical_variables
# ---------------------------------------------------------------------------

def test_analyze_categorical_returns_dict(categorical_df):
    result = DataExplorer(categorical_df).analyze_categorical_variables()
    assert isinstance(result, dict)
    assert "categorical_analysis" in result


def test_analyze_categorical_has_status(categorical_df):
    inner = DataExplorer(categorical_df).analyze_categorical_variables()["categorical_analysis"]
    assert "status" in inner


def test_analyze_categorical_skips_numeric(categorical_df):
    inner = DataExplorer(categorical_df).analyze_categorical_variables()["categorical_analysis"]
    assert "value" not in inner


# ---------------------------------------------------------------------------
# test_normality
# ---------------------------------------------------------------------------

def test_normality_returns_dict(numeric_df):
    result = DataExplorer(numeric_df).test_normality()
    assert isinstance(result, dict)
    assert "normality_tests" in result


def test_normality_has_p_value(numeric_df):
    inner = DataExplorer(numeric_df).test_normality()["normality_tests"]
    first_col = inner[list(inner.keys())[0]]
    assert "p_value" in first_col


def test_normality_has_is_normal(numeric_df):
    inner = DataExplorer(numeric_df).test_normality()["normality_tests"]
    first_col = inner[list(inner.keys())[0]]
    assert "is_normal" in first_col


def test_normality_column_subset(numeric_df):
    inner = DataExplorer(numeric_df).test_normality(columns=["x"])["normality_tests"]
    assert "x" in inner
    assert "y" not in inner


def test_normality_alpha_respected(numeric_df):
    result = DataExplorer(numeric_df).test_normality(alpha=0.01)
    assert result["alpha"] == pytest.approx(0.01)


# ---------------------------------------------------------------------------
# Smoke tests — display/plot methods
# ---------------------------------------------------------------------------

def test_plot_histogram_no_crash(numeric_df):
    DataExplorer(numeric_df).plot_histogram("x")


def test_plot_correlation_heatmap_no_crash(numeric_df):
    DataExplorer(numeric_df).plot_correlation_heatmap()


def test_plot_scatter_no_crash(numeric_df):
    DataExplorer(numeric_df).plot_scatter("x", "y")
