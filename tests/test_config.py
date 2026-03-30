"""Tests for ride/config.py — RideConfig and config management functions."""

from pathlib import Path

import pytest

from wrang.config import (
    RideConfig,
    FileFormat,
    ImputationStrategy,
    ScalingMethod,
    EncodingMethod,
    get_config,
    update_config,
    reset_config,
)


# ---------------------------------------------------------------------------
# get_config
# ---------------------------------------------------------------------------

def test_get_config_returns_rideconfig():
    assert isinstance(get_config(), RideConfig)


def test_config_default_random_state():
    assert get_config().random_state == 42


def test_config_default_chunk_size_positive():
    assert get_config().chunk_size > 0


def test_config_default_missing_threshold_in_range():
    t = get_config().missing_value_threshold
    assert 0.0 < t <= 1.0


def test_config_default_outlier_method():
    assert get_config().outlier_method in ("iqr", "zscore", "modified_zscore")


def test_config_supported_formats_non_empty():
    assert len(get_config().supported_formats) > 0


# ---------------------------------------------------------------------------
# update_config
# ---------------------------------------------------------------------------

def test_update_config_changes_value():
    update_config(random_state=99)
    assert get_config().random_state == 99
    reset_config()


def test_update_config_returns_rideconfig():
    result = update_config(chunk_size=500)
    assert isinstance(result, RideConfig)
    reset_config()


def test_update_config_does_not_affect_unrelated_fields():
    original_threshold = get_config().missing_value_threshold
    update_config(chunk_size=1234)
    assert get_config().missing_value_threshold == original_threshold
    reset_config()


# ---------------------------------------------------------------------------
# reset_config
# ---------------------------------------------------------------------------

def test_reset_config_restores_default_random_state():
    update_config(random_state=999)
    reset_config()
    assert get_config().random_state == 42


def test_reset_config_returns_rideconfig():
    assert isinstance(reset_config(), RideConfig)


# ---------------------------------------------------------------------------
# FileFormat enum
# ---------------------------------------------------------------------------

def test_file_format_csv():
    assert hasattr(FileFormat, "CSV")


def test_file_format_excel():
    assert hasattr(FileFormat, "EXCEL")


def test_file_format_parquet():
    assert hasattr(FileFormat, "PARQUET")


def test_file_format_json():
    assert hasattr(FileFormat, "JSON")


# ---------------------------------------------------------------------------
# ImputationStrategy enum
# ---------------------------------------------------------------------------

def test_imputation_drop():
    assert hasattr(ImputationStrategy, "DROP")


def test_imputation_mean():
    assert hasattr(ImputationStrategy, "MEAN")


def test_imputation_median():
    assert hasattr(ImputationStrategy, "MEDIAN")


def test_imputation_mode():
    assert hasattr(ImputationStrategy, "MODE")


def test_imputation_knn():
    assert hasattr(ImputationStrategy, "KNN")


def test_imputation_forward_fill():
    assert hasattr(ImputationStrategy, "FORWARD_FILL")


def test_imputation_backward_fill():
    assert hasattr(ImputationStrategy, "BACKWARD_FILL")


def test_imputation_custom_value():
    assert hasattr(ImputationStrategy, "CUSTOM_VALUE")


def test_imputation_distribution():
    assert hasattr(ImputationStrategy, "DISTRIBUTION")


# ---------------------------------------------------------------------------
# ScalingMethod enum
# ---------------------------------------------------------------------------

def test_scaling_standard():
    assert hasattr(ScalingMethod, "STANDARD")


def test_scaling_minmax():
    assert hasattr(ScalingMethod, "MINMAX")


def test_scaling_robust():
    assert hasattr(ScalingMethod, "ROBUST")


def test_scaling_maxabs():
    assert hasattr(ScalingMethod, "MAXABS")


def test_scaling_quantile_uniform():
    assert hasattr(ScalingMethod, "QUANTILE_UNIFORM")


def test_scaling_quantile_normal():
    assert hasattr(ScalingMethod, "QUANTILE_NORMAL")


# ---------------------------------------------------------------------------
# EncodingMethod enum
# ---------------------------------------------------------------------------

def test_encoding_label():
    assert hasattr(EncodingMethod, "LABEL")


def test_encoding_onehot():
    assert hasattr(EncodingMethod, "ONEHOT")


def test_encoding_ordinal():
    assert hasattr(EncodingMethod, "ORDINAL")


def test_encoding_target():
    assert hasattr(EncodingMethod, "TARGET")


# ---------------------------------------------------------------------------
# get_file_config (requires Path object)
# ---------------------------------------------------------------------------

def test_get_file_config_csv():
    result = get_config().get_file_config(Path("data.csv"))
    assert isinstance(result, dict)


def test_get_file_config_excel():
    result = get_config().get_file_config(Path("data.xlsx"))
    assert isinstance(result, dict)


def test_get_file_config_parquet():
    result = get_config().get_file_config(Path("data.parquet"))
    assert isinstance(result, dict)
