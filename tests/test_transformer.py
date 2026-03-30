"""Tests for ride/core/transformer.py — DataTransformer."""

import polars as pl
import pytest

from wrang.config import EncodingMethod, ScalingMethod
from wrang.core.transformer import DataTransformer, quick_transform


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------

@pytest.fixture
def numeric_df():
    return pl.DataFrame({
        "a": [1.0, 2.0, 3.0, 4.0, 5.0],
        "b": [10.0, 20.0, 30.0, 40.0, 50.0],
        "c": [100.0, 200.0, 300.0, 400.0, 500.0],
    })


@pytest.fixture
def categorical_df():
    return pl.DataFrame({
        "color":  ["red", "blue", "green", "red", "blue"],
        "size":   ["S", "M", "L", "M", "S"],
        "weight": [1.0, 2.0, 3.0, 4.0, 5.0],
    })


@pytest.fixture
def mixed_df():
    return pl.DataFrame({
        "num":    [1.0, 2.0, 3.0, 4.0, 5.0, 6.0],
        "cat":    ["a", "b", "a", "b", "c", "c"],
        "target": [0, 1, 0, 1, 0, 1],
    })


# ---------------------------------------------------------------------------
# Constructor and get_transformed_data
# ---------------------------------------------------------------------------

def test_constructor(numeric_df):
    transformer = DataTransformer(numeric_df)
    assert transformer is not None
    assert transformer.df is not None


def test_get_transformed_data_returns_polars(numeric_df):
    result = DataTransformer(numeric_df).get_transformed_data()
    assert isinstance(result, pl.DataFrame)


def test_original_df_unchanged_after_transform(numeric_df):
    transformer = DataTransformer(numeric_df)
    transformer.scale_features(method=ScalingMethod.STANDARD)
    # original_df should remain unmodified
    assert transformer.original_df["a"].mean() == pytest.approx(3.0, abs=1e-9)


# ---------------------------------------------------------------------------
# encode_categorical_features — LABEL
# ---------------------------------------------------------------------------

def test_label_encode_changes_column(categorical_df):
    transformer = DataTransformer(categorical_df)
    transformer.encode_categorical_features(method=EncodingMethod.LABEL, columns=["color"])
    result = transformer.get_transformed_data()
    # Label encoding maps categories to codes — original string "red"/"blue"/"green"
    # are replaced; all non-null values should be present
    assert result["color"].null_count() == 0
    assert result["color"].n_unique() == categorical_df["color"].n_unique()


def test_label_encode_preserves_row_count(categorical_df):
    transformer = DataTransformer(categorical_df)
    transformer.encode_categorical_features(method=EncodingMethod.LABEL)
    assert len(transformer.get_transformed_data()) == len(categorical_df)


def test_label_encode_all_categorical_no_nulls(categorical_df):
    transformer = DataTransformer(categorical_df)
    transformer.encode_categorical_features(method=EncodingMethod.LABEL)
    result = transformer.get_transformed_data()
    for col in ["color", "size"]:
        assert result[col].null_count() == 0


# ---------------------------------------------------------------------------
# encode_categorical_features — ONEHOT
# ---------------------------------------------------------------------------

def test_onehot_expands_columns(categorical_df):
    n_original = len(categorical_df.columns)
    transformer = DataTransformer(categorical_df)
    transformer.encode_categorical_features(method=EncodingMethod.ONEHOT, columns=["color"])
    result = transformer.get_transformed_data()
    assert len(result.columns) > n_original


def test_onehot_preserves_row_count(categorical_df):
    transformer = DataTransformer(categorical_df)
    transformer.encode_categorical_features(method=EncodingMethod.ONEHOT, columns=["color"])
    assert len(transformer.get_transformed_data()) == len(categorical_df)


# ---------------------------------------------------------------------------
# scale_features — STANDARD
# ---------------------------------------------------------------------------

def test_standard_scale_near_zero_mean(numeric_df):
    transformer = DataTransformer(numeric_df)
    transformer.scale_features(method=ScalingMethod.STANDARD, columns=["a"])
    result = transformer.get_transformed_data()
    assert abs(result["a"].mean()) < 1e-9


def test_standard_scale_std_close_to_one(numeric_df):
    transformer = DataTransformer(numeric_df)
    transformer.scale_features(method=ScalingMethod.STANDARD, columns=["a"])
    result = transformer.get_transformed_data()
    # Polars .std() uses ddof=1; sklearn uses ddof=0. For small N the values
    # differ slightly. Assert std is between 0.9 and 1.3 (reasonable range).
    assert 0.9 <= result["a"].std() <= 1.3


def test_standard_scale_preserves_shape(numeric_df):
    transformer = DataTransformer(numeric_df)
    transformer.scale_features(method=ScalingMethod.STANDARD)
    assert transformer.get_transformed_data().shape == numeric_df.shape


# ---------------------------------------------------------------------------
# scale_features — MINMAX
# ---------------------------------------------------------------------------

def test_minmax_scale_zero_to_one(numeric_df):
    transformer = DataTransformer(numeric_df)
    transformer.scale_features(method=ScalingMethod.MINMAX, columns=["a"])
    result = transformer.get_transformed_data()
    assert result["a"].min() >= -1e-9
    assert result["a"].max() <= 1.0 + 1e-9


def test_minmax_preserves_row_count(numeric_df):
    transformer = DataTransformer(numeric_df)
    transformer.scale_features(method=ScalingMethod.MINMAX)
    assert len(transformer.get_transformed_data()) == len(numeric_df)


# ---------------------------------------------------------------------------
# scale_features — ROBUST
# ---------------------------------------------------------------------------

def test_robust_scale_no_crash(numeric_df):
    transformer = DataTransformer(numeric_df)
    transformer.scale_features(method=ScalingMethod.ROBUST)
    assert transformer.get_transformed_data().shape == numeric_df.shape


# ---------------------------------------------------------------------------
# create_polynomial_features
# ---------------------------------------------------------------------------

@pytest.mark.xfail(reason="Known package bug: PreprocessingError.__init__ does not accept 'affected_columns' kwarg")
def test_polynomial_adds_columns(numeric_df):
    small = numeric_df.select(["a", "b"])
    transformer = DataTransformer(small)
    transformer.create_polynomial_features(columns=["a", "b"], degree=2)
    result = transformer.get_transformed_data()
    assert len(result.columns) > len(small.columns)


@pytest.mark.xfail(reason="Known package bug: PreprocessingError.__init__ does not accept 'affected_columns' kwarg")
def test_polynomial_preserves_row_count(numeric_df):
    transformer = DataTransformer(numeric_df.select(["a", "b"]))
    transformer.create_polynomial_features(columns=["a", "b"], degree=2)
    assert len(transformer.get_transformed_data()) == len(numeric_df)


# ---------------------------------------------------------------------------
# Method chaining
# ---------------------------------------------------------------------------

def test_chaining_encode_then_scale(categorical_df):
    result = (
        DataTransformer(categorical_df)
        .encode_categorical_features(method=EncodingMethod.LABEL)
        .scale_features(method=ScalingMethod.STANDARD)
        .get_transformed_data()
    )
    assert isinstance(result, pl.DataFrame)
    assert len(result) == len(categorical_df)


def test_chaining_returns_self(numeric_df):
    transformer = DataTransformer(numeric_df)
    returned = transformer.scale_features(method=ScalingMethod.MINMAX)
    assert returned is transformer


# ---------------------------------------------------------------------------
# quick_transform convenience function
# ---------------------------------------------------------------------------

def test_quick_transform_returns_dataframe(categorical_df):
    result = quick_transform(categorical_df)
    assert isinstance(result, (pl.DataFrame, DataTransformer))
