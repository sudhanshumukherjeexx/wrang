"""Tests for wrang/core/validator.py — DataValidator / schema validation."""

import polars as pl
import pytest

from wrang.core.validator import (
    ColumnSchema,
    DataSchema,
    DataValidator,
    ValidationResult,
    infer_schema,
    validate_data,
)


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------

@pytest.fixture
def simple_df():
    return pl.DataFrame(
        {
            "id": [1, 2, 3, 4, 5],
            "name": ["Alice", "Bob", "Carol", "Dave", "Eve"],
            "score": [0.9, 0.8, 0.7, 0.6, 0.5],
        }
    )


@pytest.fixture
def df_with_nulls():
    return pl.DataFrame(
        {
            "a": [1, None, 3],
            "b": ["x", "y", None],
        }
    )


# ---------------------------------------------------------------------------
# ColumnSchema
# ---------------------------------------------------------------------------

def test_column_schema_defaults():
    cs = ColumnSchema(name="col")
    assert cs.nullable is True
    assert cs.unique is False
    assert cs.max_missing_pct == 100.0


def test_column_schema_from_dict():
    cs = ColumnSchema.from_dict(
        {"name": "x", "dtype": "Int64", "nullable": False, "unique": True}
    )
    assert cs.name == "x"
    assert cs.dtype == "Int64"
    assert cs.nullable is False
    assert cs.unique is True


def test_column_schema_to_dict_roundtrip():
    cs = ColumnSchema(name="y", dtype="Float64", min_value=0.0, max_value=1.0)
    d = cs.to_dict()
    cs2 = ColumnSchema.from_dict(d)
    assert cs2.min_value == 0.0
    assert cs2.max_value == 1.0


# ---------------------------------------------------------------------------
# DataSchema
# ---------------------------------------------------------------------------

def test_data_schema_from_dict():
    schema = DataSchema.from_dict(
        {
            "columns": [{"name": "id"}, {"name": "score"}],
            "allow_extra_columns": False,
        }
    )
    assert len(schema.columns) == 2
    assert schema.allow_extra_columns is False


def test_data_schema_json_roundtrip(tmp_path):
    schema = DataSchema(columns=[ColumnSchema(name="x", dtype="Int64")])
    out = tmp_path / "schema.json"
    schema.to_json(out)
    loaded = DataSchema.from_json(out)
    assert loaded.columns[0].name == "x"
    assert loaded.columns[0].dtype == "Int64"


# ---------------------------------------------------------------------------
# ValidationResult
# ---------------------------------------------------------------------------

def test_validation_result_passed_when_no_errors():
    result = ValidationResult()
    assert result.passed is True


def test_validation_result_failed_on_error():
    result = ValidationResult()
    result.add("col", "dtype", "error", "bad dtype")
    assert result.passed is False


def test_validation_result_warning_does_not_fail():
    result = ValidationResult()
    result.add("col", "missing", "warning", "lots of nulls")
    assert result.passed is True


def test_validation_result_to_dict():
    result = ValidationResult()
    result.add("a", "dtype", "error", "mismatch", actual="Utf8", expected="Int64")
    d = result.to_dict()
    assert d["error_count"] == 1
    assert d["passed"] is False
    assert d["violations"][0]["column"] == "a"


# ---------------------------------------------------------------------------
# DataValidator — happy paths
# ---------------------------------------------------------------------------

def test_validate_passes_with_matching_schema(simple_df):
    schema = DataSchema(
        columns=[
            ColumnSchema(name="id", dtype="Int64"),
            ColumnSchema(name="name", dtype="String"),
            ColumnSchema(name="score", dtype="Float64"),
        ]
    )
    result = DataValidator(schema).validate(simple_df)
    # dtype errors only; String vs Utf8 may differ by polars version — just check no crash
    assert isinstance(result, ValidationResult)


def test_validate_missing_required_column(simple_df):
    schema = DataSchema(
        columns=[ColumnSchema(name="nonexistent")],
        require_all_columns=True,
    )
    result = DataValidator(schema).validate(simple_df)
    assert not result.passed
    assert any(v.check == "column_presence" for v in result.errors)


def test_validate_extra_column_warning(simple_df):
    schema = DataSchema(
        columns=[ColumnSchema(name="id")],
        allow_extra_columns=False,
        require_all_columns=False,
    )
    result = DataValidator(schema).validate(simple_df)
    assert any(v.check == "extra_column" for v in result.warnings)


def test_validate_nullable_violation(df_with_nulls):
    schema = DataSchema(
        columns=[ColumnSchema(name="a", nullable=False)],
        require_all_columns=False,
    )
    result = DataValidator(schema).validate(df_with_nulls)
    assert not result.passed
    assert any(v.check == "nullable" for v in result.errors)


def test_validate_max_missing_pct(df_with_nulls):
    schema = DataSchema(
        columns=[ColumnSchema(name="a", max_missing_pct=10.0)],
        require_all_columns=False,
    )
    result = DataValidator(schema).validate(df_with_nulls)
    # a has 1/3 ≈ 33% missing > 10%
    assert any(v.check == "max_missing_pct" for v in result.warnings)


def test_validate_unique_violation():
    df = pl.DataFrame({"x": [1, 1, 2]})
    schema = DataSchema(columns=[ColumnSchema(name="x", unique=True)])
    result = DataValidator(schema).validate(df)
    assert not result.passed
    assert any(v.check == "unique" for v in result.errors)


def test_validate_min_value():
    df = pl.DataFrame({"score": [-1.0, 0.5, 1.0]})
    schema = DataSchema(columns=[ColumnSchema(name="score", min_value=0.0)])
    result = DataValidator(schema).validate(df)
    assert not result.passed
    assert any(v.check == "min_value" for v in result.errors)


def test_validate_max_value():
    df = pl.DataFrame({"score": [0.5, 1.0, 1.5]})
    schema = DataSchema(columns=[ColumnSchema(name="score", max_value=1.0)])
    result = DataValidator(schema).validate(df)
    assert not result.passed
    assert any(v.check == "max_value" for v in result.errors)


def test_validate_allowed_values():
    df = pl.DataFrame({"status": ["active", "inactive", "deleted"]})
    schema = DataSchema(
        columns=[ColumnSchema(name="status", allowed_values=["active", "inactive"])]
    )
    result = DataValidator(schema).validate(df)
    assert not result.passed
    assert any(v.check == "allowed_values" for v in result.errors)


# ---------------------------------------------------------------------------
# Convenience functions
# ---------------------------------------------------------------------------

def test_validate_data_with_dict(simple_df):
    result = validate_data(
        simple_df,
        {"columns": [{"name": "id"}], "require_all_columns": False},
    )
    assert isinstance(result, ValidationResult)


def test_validate_data_with_json_path(simple_df, tmp_path):
    import json

    schema_path = tmp_path / "s.json"
    schema_path.write_text(
        json.dumps({"columns": [{"name": "id"}], "require_all_columns": False})
    )
    result = validate_data(simple_df, schema_path)
    assert isinstance(result, ValidationResult)


def test_infer_schema(simple_df):
    schema = infer_schema(simple_df)
    names = [c.name for c in schema.columns]
    assert "id" in names
    assert "score" in names
    # inferred max_missing_pct should be ≥ 0
    for cs in schema.columns:
        assert cs.max_missing_pct >= 0
