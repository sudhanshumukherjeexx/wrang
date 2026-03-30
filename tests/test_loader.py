"""Tests for ride/core/loader.py — FastDataLoader and DataSaver."""

import json
from pathlib import Path

import polars as pl
import pytest

from wrang.core.loader import FastDataLoader, DataSaver, load_data, save_data


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------

@pytest.fixture
def sample_df():
    return pl.DataFrame({
        "id":    [1, 2, 3, 4, 5],
        "name":  ["Alice", "Bob", "Carol", "Dave", "Eve"],
        "score": [88.5, 92.0, 76.3, 55.0, 95.1],
        "pass_": [True, True, True, False, True],
    })


@pytest.fixture
def csv_file(tmp_path, sample_df):
    p = tmp_path / "sample.csv"
    sample_df.write_csv(p)
    return p


@pytest.fixture
def parquet_file(tmp_path, sample_df):
    p = tmp_path / "sample.parquet"
    sample_df.write_parquet(p)
    return p


@pytest.fixture
def loader():
    return FastDataLoader()


@pytest.fixture
def saver():
    return DataSaver()


# ---------------------------------------------------------------------------
# FastDataLoader — CSV
# ---------------------------------------------------------------------------

def test_load_csv_returns_dataframe(loader, csv_file, sample_df):
    df = loader.load(csv_file)
    assert isinstance(df, pl.DataFrame)
    assert len(df) == len(sample_df)


def test_load_csv_columns(loader, csv_file, sample_df):
    df = loader.load(csv_file)
    assert set(df.columns) == set(sample_df.columns)


def test_load_csv_string_path(loader, csv_file):
    df = loader.load(str(csv_file))
    assert len(df) == 5


def test_load_csv_values(loader, csv_file):
    df = loader.load(csv_file)
    assert df["id"].to_list() == [1, 2, 3, 4, 5]


# ---------------------------------------------------------------------------
# FastDataLoader — Parquet
# ---------------------------------------------------------------------------

def test_load_parquet_returns_dataframe(loader, parquet_file, sample_df):
    df = loader.load(parquet_file)
    assert len(df) == len(sample_df)


def test_load_parquet_columns(loader, parquet_file, sample_df):
    df = loader.load(parquet_file)
    assert set(df.columns) == set(sample_df.columns)


def test_load_parquet_float_dtype(loader, parquet_file):
    df = loader.load(parquet_file)
    assert df["score"].dtype in (pl.Float32, pl.Float64)


# ---------------------------------------------------------------------------
# FastDataLoader — peek
# ---------------------------------------------------------------------------

def test_peek_returns_n_rows(loader, csv_file):
    df = loader.peek(csv_file, n_rows=3)
    assert len(df) == 3


def test_peek_returns_at_least_one_row(loader, csv_file):
    df = loader.peek(csv_file)
    assert len(df) >= 1


# ---------------------------------------------------------------------------
# FastDataLoader — stream_chunks
# ---------------------------------------------------------------------------

def test_stream_chunks_yields_dataframes(loader, csv_file):
    for chunk in loader.stream_chunks(csv_file, chunk_size=2):
        assert isinstance(chunk, pl.DataFrame)


def test_stream_chunks_total_rows(loader, csv_file):
    total = sum(len(c) for c in loader.stream_chunks(csv_file, chunk_size=2))
    assert total == 5


def test_stream_chunks_non_empty_chunks(loader, csv_file):
    for chunk in loader.stream_chunks(csv_file, chunk_size=2):
        assert len(chunk) > 0


# ---------------------------------------------------------------------------
# FastDataLoader — error handling
# ---------------------------------------------------------------------------

def test_load_missing_file_raises(loader):
    with pytest.raises(Exception):
        loader.load("/nonexistent/path/data.csv")


def test_load_unsupported_extension_raises(loader, tmp_path):
    bad = tmp_path / "data.xyz"
    bad.write_text("col1,col2\n1,2")
    with pytest.raises(Exception):
        loader.load(bad)


# ---------------------------------------------------------------------------
# DataSaver — CSV
# ---------------------------------------------------------------------------

def test_save_csv_creates_file(saver, sample_df, tmp_path):
    out = tmp_path / "out.csv"
    saver.save(sample_df, out)
    assert out.exists()


def test_save_csv_correct_row_count(saver, sample_df, tmp_path):
    out = tmp_path / "out.csv"
    saver.save(sample_df, out)
    assert len(pl.read_csv(out)) == len(sample_df)


# ---------------------------------------------------------------------------
# DataSaver — Parquet
# ---------------------------------------------------------------------------

def test_save_parquet_creates_file(saver, sample_df, tmp_path):
    out = tmp_path / "out.parquet"
    saver.save(sample_df, out)
    assert out.exists()


def test_save_parquet_correct_row_count(saver, sample_df, tmp_path):
    out = tmp_path / "out.parquet"
    saver.save(sample_df, out)
    assert len(pl.read_parquet(out)) == len(sample_df)


# ---------------------------------------------------------------------------
# Roundtrip — save then reload
# ---------------------------------------------------------------------------

def test_csv_roundtrip(loader, saver, sample_df, tmp_path):
    out = tmp_path / "round.csv"
    saver.save(sample_df, out)
    reloaded = loader.load(out)
    assert reloaded.columns == sample_df.columns
    assert len(reloaded) == len(sample_df)


def test_parquet_roundtrip(loader, saver, sample_df, tmp_path):
    out = tmp_path / "round.parquet"
    saver.save(sample_df, out)
    reloaded = loader.load(out)
    assert len(reloaded) == len(sample_df)


# ---------------------------------------------------------------------------
# Convenience functions
# ---------------------------------------------------------------------------

def test_load_data_function(csv_file, sample_df):
    df = load_data(csv_file)
    assert len(df) == len(sample_df)


def test_save_data_function(sample_df, tmp_path):
    out = tmp_path / "conv.csv"
    save_data(sample_df, out)
    assert out.exists()
