"""Tests for ride/cli/interface.py argument parser and non-interactive commands."""

import json

import polars as pl
import pytest

from wrang.cli.interface import create_argument_parser, check_dependencies


# ---------------------------------------------------------------------------
# Argument parser
# ---------------------------------------------------------------------------

@pytest.fixture
def parser():
    return create_argument_parser()


def test_parser_defaults(parser):
    args = parser.parse_args([])
    assert args.file is None
    assert args.inspect is False
    assert args.output_format == "text"
    assert args.chunk_size is None
    assert args.sql is None
    assert args.compare is None
    assert args.profile is False


def test_parser_file(parser):
    args = parser.parse_args(["data.csv"])
    assert args.file == "data.csv"


def test_parser_inspect(parser):
    args = parser.parse_args(["data.csv", "--inspect"])
    assert args.inspect is True


def test_parser_output_format_json(parser):
    args = parser.parse_args(["data.csv", "--inspect", "--output-format", "json"])
    assert args.output_format == "json"


def test_parser_sql(parser):
    args = parser.parse_args(["data.csv", "--sql", "SELECT * FROM data LIMIT 5"])
    assert "LIMIT 5" in args.sql


def test_parser_compare(parser):
    args = parser.parse_args(["--compare", "a.csv", "b.csv"])
    assert args.compare == ["a.csv", "b.csv"]


def test_parser_chunk_size(parser):
    args = parser.parse_args(["data.csv", "--chunk-size", "1000"])
    assert args.chunk_size == 1000


def test_parser_profile(parser):
    args = parser.parse_args(["data.csv", "--profile"])
    assert args.profile is True


def test_parser_version(parser):
    args = parser.parse_args(["--version"])
    assert args.version is True


# ---------------------------------------------------------------------------
# check_dependencies
# ---------------------------------------------------------------------------

def test_check_dependencies_passes():
    # polars, rich, numpy should always be available in the test environment
    assert check_dependencies() is True


# ---------------------------------------------------------------------------
# Inspect command — JSON output (end-to-end, no real file needed via loader mock)
# ---------------------------------------------------------------------------

def test_inspect_json_output(tmp_path, capsys):
    csv_file = tmp_path / "sample.csv"
    pl.DataFrame({"a": [1, 2, 3], "b": ["x", "y", "z"]}).write_csv(csv_file)

    from wrang.cli.interface import main

    assert main([str(csv_file), "--inspect", "--output-format", "json"]) == 0

    captured = capsys.readouterr()
    # Strip Rich/init noise; find the JSON object line
    json_line = next(l for l in captured.out.splitlines() if l.startswith("{"))
    data = json.loads(json_line)
    assert data["rows"] == 3
    assert data["columns"] == 2
    assert "a" in data["column_names"]


# ---------------------------------------------------------------------------
# --compare command — JSON output
# ---------------------------------------------------------------------------

def test_compare_identical_files(tmp_path, capsys):
    df = pl.DataFrame({"x": [1, 2], "y": [3.0, 4.0]})
    f1 = tmp_path / "a.csv"
    f2 = tmp_path / "b.csv"
    df.write_csv(f1)
    df.write_csv(f2)

    from wrang.cli.interface import main

    main(["--compare", str(f1), str(f2), "--output-format", "json"])
    captured = capsys.readouterr()
    json_line = next(l for l in captured.out.splitlines() if l.startswith("{"))
    result = json.loads(json_line)

    assert result["schema_match"] is True
    assert result["shape_match"] is True
    assert result["columns_only_in_a"] == []
    assert result["columns_only_in_b"] == []


def test_compare_different_shapes(tmp_path, capsys):
    df_a = pl.DataFrame({"x": [1, 2, 3]})
    df_b = pl.DataFrame({"x": [1, 2]})
    f1 = tmp_path / "a.csv"
    f2 = tmp_path / "b.csv"
    df_a.write_csv(f1)
    df_b.write_csv(f2)

    from wrang.cli.interface import main

    main(["--compare", str(f1), str(f2), "--output-format", "json"])
    captured = capsys.readouterr()
    json_line = next(l for l in captured.out.splitlines() if l.startswith("{"))
    result = json.loads(json_line)

    assert result["shape_match"] is False


def test_compare_missing_column(tmp_path, capsys):
    f1 = tmp_path / "a.csv"
    f2 = tmp_path / "b.csv"
    pl.DataFrame({"x": [1], "y": [2]}).write_csv(f1)
    pl.DataFrame({"x": [1]}).write_csv(f2)

    from wrang.cli.interface import main

    main(["--compare", str(f1), str(f2), "--output-format", "json"])
    captured = capsys.readouterr()
    json_line = next(l for l in captured.out.splitlines() if l.startswith("{"))
    result = json.loads(json_line)

    assert "y" in result["columns_only_in_a"]
