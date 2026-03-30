#!/usr/bin/env python3
"""
ride/viz/export_utils.py
HTML data-profile report generator.

Produces a self-contained, single-file HTML report with:
  - Dataset overview (shape, memory, dtypes)
  - Per-column cards (stats, missing %, distribution bar)
  - Correlation heatmap table (numeric cols)
  - Sample rows
  - No external JS/CSS dependencies (everything is inline)
"""

from __future__ import annotations

import html
import json
from pathlib import Path
from typing import Any, Dict, List, Optional, Union

import polars as pl

from wrang.utils.constants import (
    numeric_columns,
    categorical_columns,
    format_memory,
    classify_correlation,
)


# ---------------------------------------------------------------------------
# Public entry point
# ---------------------------------------------------------------------------

def generate_html_report(
    df: pl.DataFrame,
    output_path: Union[str, Path] = "data_profile.html",
    title: str = "wrang — Data Profile Report",
    sample_rows: int = 10,
    max_cat_values: int = 10,
) -> Path:
    """
    Generate a self-contained HTML data-profile report for *df*.

    Parameters
    ----------
    df:             The DataFrame to profile.
    output_path:    Where to write the .html file.
    title:          Report title shown in the browser tab and heading.
    sample_rows:    Number of sample rows to include at the bottom.
    max_cat_values: Maximum distinct values shown in categorical bar charts.

    Returns
    -------
    Path to the written file.
    """
    output_path = Path(output_path)
    overview = _build_overview(df)
    col_profiles = _build_column_profiles(df, max_cat_values)
    corr_table = _build_correlation_table(df)
    sample = _build_sample(df, sample_rows)

    html_content = _render_html(title, overview, col_profiles, corr_table, sample)
    output_path.write_text(html_content, encoding="utf-8")
    return output_path


# ---------------------------------------------------------------------------
# Data collection helpers
# ---------------------------------------------------------------------------

def _build_overview(df: pl.DataFrame) -> Dict[str, Any]:
    n_rows, n_cols = df.shape
    mem_mb = df.estimated_size() / (1024 ** 2)

    dtype_counts: Dict[str, int] = {}
    for dtype in df.dtypes:
        key = str(dtype)
        dtype_counts[key] = dtype_counts.get(key, 0) + 1

    missing_cells = sum(df[c].null_count() for c in df.columns)
    total_cells = n_rows * n_cols

    return {
        "rows": n_rows,
        "cols": n_cols,
        "memory": format_memory(mem_mb),
        "dtype_counts": dtype_counts,
        "missing_cells": missing_cells,
        "missing_pct": round(missing_cells / total_cells * 100, 2) if total_cells else 0.0,
        "duplicate_rows": df.is_duplicated().sum(),
    }


def _build_column_profiles(df: pl.DataFrame, max_cat_values: int) -> List[Dict[str, Any]]:
    profiles = []
    n = len(df)
    num_cols = set(numeric_columns(df))
    cat_cols = set(categorical_columns(df))

    for col in df.columns:
        s = df[col]
        null_count = s.null_count()
        null_pct = round(null_count / n * 100, 2) if n else 0.0
        p: Dict[str, Any] = {
            "name": col,
            "dtype": str(s.dtype),
            "null_count": null_count,
            "null_pct": null_pct,
            "unique": s.n_unique(),
        }

        if col in num_cols:
            non_null = s.drop_nulls()
            if len(non_null):
                p.update(
                    {
                        "kind": "numeric",
                        "mean": round(float(non_null.mean()), 4),
                        "std": round(float(non_null.std()), 4),
                        "min": round(float(non_null.min()), 4),
                        "q25": round(float(non_null.quantile(0.25)), 4),
                        "median": round(float(non_null.median()), 4),
                        "q75": round(float(non_null.quantile(0.75)), 4),
                        "max": round(float(non_null.max()), 4),
                    }
                )
            else:
                p["kind"] = "numeric"

        elif col in cat_cols:
            vc = (
                s.value_counts(sort=True)
                .head(max_cat_values)
                .to_pandas()  # for easy iteration
            )
            p["kind"] = "categorical"
            p["top_values"] = [
                {"value": str(row.iloc[0]), "count": int(row.iloc[1])}
                for _, row in vc.iterrows()
            ]

        else:
            p["kind"] = "other"

        profiles.append(p)
    return profiles


def _build_correlation_table(df: pl.DataFrame) -> Optional[Dict[str, Any]]:
    num_cols = numeric_columns(df)
    if len(num_cols) < 2:
        return None

    try:
        corr_matrix: Dict[str, Dict[str, float]] = {}
        for c1 in num_cols:
            corr_matrix[c1] = {}
            for c2 in num_cols:
                s1 = df[c1].drop_nulls().cast(pl.Float64)
                s2 = df[c2].drop_nulls().cast(pl.Float64)
                min_len = min(len(s1), len(s2))
                if min_len < 2:
                    corr_matrix[c1][c2] = 0.0
                else:
                    corr_matrix[c1][c2] = round(
                        float(s1[:min_len].pearson_corr(s2[:min_len])), 3
                    )
        return {"columns": num_cols, "matrix": corr_matrix}
    except Exception:
        return None


def _build_sample(df: pl.DataFrame, n: int) -> Dict[str, Any]:
    sample_df = df.head(n)
    return {
        "columns": sample_df.columns,
        "rows": [
            [str(v) if v is not None else "" for v in row]
            for row in sample_df.iter_rows()
        ],
    }


# ---------------------------------------------------------------------------
# HTML rendering
# ---------------------------------------------------------------------------

def _render_html(
    title: str,
    overview: Dict[str, Any],
    col_profiles: List[Dict[str, Any]],
    corr_table: Optional[Dict[str, Any]],
    sample: Dict[str, Any],
) -> str:
    css = _css()
    overview_html = _render_overview(overview)
    columns_html = _render_columns(col_profiles)
    corr_html = _render_correlation(corr_table)
    sample_html = _render_sample(sample)

    return f"""<!DOCTYPE html>
<html lang="en">
<head>
  <meta charset="UTF-8"/>
  <meta name="viewport" content="width=device-width, initial-scale=1.0"/>
  <title>{html.escape(title)}</title>
  <style>{css}</style>
</head>
<body>
  <header>
    <h1>{html.escape(title)}</h1>
    <p class="subtitle">Generated by <strong>RIDE CLI</strong> — Rapid Insights Data Engine</p>
  </header>
  <main>
    {overview_html}
    {columns_html}
    {corr_html}
    {sample_html}
  </main>
  <footer>
    <p>Generated with <a href="https://github.com/sudhanshumukherjeexx/wrang">wrang</a></p>
  </footer>
</body>
</html>"""


def _render_overview(ov: Dict[str, Any]) -> str:
    dtype_rows = "".join(
        f"<tr><td>{html.escape(k)}</td><td>{v}</td></tr>"
        for k, v in ov["dtype_counts"].items()
    )
    return f"""
<section id="overview">
  <h2>Dataset Overview</h2>
  <div class="card-grid">
    <div class="card stat-card"><div class="stat-val">{ov['rows']:,}</div><div class="stat-lbl">Rows</div></div>
    <div class="card stat-card"><div class="stat-val">{ov['cols']:,}</div><div class="stat-lbl">Columns</div></div>
    <div class="card stat-card"><div class="stat-val">{ov['memory']}</div><div class="stat-lbl">Memory</div></div>
    <div class="card stat-card"><div class="stat-val">{ov['missing_pct']}%</div><div class="stat-lbl">Missing Cells</div></div>
    <div class="card stat-card"><div class="stat-val">{ov['duplicate_rows']:,}</div><div class="stat-lbl">Duplicate Rows</div></div>
  </div>
  <div class="card">
    <h3>Data Types</h3>
    <table class="info-table"><thead><tr><th>Dtype</th><th>Count</th></tr></thead><tbody>{dtype_rows}</tbody></table>
  </div>
</section>"""


def _render_columns(profiles: List[Dict[str, Any]]) -> str:
    cards = []
    for p in profiles:
        kind = p.get("kind", "other")
        name = html.escape(p["name"])
        dtype = html.escape(p["dtype"])
        null_pct = p["null_pct"]
        unique = p["unique"]

        # missing bar
        bar_fill = _color_pct(null_pct)
        missing_bar = (
            f'<div class="bar-bg"><div class="bar-fill" style="width:{null_pct}%;background:{bar_fill};"></div></div>'
            f'<span class="bar-label">{null_pct}% missing ({p["null_count"]:,} cells)</span>'
        )

        body = ""
        if kind == "numeric":
            stats = {k: p.get(k, "—") for k in ("mean", "std", "min", "q25", "median", "q75", "max")}
            stat_cells = "".join(
                f'<div class="stat-mini"><span class="lbl">{k}</span><span class="val">{v}</span></div>'
                for k, v in stats.items()
            )
            body = f'<div class="stat-mini-grid">{stat_cells}</div>'

        elif kind == "categorical" and p.get("top_values"):
            top_vals = p["top_values"]
            max_count = max(tv["count"] for tv in top_vals) or 1
            rows = ""
            for tv in top_vals:
                pct = tv["count"] / max_count * 100
                rows += (
                    f'<div class="cat-row">'
                    f'<span class="cat-val">{html.escape(str(tv["value"]))}</span>'
                    f'<div class="bar-bg sm"><div class="bar-fill" style="width:{pct}%;background:#6c8ebf;"></div></div>'
                    f'<span class="cat-count">{tv["count"]:,}</span>'
                    f'</div>'
                )
            body = f'<div class="cat-bars">{rows}</div>'

        card = f"""
  <div class="col-card card">
    <div class="col-header">
      <span class="col-name">{name}</span>
      <span class="badge badge-{kind}">{dtype}</span>
      <span class="badge badge-dim">{unique:,} unique</span>
    </div>
    <div class="missing-bar">{missing_bar}</div>
    {body}
  </div>"""
        cards.append(card)

    return f'<section id="columns"><h2>Column Profiles</h2><div class="col-grid">{"".join(cards)}</div></section>'


def _render_correlation(corr_table: Optional[Dict[str, Any]]) -> str:
    if not corr_table:
        return ""
    cols = corr_table["columns"]
    matrix = corr_table["matrix"]

    header = "".join(f"<th>{html.escape(c)}</th>" for c in cols)
    rows_html = ""
    for c1 in cols:
        row_cells = "".join(
            f'<td style="background:{_corr_color(matrix[c1][c2])};" title="{classify_correlation(abs(matrix[c1][c2]))}">'
            f'{matrix[c1][c2]}</td>'
            for c2 in cols
        )
        rows_html += f"<tr><th>{html.escape(c1)}</th>{row_cells}</tr>"

    return f"""
<section id="correlation">
  <h2>Correlation Matrix (Pearson)</h2>
  <div class="table-scroll">
    <table class="corr-table">
      <thead><tr><th></th>{header}</tr></thead>
      <tbody>{rows_html}</tbody>
    </table>
  </div>
</section>"""


def _render_sample(sample: Dict[str, Any]) -> str:
    header = "".join(f"<th>{html.escape(c)}</th>" for c in sample["columns"])
    rows_html = ""
    for row in sample["rows"]:
        cells = "".join(f"<td>{html.escape(str(v))}</td>" for v in row)
        rows_html += f"<tr>{cells}</tr>"
    return f"""
<section id="sample">
  <h2>Sample Data (first {len(sample['rows'])} rows)</h2>
  <div class="table-scroll">
    <table class="data-table">
      <thead><tr>{header}</tr></thead>
      <tbody>{rows_html}</tbody>
    </table>
  </div>
</section>"""


# ---------------------------------------------------------------------------
# Color helpers
# ---------------------------------------------------------------------------

def _color_pct(pct: float) -> str:
    """Red for high missing %, green for low."""
    if pct > 50:
        return "#e74c3c"
    if pct > 20:
        return "#f39c12"
    return "#27ae60"


def _corr_color(val: float) -> str:
    """Blue for negative, white for zero, red for positive correlation."""
    v = max(-1.0, min(1.0, val))
    if v >= 0:
        r = int(220 + (255 - 220) * (1 - v))
        g = int(220 - 220 * v)
        b = int(220 - 220 * v)
    else:
        r = int(220 - 220 * (-v))
        g = int(220 - 220 * (-v))
        b = int(220 + (255 - 220) * (-v))
    return f"rgb({r},{g},{b})"


# ---------------------------------------------------------------------------
# Inline CSS
# ---------------------------------------------------------------------------

def _css() -> str:
    return """
* { box-sizing: border-box; margin: 0; padding: 0; }
body { font-family: -apple-system, BlinkMacSystemFont, 'Segoe UI', Roboto, sans-serif;
       background: #f4f6f9; color: #2c3e50; font-size: 14px; }
header { background: linear-gradient(135deg, #1a1a2e 0%, #16213e 50%, #0f3460 100%);
         color: #fff; padding: 32px 40px; }
header h1 { font-size: 28px; font-weight: 700; margin-bottom: 6px; }
.subtitle { opacity: 0.7; font-size: 13px; }
main { max-width: 1400px; margin: 0 auto; padding: 32px 24px; }
footer { text-align: center; padding: 24px; color: #7f8c8d; font-size: 12px; }
footer a { color: #3498db; text-decoration: none; }
section { margin-bottom: 40px; }
section h2 { font-size: 20px; font-weight: 600; margin-bottom: 16px;
             padding-bottom: 8px; border-bottom: 2px solid #3498db; }
section h3 { font-size: 16px; font-weight: 600; margin-bottom: 12px; }
.card { background: #fff; border-radius: 8px; padding: 20px;
        box-shadow: 0 2px 8px rgba(0,0,0,0.06); }
.card-grid { display: grid; grid-template-columns: repeat(auto-fit, minmax(160px, 1fr));
             gap: 16px; margin-bottom: 16px; }
.stat-card { text-align: center; }
.stat-val { font-size: 28px; font-weight: 700; color: #3498db; }
.stat-lbl { font-size: 12px; color: #7f8c8d; margin-top: 4px; }
.info-table { width: 100%; border-collapse: collapse; }
.info-table th, .info-table td { padding: 8px 12px; text-align: left;
  border-bottom: 1px solid #ecf0f1; }
.info-table th { background: #f8f9fa; font-weight: 600; }
.col-grid { display: grid; grid-template-columns: repeat(auto-fill, minmax(340px, 1fr)); gap: 16px; }
.col-card { padding: 16px; }
.col-header { display: flex; align-items: center; gap: 8px; flex-wrap: wrap; margin-bottom: 10px; }
.col-name { font-weight: 600; font-size: 15px; }
.badge { font-size: 11px; padding: 2px 8px; border-radius: 12px; font-weight: 500; }
.badge-numeric { background: #d5e8f4; color: #1a5276; }
.badge-categorical { background: #d5f5e3; color: #1e8449; }
.badge-other { background: #f9ebea; color: #922b21; }
.badge-dim { background: #f2f3f4; color: #7f8c8d; }
.missing-bar { margin-bottom: 10px; }
.bar-bg { background: #ecf0f1; border-radius: 4px; height: 8px; width: 100%; overflow: hidden; }
.bar-bg.sm { height: 6px; }
.bar-fill { height: 100%; border-radius: 4px; transition: width 0.3s; }
.bar-label { font-size: 11px; color: #7f8c8d; margin-top: 3px; display: block; }
.stat-mini-grid { display: grid; grid-template-columns: repeat(4, 1fr); gap: 6px; margin-top: 6px; }
.stat-mini { background: #f8f9fa; border-radius: 4px; padding: 6px 8px; text-align: center; }
.stat-mini .lbl { display: block; font-size: 10px; color: #7f8c8d; text-transform: uppercase; }
.stat-mini .val { display: block; font-size: 13px; font-weight: 600; color: #2c3e50; margin-top: 2px; }
.cat-bars { margin-top: 6px; }
.cat-row { display: flex; align-items: center; gap: 8px; margin-bottom: 5px; }
.cat-val { width: 120px; overflow: hidden; text-overflow: ellipsis; white-space: nowrap;
           font-size: 12px; color: #2c3e50; }
.cat-row .bar-bg { flex: 1; }
.cat-count { width: 50px; text-align: right; font-size: 11px; color: #7f8c8d; }
.table-scroll { overflow-x: auto; }
.corr-table, .data-table { border-collapse: collapse; font-size: 12px; min-width: 100%; }
.corr-table th, .corr-table td { border: 1px solid #ddd; padding: 6px 10px; text-align: center; }
.corr-table th { background: #f8f9fa; font-weight: 600; }
.data-table th { background: #1a1a2e; color: #fff; padding: 8px 12px; text-align: left; }
.data-table td { padding: 7px 12px; border-bottom: 1px solid #ecf0f1; }
.data-table tr:hover td { background: #f8f9fa; }
"""
