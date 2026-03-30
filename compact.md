# wrang — Project Handoff Report

**Date:** 2026-03-30
**Repo:** `/home/sud_bit/Documents/GitHub/ride_cli`
**Branch:** `main`
**Status:** Phase 1 complete. Phase 2 (GitHub Pages website) not yet started.

---

## 1. What this project is

`wrang` is a terminal-native data wrangling toolkit built on top of [Polars](https://pola.rs). It started life as `prepup-linux`, was rebranded to `ride-cli`, and has now been relaunched as `wrang` at a clean `v0.1.0`.

It ships as two things simultaneously:
- A **CLI tool** (`wrang` command) — a fully interactive menu-driven terminal application for loading, inspecting, cleaning, transforming, and exporting datasets
- A **Python library** (`import wrang`) — every core module is independently importable and usable in scripts or notebooks

---

## 2. What was done in this session

### Step 1 — Full project audit
A deep exploration of the entire codebase was done. All modules, classes, methods, CLI commands, configuration system, data flow, test structure, and dependencies were documented. This was saved to:

**[`project_details.md`](project_details.md)** — the single source of truth for any developer new to this project. Read this first.

### Step 2 — Rebranding: `ride` → `wrang`

The package was fully rebranded. Every file was touched. Specifically:

| What | Before | After |
|---|---|---|
| PyPI package name | `ride-cli` | `wrang` |
| Python package folder | `ride/` | `wrang/` |
| CLI command | `ride` | `wrang` |
| Version | `0.4.0` | `0.1.0` (clean semver restart) |
| Config dir | `~/.ride/` | `~/.wrang/` |
| Env var | `RIDE_NO_WELCOME` | `WRANG_NO_WELCOME` |
| GitHub URL | `.../ride-cli` | `.../wrang` |
| Import | `from ride.x import Y` | `from wrang.x import Y` |

**Files created or rewritten:**

| File | Action |
|---|---|
| `wrang/` | Created (copied from `ride/`, all imports updated) |
| `wrang/__init__.py` | Rewritten — v0.1.0, wrang brand, new env var, new config dir |
| `wrang/main.py` | Rewritten — clean, wrang brand |
| `wrang/config.py` | Updated — `~/.wrang/` config dir, docstrings |
| `wrang/cli/interface.py` | Updated — `prog='wrang'`, all help text, all URLs |
| `wrang/cli/formatters.py` | Updated — banner text, version, menu labels |
| `wrang/cli/menus.py` | Updated — brand strings, GitHub URLs |
| `wrang/viz/export_utils.py` | Updated — report title, GitHub URL |
| `wrang/utils/exceptions.py` | Updated — brand strings |
| `wrang/utils/constants.py` | Updated — brand strings |
| `wrang/_public_api.py` | Updated — all `from ride.` → `from wrang.` |
| `tests/*.py` (all 10) | Updated — all `from ride.` → `from wrang.` |
| `test_file.py` | Updated — imports, tmp dir `/tmp/wrang_test`, print strings |
| `test_cli.md` | Updated — all `ride` CLI commands → `wrang` |
| `test_notebook.ipynb` | Updated — all `ride` refs → `wrang` throughout cells |
| `pyproject.toml` | Updated — name, version, script entry point, URLs, include pattern |
| `CHANGELOG.md` | Rewritten — fresh `v0.1.0` entry, old history dropped |
| `README.md` | Rewritten from scratch — modern minimalist, industry standard |
| `project_details.md` | Created — full developer reference document |

**The old `ride/` directory** has not been deleted yet. It is safe to delete — nothing references it. `pyproject.toml` only includes `wrang*`, and all tests and source files import from `wrang.*`.

---

## 3. Current repo layout

```
ride_cli/                        ← repo root (folder name is still ride_cli)
│
├── wrang/                       ← THE active Python package
│   ├── __init__.py              ← lazy public API, v0.1.0
│   ├── __main__.py              ← python -m wrang entry
│   ├── main.py                  ← cli_entry_point()
│   ├── config.py                ← RideConfig dataclass, enums, ~/.wrang/
│   ├── _public_api.py           ← eager imports of all public symbols
│   ├── cli/
│   │   ├── interface.py         ← RideCLI, argument parser, mode dispatch
│   │   ├── formatters.py        ← RideFormatter (Rich-based terminal UI)
│   │   └── menus.py             ← MenuHandler (interactive session state)
│   ├── core/
│   │   ├── loader.py            ← FastDataLoader, DataSaver
│   │   ├── inspector.py         ← DataInspector
│   │   ├── explorer.py          ← DataExplorer
│   │   ├── cleaner.py           ← DataCleaner, BatchCleaner
│   │   ├── transformer.py       ← DataTransformer, TransformationPipeline
│   │   └── validator.py         ← DataValidator, DataSchema, ColumnSchema
│   ├── utils/
│   │   ├── exceptions.py        ← RideError hierarchy
│   │   └── constants.py         ← type helpers, formatters
│   └── viz/
│       └── export_utils.py      ← generate_html_report()
│
├── tests/                       ← 217 passing, 2 xfail (all import from wrang.)
│   ├── test_loader.py
│   ├── test_inspector.py
│   ├── test_explorer.py
│   ├── test_cleaner.py
│   ├── test_transformer.py
│   ├── test_config.py
│   ├── test_cli.py
│   ├── test_automl.py
│   ├── test_basic.py
│   └── test_utils.py
│
├── pyproject.toml               ← name="wrang", version="0.1.0", script=wrang
├── requirements.txt
├── README.md                    ← rewritten from scratch
├── CHANGELOG.md                 ← fresh v0.1.0
├── project_details.md           ← full developer reference
├── compact.md                   ← this file
│
├── test_file.py                 ← end-to-end library demo (python test_file.py)
├── test_cli.md                  ← manual CLI testing guide
├── test_notebook.ipynb          ← notebook usage demo
├── titanic_schema.json          ← sample validation schema
│
└── ride/                        ← OLD package — safe to delete, nothing uses it
```

---

## 4. How to install and run

```bash
# Install in development mode
cd /home/sud_bit/Documents/GitHub/ride_cli
pip install -e .

# Verify
wrang --version

# Run interactive mode
wrang

# Run tests
python -m pytest tests/ -v
# Expected: 217 passed, 2 xfailed, 0 failed

# Run the library demo
python test_file.py

# Run module entry point
python -m wrang
```

---

## 5. Test suite status (as of 2026-03-29)

| File | Passed | XFailed | Failed |
|---|---|---|---|
| test_automl.py | 21 | 0 | 0 |
| test_basic.py | 3 | 0 | 0 |
| test_cleaner.py | 27 | 0 | 0 |
| test_cli.py | 13 | 0 | 0 |
| test_config.py | 26 | 0 | 0 |
| test_explorer.py | 27 | 0 | 0 |
| test_inspector.py | 20 | 0 | 0 |
| test_loader.py | 22 | 0 | 0 |
| test_transformer.py | 15 | 2 | 0 |
| test_utils.py | 23 | 0 | 0 |
| **Total** | **217** | **2** | **0** |

**Known xfail** (pre-existing, not regressions):
- `test_polynomial_adds_columns` — `PreprocessingError` in the polynomial feature path does not accept `affected_columns` kwarg
- `test_polynomial_preserves_row_count` — same root cause
- Both are in `tests/test_transformer.py` and marked `@pytest.mark.xfail`

---

## 6. Known issues / immediate cleanup

| Item | Action needed |
|---|---|
| `ride/` directory still exists in repo | Delete it — nothing references it |
| `wrang.egg-info/` and `ride_cli.egg-info/` exist | Normal after `pip install -e .` |
| Polynomial `PreprocessingError` xfail | Fix `DataTransformer.create_polynomial_features()` to pass `affected_columns` to `PreprocessingError.__init__()` in `wrang/utils/exceptions.py` |

---

## 7. What comes next — Phase 2 (not started)

**GitHub Pages website** for `wrang`.

Agreed plan:
- Lives in `docs/` folder on `main` branch → enabled via GitHub Pages settings
- Pure static HTML/CSS/JS — no build step, no Jekyll, no CI required
- Hosted at: `https://sudhanshumukherjeexx.github.io/wrang`

Planned directory structure:
```
docs/
├── index.html               ← Landing page (hero, install, feature cards, quick-start)
├── assets/
│   ├── css/style.css
│   └── js/main.js
├── examples/
│   ├── index.html           ← Examples hub
│   ├── terminal.html        ← Interactive CLI walkthrough (styled terminal output)
│   ├── python-api.html      ← .py file usage with annotated snippets
│   └── notebook.html        ← Rendered notebook cells (pre-rendered HTML)
└── docs/
    ├── quickstart.html      ← Install + first run
    ├── api-reference.html   ← All classes/functions
    ├── configuration.html   ← RideConfig + ~/.wrang/config.json
    └── changelog.html       ← Version history
```

Design decisions already agreed:
- Dark/light theme toggle
- highlight.js for syntax highlighting (CDN)
- Styled `<pre>` blocks that mimic real terminal (ANSI colors via CSS)
- Notebook examples: pre-rendered HTML cells, no nbconvert server
- GitHub username: `sudhanshumukherjeexx`
- Same repo as code (`docs/` folder), not a separate `gh-pages` branch

---

## 8. Key reference files

| File | Purpose |
|---|---|
| [`project_details.md`](project_details.md) | Full developer reference — architecture, all classes/methods, CLI, config, data flow, tests |
| [`README.md`](README.md) | User-facing documentation — install, quick start, API examples, CLI reference |
| [`CHANGELOG.md`](CHANGELOG.md) | Version history starting at v0.1.0 |
| [`test_cli.md`](test_cli.md) | Manual QA guide — every interactive menu option + all CLI flags |
| [`test_file.py`](test_file.py) | End-to-end library usage demo |
| [`test_notebook.ipynb`](test_notebook.ipynb) | Notebook usage demo |

---

## 9. Quick architecture recap

```
wrang command
  └─► wrang/main.py : cli_entry_point()
        └─► wrang/cli/interface.py : RideCLI.run()
              ├── Interactive  → wrang/cli/menus.py : MenuHandler.run_main_menu()
              └── CLI flags   → RideCLI._run_inspect / _run_sql / _run_compare / etc.

Python import
  └─► import wrang           (lazy — fast, no deps loaded until needed)
        └─► wrang/__init__.py : __getattr__()
              └─► wrang/_public_api.py  (loaded on first symbol access)
                    └─► wrang/core/* , wrang/config.py , wrang/cli/formatters.py
```

**Data flow (typical pipeline):**
```
File → FastDataLoader → pl.DataFrame
         → DataInspector   (profile)
         → DataCleaner     (impute, deduplicate, outliers)
         → DataTransformer (encode, scale, feature engineering)
         → DataValidator   (schema check)
         → DataSaver       → output file
```

**Config:** single `RideConfig` dataclass singleton via `get_config()`. Persisted at `~/.wrang/config.json`.

---

*Handoff prepared: 2026-03-30*
