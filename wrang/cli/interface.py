#!/usr/bin/env python3
"""
cli/interface.py
"""

import sys
import os
import signal
import argparse
import traceback
from pathlib import Path
from typing import Optional, List, Any
from rich.console import Console
from rich.traceback import install

# Install rich traceback handler for better error display
try:
    install(show_locals=True)
except Exception:
    pass

console = Console()

class RideCLI:
    """
    Main RIDE CLI Application
    """
    
    def __init__(self):
        self.config = None
        self.formatter = None
        self.menu_handler = None
        self.loader = None
        
        # Test each import individually for better debugging
        console.print("[blue]Initializing wrang...[/blue]")
        
        # Config
        try:
            from wrang.config import get_config
            self.config = get_config()
            console.print("✅ Config loaded")
        except Exception as e:
            console.print(f"❌ Config error: {e}")
            self.config = type('Config', (), {'debug': False, 'verbose': False})()
        
        # Formatter
        try:
            from wrang.cli.formatters import get_formatter
            self.formatter = get_formatter()
            console.print("✅ Formatter loaded")
        except Exception as e:
            console.print(f"❌ Formatter error: {e}")
            class DummyFormatter:
                colors = {'warning': 'yellow', 'error': 'red', 'success': 'green', 'info': 'blue', 'primary': 'cyan', 'dim': 'dim'}
            self.formatter = DummyFormatter()
        
        # Data Loader
        try:
            from wrang.core.loader import FastDataLoader
            self.loader = FastDataLoader()
            console.print("✅ Data loader loaded")
        except Exception as e:
            console.print(f"❌ Data loader error: {e}")
            self.loader = None
        
        # Menu Handler - This is the critical one
        try:
            from wrang.cli.menus import MenuHandler
            self.menu_handler = MenuHandler()
            console.print("✅ Menu handler loaded")
        except Exception as e:
            console.print(f"❌ Menu handler error: {e}")
            console.print(f"[red]Detailed error:[/red]")
            console.print(f"[dim]{traceback.format_exc()}[/dim]")
            self.menu_handler = None
        
        # Setup signal handlers for graceful shutdown
        try:
            signal.signal(signal.SIGINT, self._signal_handler)
            signal.signal(signal.SIGTERM, self._signal_handler)
        except OSError:
            pass
    
    def run(self, args: Optional[argparse.Namespace] = None) -> int:
        """Main application entry point"""
        try:
            # Handle version display
            if args and getattr(args, 'version', False):
                self._show_version()
                return 0
            
            # Handle help display
            if args and getattr(args, 'help_topic', None):
                self._show_help(args.help_topic)
                return 0
            
            # --compare: dataset diff mode (no file required, two files given)
            if args and getattr(args, 'compare', None):
                return self._run_compare(args.compare, args)

            # --sql: DuckDB query mode
            if args and getattr(args, 'sql', None):
                if not getattr(args, 'file', None):
                    self._output_error("--sql requires a FILE argument", args)
                    return 2
                return self._run_sql(args.file, args.sql, args)

            # Handle direct file loading if provided
            if args and getattr(args, 'file', None):
                return self._run_with_file(args.file, args)

            # Run interactive mode
            return self._run_interactive_mode()
            
        except KeyboardInterrupt:
            console.print(f"\n[yellow]Application interrupted by user[/yellow]")
            return 0
        except Exception as e:
            self._handle_fatal_error(e)
            return 1
    
    def _run_interactive_mode(self) -> int:
        """Run the main interactive menu system"""
        try:
            if self.menu_handler:
                console.print("[green]Starting wrang...[/green]")
                self.menu_handler.run_main_menu()
                return 0
            else:
                console.print("[yellow]⚠️ Rich interface unavailable, using basic mode[/yellow]")
                console.print("This usually means there's an import error with the core modules.")
                console.print("Try running: pip install -e . --force-reinstall")
                console.print()
                
                # Fallback to basic interface
                return self._run_basic_interface()
            
        except Exception as e:
            self._handle_fatal_error(e)
            return 1
    
    def _run_basic_interface(self) -> int:
        """Basic fallback interface"""
        console.print("[cyan]wrang - Basic Mode[/cyan]")
        console.print("Available commands:")
        console.print("  help    - Show available commands")
        console.print("  version - Show version information")
        console.print("  debug   - Show debug information")
        console.print("  q       - Quit")
        
        while True:
            try:
                user_input = input("\nwrang> ").strip().lower()
                
                if user_input in ['q', 'quit', 'exit']:
                    break
                elif user_input == 'help':
                    console.print("Available commands: help, version, debug, q")
                elif user_input == 'version':
                    self._show_version()
                elif user_input == 'debug':
                    self._show_debug_info()
                elif user_input.startswith('load '):
                    filename = user_input[5:].strip()
                    self._try_load_file(filename)
                else:
                    console.print("Unknown command. Type 'help' for available commands.")
                    
            except (EOFError, KeyboardInterrupt):
                break
        
        console.print("[green]Goodbye![/green]")
        return 0
    
    def _try_load_file(self, filename: str) -> None:
        """Try to load a file in basic mode"""
        if not self.loader:
            console.print("[red]❌ Data loader not available[/red]")
            return
        
        try:
            from pathlib import Path
            file_path = Path(filename)
            
            if not file_path.exists():
                console.print(f"[red]❌ File '{filename}' not found[/red]")
                return
            
            console.print(f"[blue]Loading {filename}...[/blue]")
            df = self.loader.load(file_path)
            
            console.print(f"[green]✅ Loaded {len(df)} rows × {len(df.columns)} columns[/green]")
            console.print(f"Columns: {', '.join(df.columns[:5])}{'...' if len(df.columns) > 5 else ''}")
            
        except Exception as e:
            console.print(f"[red]❌ Error loading file: {e}[/red]")
    
    def _show_debug_info(self) -> None:
        """Show debugging information"""
        console.print("[cyan]wrang Debug Information[/cyan]")
        console.print(f"Python version: {sys.version}")
        console.print(f"Python path: {sys.path[0]}")
        
        # Test core dependencies
        deps = ['polars', 'rich', 'numpy', 'pandas', 'scikit_learn', 'scipy']
        console.print("\nDependency Status:")
        for dep in deps:
            try:
                __import__(dep)
                console.print(f"  ✅ {dep}")
            except ImportError:
                console.print(f"  ❌ {dep}")
        
        # Test RIDE modules step by step
        console.print("\nRIDE Module Status:")
        
        # Test individual core modules
        modules_to_test = [
            'wrang.config',
            'wrang.utils.exceptions', 
            'wrang.core.loader',
            'wrang.core.inspector',
            'wrang.core.explorer',
            'wrang.core.cleaner',
            'wrang.core.transformer',
            'wrang.cli.formatters',
            'wrang.cli.menus'
        ]
        
        for module in modules_to_test:
            try:
                __import__(module)
                console.print(f"  ✅ {module}")
            except ImportError as e:
                console.print(f"  ❌ {module}: {e}")
        
        # Test specific classes
        console.print("\nClass Import Status:")
        class_tests = [
            ('wrang.core.inspector', 'DataInspector'),
            ('wrang.core.explorer', 'DataExplorer'),
            ('wrang.core.cleaner', 'DataCleaner'),
            ('wrang.core.transformer', 'DataTransformer'),
            ('wrang.cli.menus', 'MenuHandler')
        ]
        
        for module_name, class_name in class_tests:
            try:
                module = __import__(module_name, fromlist=[class_name])
                getattr(module, class_name)
                console.print(f"  ✅ {module_name}.{class_name}")
            except Exception as e:
                console.print(f"  ❌ {module_name}.{class_name}: {e}")
    
    def _run_with_file(self, file_path: str, args: argparse.Namespace) -> int:
        """Run RIDE with a pre-loaded file"""
        try:
            file_path = Path(file_path)

            if not file_path.exists():
                self._output_error(f"File '{file_path}' not found", args)
                return 2

            if not self.loader:
                self._output_error("Data loader not available", args)
                return 1

            chunk_size = getattr(args, 'chunk_size', None)
            output_format = getattr(args, 'output_format', 'text')

            if chunk_size:
                # Streaming mode: show progress only, no interactive menu
                return self._run_streaming(file_path, chunk_size, args)

            console.print(f"[blue]Loading {file_path.name}...[/blue]")
            df = self.loader.load(file_path)

            # Handle additional command line options
            if getattr(args, 'inspect', False):
                return self._run_command_line_inspect(df, output_format)
            elif getattr(args, 'profile', False):
                return self._run_html_profile(df, file_path, args)
            elif getattr(args, 'export', None):
                return self._run_command_line_export(df, args.export, getattr(args, 'format', None))
            else:
                # Show file info and continue to interactive mode
                console.print(f"[green]File loaded successfully: {len(df)} rows, {len(df.columns)} columns[/green]")
                input("\nPress Enter to continue to interactive mode...")
                return self._run_interactive_mode()

        except Exception as e:
            self._output_error(str(e), args)
            return 1

    # ------------------------------------------------------------------
    # DuckDB SQL mode
    # ------------------------------------------------------------------

    def _run_sql(self, file_path: str, query: str, args: argparse.Namespace) -> int:
        """Execute a DuckDB SQL query against a file and print results."""
        try:
            import duckdb  # type: ignore
        except ImportError:
            self._output_error(
                "duckdb is required for --sql mode. Install with: pip install duckdb", args
            )
            return 1

        output_format = getattr(args, 'output_format', 'text')
        file_path = Path(file_path)

        if not file_path.exists():
            self._output_error(f"File '{file_path}' not found", args)
            return 2

        try:
            con = duckdb.connect(database=":memory:")
            ext = file_path.suffix.lower()

            if ext == ".parquet":
                con.execute(f"CREATE VIEW data AS SELECT * FROM read_parquet('{file_path}')")
            elif ext in {".xlsx", ".xls"}:
                # load via polars then register
                import polars as pl
                df = pl.read_excel(file_path)
                con.register("data", df.to_arrow())
            elif ext == ".json":
                con.execute(f"CREATE VIEW data AS SELECT * FROM read_json_auto('{file_path}')")
            else:
                # Default: CSV
                con.execute(
                    f"CREATE VIEW data AS SELECT * FROM read_csv_auto('{file_path}', header=true)"
                )

            result = con.execute(query).pl()  # returns Polars DataFrame

            if output_format == "json":
                import json as _json
                print(_json.dumps(result.to_dicts(), default=str))
            else:
                try:
                    from rich.table import Table
                    from rich import box

                    tbl = Table(box=box.SIMPLE_HEAD, show_lines=False)
                    for col in result.columns:
                        tbl.add_column(col, style="cyan", no_wrap=True)
                    for row in result.iter_rows():
                        tbl.add_row(*[str(v) if v is not None else "" for v in row])
                    console.print(tbl)
                    console.print(f"[dim]{len(result)} row(s)[/dim]")
                except Exception:
                    print(result)

            return 0

        except Exception as e:
            self._output_error(f"SQL error: {e}", args)
            return 1

    # ------------------------------------------------------------------
    # Dataset diff / compare
    # ------------------------------------------------------------------

    def _run_compare(self, files: List[str], args: argparse.Namespace) -> int:
        """Compare two datasets and report differences."""
        if len(files) != 2:
            self._output_error("--compare requires exactly two file paths", args)
            return 2

        import polars as pl

        output_format = getattr(args, 'output_format', 'text')
        paths = [Path(f) for f in files]

        for p in paths:
            if not p.exists():
                self._output_error(f"File '{p}' not found", args)
                return 2

        try:
            dfs = [self.loader.load(p) for p in paths]
        except Exception as e:
            self._output_error(f"Failed to load files: {e}", args)
            return 1

        df_a, df_b = dfs
        diff = self._compute_diff(df_a, df_b, paths[0].name, paths[1].name)

        if output_format == "json":
            import json as _json
            print(_json.dumps(diff, default=str))
            return 0 if diff["schema_match"] and diff["shape_match"] else 1

        self._display_diff(diff, paths)
        return 0 if diff["schema_match"] and diff["shape_match"] else 1

    def _compute_diff(
        self,
        df_a: "pl.DataFrame",
        df_b: "pl.DataFrame",
        name_a: str,
        name_b: str,
    ) -> dict:
        import polars as pl

        cols_a = set(df_a.columns)
        cols_b = set(df_b.columns)

        schema_a = {c: str(df_a[c].dtype) for c in df_a.columns}
        schema_b = {c: str(df_b[c].dtype) for c in df_b.columns}

        only_in_a = sorted(cols_a - cols_b)
        only_in_b = sorted(cols_b - cols_a)
        shared = sorted(cols_a & cols_b)

        dtype_changes = {
            c: {"in_a": schema_a[c], "in_b": schema_b[c]}
            for c in shared
            if schema_a[c] != schema_b[c]
        }

        numeric_stats: dict = {}
        for c in shared:
            if str(df_a[c].dtype) in {"Float64", "Float32", "Int64", "Int32", "Int16", "Int8",
                                       "UInt64", "UInt32", "UInt16", "UInt8"}:
                try:
                    mean_a = float(df_a[c].drop_nulls().mean() or 0)
                    mean_b = float(df_b[c].drop_nulls().mean() or 0)
                    numeric_stats[c] = {
                        "mean_a": round(mean_a, 4),
                        "mean_b": round(mean_b, 4),
                        "mean_delta": round(mean_b - mean_a, 4),
                        "null_pct_a": round(df_a[c].null_count() / len(df_a) * 100, 2),
                        "null_pct_b": round(df_b[c].null_count() / len(df_b) * 100, 2),
                    }
                except Exception:
                    pass

        return {
            "file_a": name_a,
            "file_b": name_b,
            "shape_a": list(df_a.shape),
            "shape_b": list(df_b.shape),
            "shape_match": df_a.shape == df_b.shape,
            "schema_match": schema_a == schema_b,
            "columns_only_in_a": only_in_a,
            "columns_only_in_b": only_in_b,
            "dtype_changes": dtype_changes,
            "numeric_stats": numeric_stats,
        }

    def _display_diff(self, diff: dict, paths: list) -> None:
        from rich.table import Table
        from rich import box

        console.print(f"\n[bold cyan]Dataset Comparison[/bold cyan]")
        console.print(f"  A: [green]{diff['file_a']}[/green]  shape={diff['shape_a']}")
        console.print(f"  B: [green]{diff['file_b']}[/green]  shape={diff['shape_b']}")

        shape_icon = "✅" if diff["shape_match"] else "❌"
        schema_icon = "✅" if diff["schema_match"] else "❌"
        console.print(f"\n  Shape match: {shape_icon}   Schema match: {schema_icon}\n")

        if diff["columns_only_in_a"]:
            console.print(f"  [yellow]Columns only in A:[/yellow] {', '.join(diff['columns_only_in_a'])}")
        if diff["columns_only_in_b"]:
            console.print(f"  [yellow]Columns only in B:[/yellow] {', '.join(diff['columns_only_in_b'])}")

        if diff["dtype_changes"]:
            tbl = Table(title="Dtype Changes", box=box.SIMPLE_HEAD)
            tbl.add_column("Column", style="cyan")
            tbl.add_column("Type in A", style="yellow")
            tbl.add_column("Type in B", style="magenta")
            for col, chg in diff["dtype_changes"].items():
                tbl.add_row(col, chg["in_a"], chg["in_b"])
            console.print(tbl)

        if diff["numeric_stats"]:
            tbl = Table(title="Numeric Column Stats", box=box.SIMPLE_HEAD)
            tbl.add_column("Column", style="cyan")
            tbl.add_column("Mean A", justify="right")
            tbl.add_column("Mean B", justify="right")
            tbl.add_column("Delta", justify="right")
            tbl.add_column("Null% A", justify="right")
            tbl.add_column("Null% B", justify="right")
            for col, s in diff["numeric_stats"].items():
                delta_style = "red" if abs(s["mean_delta"]) > 0 else "green"
                tbl.add_row(
                    col,
                    str(s["mean_a"]),
                    str(s["mean_b"]),
                    f"[{delta_style}]{s['mean_delta']:+}[/{delta_style}]",
                    f"{s['null_pct_a']}%",
                    f"{s['null_pct_b']}%",
                )
            console.print(tbl)

    # ------------------------------------------------------------------
    # HTML profile report
    # ------------------------------------------------------------------

    def _run_html_profile(self, df, file_path: Path, args: argparse.Namespace) -> int:
        try:
            from wrang.viz.export_utils import generate_html_report
            out = file_path.with_suffix(".html")
            if getattr(args, 'export', None):
                out = Path(args.export)
            result = generate_html_report(df, output_path=out, title=f"wrang Profile — {file_path.name}")
            console.print(f"[green]✅  HTML profile report saved to: {result}[/green]")
            return 0
        except Exception as e:
            self._output_error(f"Profile generation failed: {e}", args)
            return 1

    # ------------------------------------------------------------------
    # Streaming mode
    # ------------------------------------------------------------------

    def _run_streaming(self, file_path: Path, chunk_size: int, args: argparse.Namespace) -> int:
        """Process file in chunks and report per-chunk shape."""
        try:
            total_rows = 0
            chunk_n = 0
            for chunk in self.loader.stream_chunks(file_path, chunk_size=chunk_size):
                chunk_n += 1
                total_rows += len(chunk)
                if getattr(args, 'verbose', False):
                    console.print(f"  Chunk {chunk_n}: {len(chunk)} rows × {len(chunk.columns)} cols")

            console.print(
                f"[green]✅  Streamed {total_rows:,} rows in {chunk_n} chunk(s) "
                f"(chunk_size={chunk_size:,})[/green]"
            )
            return 0
        except Exception as e:
            self._output_error(f"Streaming error: {e}", args)
            return 1

    # ------------------------------------------------------------------
    # Output helpers (text vs JSON)
    # ------------------------------------------------------------------

    def _output_error(self, message: str, args=None) -> None:
        """Emit an error in the requested output format."""
        fmt = getattr(args, 'output_format', 'text') if args else 'text'
        if fmt == 'json':
            import json as _json
            print(_json.dumps({"error": message}))
        else:
            console.print(f"[red]Error: {message}[/red]")
    
    def _run_command_line_inspect(self, df, output_format: str = 'text') -> int:
        """Run inspection from command line"""
        info = {
            "rows": len(df),
            "columns": len(df.columns),
            "column_names": df.columns,
            "dtypes": {c: str(df[c].dtype) for c in df.columns},
            "missing_pct": {
                c: round(df[c].null_count() / len(df) * 100, 2) for c in df.columns
            },
        }

        if output_format == 'json':
            import json as _json
            print(_json.dumps(info, default=str))
            return 0

        try:
            from wrang.core.inspector import DataInspector
            inspector = DataInspector(df)
            inspector.display_overview()
            return 0
        except Exception:
            console.print(f"[green]Dataset Overview:[/green]")
            console.print(f"  Rows: {info['rows']:,}")
            console.print(f"  Columns: {info['columns']}")
            console.print(f"  Column names: {', '.join(info['column_names'])}")
            return 0
    
    def _run_command_line_export(self, df, output_path: str, format_type: Optional[str]) -> int:
        """Run export from command line"""
        try:
            output_path = Path(output_path)
            
            # Determine format from extension if not specified
            if not format_type:
                format_map = {
                    '.csv': 'csv',
                    '.xlsx': 'excel',
                    '.parquet': 'parquet', 
                    '.json': 'json'
                }
                format_type = format_map.get(output_path.suffix.lower(), 'csv')
            
            # Simple CSV export fallback
            if format_type == 'csv' or output_path.suffix.lower() == '.csv':
                df.write_csv(output_path)
                console.print(f"[green]Data exported to {output_path}[/green]")
                return 0
            else:
                console.print(f"[yellow]Export format {format_type} not implemented in basic mode[/yellow]")
                console.print(f"[yellow]Exporting as CSV instead...[/yellow]")
                csv_path = output_path.with_suffix('.csv')
                df.write_csv(csv_path)
                console.print(f"[green]Data exported to {csv_path}[/green]")
                return 0
            
        except Exception as e:
            console.print(f"[red]Error during export: {e}[/red]")
            return 1
    
    def _show_version(self) -> None:
        """Display version information"""
        try:
            from ride import __version__
            version = __version__
        except ImportError:
            version = "0.1.0"
        
        version_info = f"""wrang v{version}

lightning-fast data wrangling toolkit

Features:
• Fast data processing with Polars
• Interactive CLI with beautiful formatting  
• Advanced data cleaning and transformation
• Statistical analysis and visualization
• Multiple file format support (CSV, Excel, Parquet, JSON)

System Information:
• Python: {sys.version.split()[0]}
• Platform: {sys.platform}
• Terminal: {os.environ.get('TERM', 'unknown')}

For more information, visit: https://github.com/sudhanshumukherjeexx/wrang"""
        
        console.print(version_info)
    
    def _show_help(self, topic: str) -> None:
        """Display help for a specific topic"""
        if topic == 'usage':
            self._help_usage()
        elif topic == 'examples':
            self._help_examples()
        elif topic == 'formats':
            self._help_formats()
        elif topic == 'config':
            self._help_config()
        else:
            console.print(f"[red]Unknown help topic: {topic}[/red]")
            console.print("Available topics: usage, examples, formats, config")
    
    def _help_usage(self) -> None:
        """Display usage help"""
        usage_text = """wrang Usage

Interactive Mode:
  wrang                             Start interactive mode
  wrang [FILE]                      Start with pre-loaded file

Non-interactive Mode:
  wrang [FILE] --inspect            Quick data inspection (text or JSON)
  wrang [FILE] --profile            Generate self-contained HTML report
  wrang [FILE] --sql "SELECT ..."   Execute a DuckDB SQL query against FILE
  wrang [FILE] --export OUTPUT      Export data to file
  wrang [FILE] --chunk-size N       Stream FILE in N-row chunks
  wrang --compare FILE_A FILE_B     Diff two datasets

Output Flags:
  --output-format json              Machine-readable JSON output (CI-friendly)
  --output-format text              Human-readable text output (default)

Other Options:
  --inspect                         Show data overview and exit
  --export PATH                     Export data to specified path
  --format FORMAT                   Export format (csv, excel, parquet, json)
  --version                         Show version information
  --help-topic TOPIC                Show help for specific topic
  --debug / --verbose               Debug / verbose output

Help Topics:
  usage, examples, formats, config

Interactive Navigation:
  • Use numbers to select menu options
  • Press 'q' to quit from any menu"""

        console.print(usage_text)
    
    def _help_examples(self) -> None:
        """Display usage examples"""
        examples_text = """wrang Examples

Basic Usage:
  python -m wrang                                    # Start interactive mode
  python -m wrang data.csv                          # Load file and start interactive
  python -m wrang data.xlsx --inspect               # Quick inspection of Excel file
  
Data Processing:
  python -m wrang sales_data.csv                    # Load sales data
  → Choose "2" for inspection
  → Choose "4" for cleaning  
  → Choose "Quick Clean - ML Ready"
  → Choose "7" to export processed data

File Format Examples:
  python -m wrang customer_data.csv                 # CSV file
  python -m wrang financial_report.xlsx             # Excel file  
  python -m wrang large_dataset.parquet             # Parquet file
  python -m wrang api_response.json                 # JSON file

Command Line Export:
  python -m wrang raw_data.csv --export clean_data.csv
  python -m wrang data.xlsx --export output.parquet --format parquet"""
        
        console.print(examples_text)
    
    def _help_formats(self) -> None:
        """Display supported file formats"""
        formats_text = """Supported File Formats

Input Formats:
  📄 CSV (.csv)
     • Automatic delimiter detection
     • Multiple encoding support (UTF-8, Latin-1)
     • Header detection
     • Large file streaming support

  📊 Excel (.xlsx, .xls)  
     • Multiple worksheet support
     • Automatic data type detection
     • Formula value extraction
     • Compatible with Excel 2007+

  🗂️ Parquet (.parquet)
     • Ultra-fast columnar format
     • Excellent compression
     • Preserves data types perfectly
     • Optimized for analytics

  🔗 JSON (.json, .jsonl)
     • Nested structure flattening
     • JSON Lines support
     • Automatic schema inference
     • Web API data compatible

Output Formats:
  All input formats plus:
  • Automatic format detection from extension
  • Compression options for Parquet
  • Custom delimiters for CSV
  • Multiple sheets for Excel"""
        
        console.print(formats_text)
    
    def _help_config(self) -> None:
        """Display configuration help"""
        config_text = """wrang Configuration

Configuration Location:
  Config Directory: ~/.wrang/
  Config File: ~/.wrang/config.json

Key Settings:
  • Memory limit for large files
  • Default file encodings
  • Visualization parameters
  • Data processing thresholds
  • Random seeds for reproducibility

Accessing Settings:
  In interactive mode: Main Menu → "8. Settings"
  
Settings Categories:
  📊 Memory Settings - Control memory usage and chunk sizes
  🎨 Visualization - Plot dimensions and display options  
  📁 File Formats - Default encodings and delimiters
  ⚡ Performance - Parallel processing and optimization"""
        
        console.print(config_text)
    
    def _signal_handler(self, signum, frame) -> None:
        """Handle system signals for graceful shutdown"""
        console.print(f"\n[yellow]Received signal {signum}. Shutting down gracefully...[/yellow]")
        sys.exit(0)
    
    def _handle_fatal_error(self, error: Exception) -> None:
        """Handle fatal application errors"""
        console.print(f"\n[red]💥 Fatal Error[/red]")
        console.print(f"[red]An unexpected error occurred: {error}[/red]")
        
        if getattr(self.config, 'debug', False):
            console.print(f"\n[dim]Debug information:[/dim]")
            console.print_exception()
        else:
            console.print(f"\n[dim]Run with --debug for detailed error information[/dim]")


def create_argument_parser() -> argparse.ArgumentParser:
    """Create and configure the command line argument parser"""
    parser = argparse.ArgumentParser(
        prog='wrang',
        description='wrang: lightning-fast data wrangling toolkit',
        epilog='For more information, visit: https://github.com/sudhanshumukherjeexx/wrang',
        formatter_class=argparse.RawDescriptionHelpFormatter
    )
    
    # Positional arguments
    parser.add_argument(
        'file',
        nargs='?',
        help='Dataset file to load (CSV, Excel, Parquet, JSON)'
    )
    
    # Action arguments
    parser.add_argument(
        '--inspect',
        action='store_true',
        help='Display dataset overview and exit'
    )
    
    parser.add_argument(
        '--export',
        metavar='PATH',
        help='Export data to specified path'
    )
    
    parser.add_argument(
        '--format',
        choices=['csv', 'excel', 'parquet', 'json'],
        help='Export format (auto-detected from extension if not specified)'
    )
    
    # Information arguments
    parser.add_argument(
        '--version',
        action='store_true',
        help='Show version information and exit'
    )
    
    parser.add_argument(
        '--help-topic',
        choices=['usage', 'examples', 'formats', 'config'],
        help='Show help for specific topic'
    )
    
    # SQL mode (DuckDB)
    parser.add_argument(
        '--sql',
        metavar='QUERY',
        help='Execute a DuckDB SQL query against FILE (table name: "data") and print results',
    )

    # Dataset comparison
    parser.add_argument(
        '--compare',
        nargs=2,
        metavar=('FILE_A', 'FILE_B'),
        help='Compare two datasets and report structural / statistical differences',
    )

    # Streaming / chunked processing
    parser.add_argument(
        '--chunk-size',
        type=int,
        metavar='N',
        dest='chunk_size',
        help='Process FILE in chunks of N rows (streaming mode)',
    )

    # HTML profile report
    parser.add_argument(
        '--profile',
        action='store_true',
        help='Generate a self-contained HTML data-profile report for FILE and exit',
    )

    # Output format (CI-ready JSON)
    parser.add_argument(
        '--output-format',
        choices=['text', 'json'],
        default='text',
        dest='output_format',
        help='Output format: "text" (default) or "json" for machine-readable output',
    )

    # Development arguments
    parser.add_argument(
        '--debug',
        action='store_true',
        help='Enable debug mode with detailed error information'
    )

    parser.add_argument(
        '--verbose',
        action='store_true',
        help='Enable verbose output'
    )

    return parser


def main(argv: Optional[List[str]] = None) -> int:
    """
    Main entry point for the RIDE CLI application
    
    Args:
        argv: Command line arguments (defaults to sys.argv)
        
    Returns:
        Exit code (0 for success, 1 for error)
    """
    try:
        # Parse command line arguments
        parser = create_argument_parser()
        args = parser.parse_args(argv)
        
        # Update configuration with command line options
        if getattr(args, 'debug', False):
            try:
                from wrang.config import update_config
                update_config(debug=True, verbose=True)
            except ImportError:
                pass
        elif getattr(args, 'verbose', False):
            try:
                from wrang.config import update_config
                update_config(verbose=True)
            except ImportError:
                pass
        
        # Create and run the CLI application
        app = RideCLI()
        return app.run(args)
        
    except KeyboardInterrupt:
        console.print(f"\n[yellow]Operation cancelled by user[/yellow]")
        return 0
    except Exception as e:
        console.print(f"[red]Failed to start wrang: {e}[/red]")
        if argv and '--debug' in argv:
            traceback.print_exc()
        return 1


def check_dependencies() -> bool:
    """Check if all required dependencies are available"""
    required_packages = [
        'polars', 'rich', 'numpy'  # Core required packages
    ]
    
    missing_packages = []
    
    for package in required_packages:
        try:
            __import__(package)
        except ImportError:
            missing_packages.append(package)
    
    if missing_packages:
        console.print(f"[red]Missing required packages: {', '.join(missing_packages)}[/red]")
        console.print("[yellow]Please install with: pip install polars rich numpy[/yellow]")
        return False
    
    return True


if __name__ == '__main__':
    # Quick dependency check
    if not check_dependencies():
        sys.exit(1)
    
    # Run the application
    exit_code = main()
    sys.exit(exit_code)