#!/usr/bin/env python3
"""
wrang — Main Entry Point
Lightning-fast data wrangling toolkit with beautiful terminal interface
"""

import sys
import os
from pathlib import Path
from typing import Optional, List

# Add the current directory to Python path for development
if __name__ == '__main__':
    current_dir = Path(__file__).parent
    if str(current_dir) not in sys.path:
        sys.path.insert(0, str(current_dir))

# Import the CLI interface
try:
    from wrang.cli.interface import main as cli_main, check_dependencies
    from wrang.cli.formatters import get_formatter, clear_screen
    from wrang import __version__
except ImportError as e:
    print(f"Error importing wrang modules: {e}")
    print("Please ensure wrang is properly installed with: pip install wrang")
    sys.exit(1)


def main(argv: Optional[List[str]] = None) -> int:
    """
    Main entry point for wrang CLI

    Args:
        argv: Command line arguments (defaults to sys.argv)

    Returns:
        Exit code (0 for success, 1 for error)
    """

    # Quick dependency check
    if not check_dependencies():
        print("Dependencies missing. Please install with: pip install wrang")
        return 1

    # Handle special debug argument early
    if argv is None:
        argv = sys.argv[1:]

    if '--system-info' in argv:
        print_system_info()
        return 0

    # Delegate to the CLI interface
    try:
        return cli_main(argv)
    except Exception as e:
        print(f"Fatal error in wrang CLI: {e}")
        if '--debug' in argv:
            import traceback
            traceback.print_exc()
        return 1


# Entry point for pip-installed command
def cli_entry_point():
    """Entry point for the 'wrang' command"""
    exit_code = main()
    sys.exit(exit_code)


# Support for direct script execution
if __name__ == '__main__':
    try:
        exit_code = main()
        sys.exit(exit_code)
    except KeyboardInterrupt:
        print("\nOperation cancelled by user")
        sys.exit(0)
    except Exception as e:
        print(f"Fatal error: {e}")

        if '--debug' in sys.argv:
            import traceback
            traceback.print_exc()
        else:
            print("   Run with --debug for detailed error information")

        print("\nPlease report issues at: https://github.com/sudhanshumukherjeexx/wrang/issues")
        sys.exit(1)


def get_version() -> str:
    """Get wrang version string"""
    try:
        return __version__
    except Exception:
        return "unknown"


def print_system_info() -> None:
    """Print system information for debugging"""
    print("wrang System Information")
    print("=" * 30)
    print(f"Version: {get_version()}")
    print(f"Python: {sys.version}")
    print(f"Platform: {sys.platform}")
    print(f"Python Path: {sys.path[0]}")

    # Check dependencies
    deps = ['polars', 'rich', 'numpy', 'scikit_learn']
    print("\nDependencies:")
    for dep in deps:
        try:
            __import__(dep)
            print(f"  OK  {dep}")
        except ImportError:
            print(f"  MISSING  {dep}")


__doc__ = """
wrang — data wrangling toolkit

Usage:
    wrang                    # Start interactive mode
    wrang data.csv           # Load file and start interactive mode
    wrang data.csv --inspect # Quick data inspection
    wrang data.csv --sql "SELECT * FROM data LIMIT 10"
    wrang --compare a.csv b.csv
    wrang --help-topic usage # Show detailed usage guide

For more information:
    https://github.com/sudhanshumukherjeexx/wrang
"""
