#!/usr/bin/env python3
"""
ride/__main__.py
Main entry point for wrang CLI when called with python -m ride
"""

import sys
import os
from pathlib import Path

# Add the parent directory to the path so we can import from cli
current_dir = Path(__file__).parent
parent_dir = current_dir.parent
if str(parent_dir) not in sys.path:
    sys.path.insert(0, str(parent_dir))

def main():
    """Main entry point for python -m ride"""
    try:
        from wrang.cli.interface import main as cli_main
        return cli_main()
    except ImportError as e:
        print(f"Error importing CLI interface: {e}")
        print("Make sure all wrang modules are properly installed.")
        return 1
    except Exception as e:
        print(f"Error starting wrang CLI: {e}")
        return 1

if __name__ == "__main__":
    sys.exit(main())