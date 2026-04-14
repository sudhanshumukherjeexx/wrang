#!/usr/bin/env python3
"""
wrang — lightning-fast data wrangling toolkit for the terminal

A modern, efficient toolkit for data analysis and preparation that provides:
• 10x faster data processing with Polars
• Interactive CLI with Rich formatting
• Advanced data cleaning and transformation
• Statistical analysis and visualization
• Support for multiple file formats
• No dependency bloat or CUDA requirements
"""

import os

# Version information
__version__ = "0.2.2"
__author__ = "Sudhanshu Mukherjee"
__email__ = "sudhanshumukherjeexx@gmail.com"
__url__ = "https://github.com/sudhanshumukherjeexx/wrang"
__license__ = "MIT"
__description__ = "wrang: lightning-fast data wrangling toolkit for the terminal"

# Feature availability flags (populated lazily)
_CORE_AVAILABLE = False
_HAS_POLARS = False
_HAS_RICH = False
_HAS_PLOTEXT = False

# --- Lazy public API -------------------------------------------------------
# Symbols are imported on first attribute access so that `import wrang` is
# fast and never fails just because an optional dependency is missing.

def __getattr__(name: str):
    """Lazy import of public API symbols and legacy deprecation warnings."""

    _public = {
        'FastDataLoader', 'DataSaver', 'load_data', 'save_data',
        'DataInspector', 'inspect_data',
        'DataExplorer', 'explore_data',
        'DataCleaner', 'BatchCleaner', 'clean_data', 'quick_clean',
        'DataTransformer', 'TransformationPipeline',
        'transform_data', 'create_pipeline', 'quick_transform',
        'DataValidator', 'DataSchema', 'ColumnSchema', 'ValidationResult',
        'validate_data', 'infer_schema',
        'get_config', 'update_config', 'reset_config',
        'get_formatter',
    }

    _legacy_mappings = {
        'AutoMLProcessor': 'DataTransformer',
        'automl_processor': 'core.transformer',
        'utils': 'core modules (loader, inspector, explorer, cleaner, transformer)',
    }

    if name in _legacy_mappings:
        import warnings
        warnings.warn(
            f"'{name}' has been deprecated. Please use: {_legacy_mappings[name]}",
            DeprecationWarning,
            stacklevel=2,
        )
        raise AttributeError(f"'{name}' is no longer available. Use: {_legacy_mappings[name]}")

    if name == 'Prepup':
        return DeprecatedPrepup

    if name in _public:
        _load_core()
        import wrang._public_api as _api  # populated by _load_core()
        if hasattr(_api, name):
            return getattr(_api, name)

    raise AttributeError(f"module 'wrang' has no attribute '{name}'")


def _load_core() -> bool:
    """Import core modules once and cache results in the global flags."""
    global _CORE_AVAILABLE, _HAS_POLARS, _HAS_RICH, _HAS_PLOTEXT

    if _CORE_AVAILABLE:
        return True

    try:
        import wrang._public_api  # noqa: F401 – side-effect: populates the module
        _CORE_AVAILABLE = True
    except ImportError as exc:
        import warnings
        warnings.warn(
            f"Some wrang modules could not be imported: {exc}\n"
            "Some functionality may be limited. Please reinstall with: pip install wrang",
            ImportWarning,
            stacklevel=3,
        )
        _CORE_AVAILABLE = False

    try:
        import polars  # noqa: F401
        _HAS_POLARS = True
    except ImportError:
        pass

    try:
        import rich  # noqa: F401
        _HAS_RICH = True
    except ImportError:
        pass

    try:
        import plotext  # noqa: F401
        _HAS_PLOTEXT = True
    except ImportError:
        pass

    return _CORE_AVAILABLE


# Backward compatibility — raises NotImplementedError with migration guide
class DeprecatedPrepup:
    """Deprecated Prepup class — kept for import compatibility only."""

    def __init__(self, *args, **kwargs):
        import warnings
        warnings.warn(
            "The 'Prepup' class is deprecated and will be removed in v0.2.0. "
            "Please use the new modular components:\n"
            "  • DataCleaner for data cleaning\n"
            "  • DataTransformer for feature transformation\n"
            "  • DataInspector for data inspection\n"
            "  • DataExplorer for statistical analysis\n"
            "Or use the interactive CLI with 'wrang' command.",
            DeprecationWarning,
            stacklevel=2,
        )
        raise NotImplementedError(
            "Prepup class has been replaced. "
            "Use the new modular components or the 'wrang' CLI."
        )


Prepup = DeprecatedPrepup

# Package exports
__all__ = [
    '__version__', '__author__', '__email__', '__url__', '__license__', '__description__',
    'FastDataLoader', 'DataSaver',
    'DataInspector', 'DataExplorer',
    'DataCleaner', 'BatchCleaner',
    'DataTransformer', 'TransformationPipeline',
    'DataValidator', 'DataSchema', 'ColumnSchema', 'ValidationResult',
    'load_data', 'save_data',
    'inspect_data', 'explore_data',
    'clean_data', 'quick_clean',
    'transform_data', 'create_pipeline', 'quick_transform',
    'validate_data', 'infer_schema',
    'get_config', 'update_config', 'reset_config',
    'get_formatter',
    'Prepup',
]


# Show welcome message once for brand-new installs (suppressed via env var)
def _show_welcome_message() -> None:
    config_dir = os.path.expanduser("~/.wrang")
    if not os.path.exists(config_dir):
        print(f"\nWelcome to wrang v{__version__}!")
        print("Run 'wrang' to start the interactive data wrangling interface.")
        print("For examples run: wrang --help-topic examples\n")


if not os.environ.get('WRANG_NO_WELCOME'):
    try:
        _show_welcome_message()
    except Exception:
        pass
