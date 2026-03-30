#!/usr/bin/env python3
"""
ride/_public_api.py
Eagerly imports all public symbols so __init__.py can stay lazy.
This module is only loaded when a public symbol is first accessed.
"""

from wrang.core.loader import FastDataLoader, DataSaver, load_data, save_data
from wrang.core.inspector import DataInspector, inspect_data
from wrang.core.explorer import DataExplorer, explore_data
from wrang.core.cleaner import DataCleaner, BatchCleaner, clean_data, quick_clean
from wrang.core.transformer import (
    DataTransformer, TransformationPipeline,
    transform_data, create_pipeline, quick_transform,
)
from wrang.core.validator import (
    DataValidator, DataSchema, ColumnSchema, ValidationResult,
    validate_data, infer_schema,
)
from wrang.config import get_config, update_config, reset_config
from wrang.cli.formatters import get_formatter

__all__ = [
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
]
