#!/usr/bin/env python3
"""
wrang Configuration Management
Centralized configuration for all wrang operations
"""

import os
from dataclasses import dataclass, field
from pathlib import Path
from typing import Dict, Any, Optional, List
import json
from enum import Enum


class FileFormat(Enum):
    """Supported file formats"""
    CSV = "csv"
    EXCEL = "xlsx"
    PARQUET = "parquet"
    JSON = "json"


class ImputationStrategy(Enum):
    """Missing value imputation strategies"""
    DROP = "drop"
    MEAN = "mean"
    MEDIAN = "median"
    MODE = "mode"
    FORWARD_FILL = "ffill"
    BACKWARD_FILL = "bfill"
    CUSTOM_VALUE = "custom"
    DISTRIBUTION = "distribution"
    KNN = "knn"


class ScalingMethod(Enum):
    """Feature scaling methods"""
    STANDARD = "standard"
    MINMAX = "minmax"
    ROBUST = "robust"
    MAXABS = "maxabs"
    QUANTILE_UNIFORM = "quantile_uniform"
    QUANTILE_NORMAL = "quantile_normal"


class EncodingMethod(Enum):
    """Categorical encoding methods"""
    LABEL = "label"
    ONEHOT = "onehot"
    ORDINAL = "ordinal"
    TARGET = "target"


@dataclass
class RideConfig:
    """
    Global configuration for wrang operations

    This class centralizes all configuration settings to avoid hardcoded values
    throughout the codebase and make the library more maintainable.
    """
    
    # Performance Settings
    random_state: int = 42
    max_memory_usage_mb: int = 1024  # Maximum memory usage for operations
    chunk_size: int = 10000  # Default chunk size for streaming operations
    sample_size: int = 1000  # Default sample size for quick previews
    max_unique_values_display: int = 20  # Max unique values to show in summaries
    
    # File Handling
    supported_formats: List[str] = field(default_factory=lambda: [
        "csv", "xlsx", "xls", "parquet", "json"
    ])
    default_encoding: str = "utf-8"
    csv_delimiter: str = ","
    excel_sheet_name: Optional[str] = None  # None means first sheet
    
    # Data Processing
    missing_value_threshold: float = 0.9  # Drop columns with >90% missing values
    correlation_threshold: float = 0.90  # High correlation threshold
    outlier_method: str = "iqr"  # Methods: 'iqr', 'zscore', 'isolation'
    outlier_factor: float = 1.5  # IQR multiplier for outlier detection
    
    # Visualization Settings
    plot_width: int = 80
    plot_height: int = 20
    max_categories_plot: int = 20  # Max categories to plot in bar charts
    color_palette: str = "viridis"
    
    # CLI Interface Settings
    terminal_width: int = 120
    show_progress_bars: bool = True
    verbose: bool = False
    debug: bool = False
    
    # Feature Engineering
    max_features_for_onehot: int = 30  # Max unique values for one-hot encoding
    max_features_for_label: int = 300  # Max unique values for label encoding
    default_test_size: float = 0.2
    cross_validation_folds: int = 5
    
    # Export Settings
    default_export_format: str = "csv"
    include_index_in_export: bool = False
    export_compression: Optional[str] = None  # None, 'gzip', 'bz2', 'xz'
    
    # Advanced Settings
    enable_lazy_loading: bool = True
    enable_query_optimization: bool = True
    parallel_processing: bool = True
    max_workers: Optional[int] = None  # None means use all available cores
    
    def __post_init__(self):
        """Validate configuration after initialization"""
        self._validate_config()
        
        # Set terminal width dynamically if not explicitly set
        if self.terminal_width == 120:
            try:
                import shutil
                self.terminal_width = min(shutil.get_terminal_size().columns, 120)
            except Exception:
                pass  # Keep default
    
    def _validate_config(self) -> None:
        """Validate configuration values"""
        if self.random_state < 0:
            raise ValueError("random_state must be non-negative")
        
        if not 0 < self.missing_value_threshold <= 1:
            raise ValueError("missing_value_threshold must be between 0 and 1")
        
        if not 0 < self.correlation_threshold <= 1:
            raise ValueError("correlation_threshold must be between 0 and 1")
        
        if self.chunk_size <= 0:
            raise ValueError("chunk_size must be positive")
        
        if self.sample_size <= 0:
            raise ValueError("sample_size must be positive")
    
    @classmethod
    def from_file(cls, config_path: Path) -> 'RideConfig':
        """
        Load configuration from JSON file
        
        Args:
            config_path: Path to configuration file
            
        Returns:
            RideConfig instance with loaded settings
        """
        try:
            with open(config_path, 'r', encoding='utf-8') as f:
                config_dict = json.load(f)
            return cls(**config_dict)
        except FileNotFoundError:
            raise FileNotFoundError(f"Configuration file not found: {config_path}")
        except json.JSONDecodeError as e:
            raise ValueError(f"Invalid JSON in configuration file: {e}")
    
    def to_file(self, config_path: Path) -> None:
        """
        Save configuration to JSON file
        
        Args:
            config_path: Path where to save configuration
        """
        # Convert to dictionary, handling non-serializable fields
        config_dict = {}
        for key, value in self.__dict__.items():
            if isinstance(value, (list, dict, str, int, float, bool)) or value is None:
                config_dict[key] = value
            else:
                config_dict[key] = str(value)
        
        config_path.parent.mkdir(parents=True, exist_ok=True)
        with open(config_path, 'w', encoding='utf-8') as f:
            json.dump(config_dict, f, indent=2, ensure_ascii=False)
    
    def update(self, **kwargs) -> 'RideConfig':
        """
        Create a new config with updated values
        
        Args:
            **kwargs: Configuration parameters to update
            
        Returns:
            New RideConfig instance with updated values
        """
        current_dict = self.__dict__.copy()
        current_dict.update(kwargs)
        return RideConfig(**current_dict)
    
    def get_file_config(self, file_path: Path) -> Dict[str, Any]:
        """
        Get file-specific configuration based on file extension
        
        Args:
            file_path: Path to the file
            
        Returns:
            Dictionary with file-specific settings
        """
        suffix = file_path.suffix.lower().lstrip('.')
        
        base_config = {
            'encoding': self.default_encoding,
            'chunk_size': self.chunk_size,
        }
        
        if suffix == 'csv':
            base_config.update({
                'delimiter': self.csv_delimiter,
                'quoting': 1,  # csv.QUOTE_ALL
                'escapechar': None,
            })
        elif suffix in ['xlsx', 'xls']:
            base_config.update({
                'sheet_name': self.excel_sheet_name,
                'engine': 'openpyxl' if suffix == 'xlsx' else 'xlrd',
            })
        elif suffix == 'parquet':
            base_config.update({
                'engine': 'pyarrow',
                'use_pandas_metadata': True,
            })
        elif suffix == 'json':
            base_config.update({
                'orient': 'records',
                'lines': False,
            })
        
        return base_config


# Global configuration instance
_global_config: Optional[RideConfig] = None


def get_config() -> RideConfig:
    """
    Get the global configuration instance
    
    Returns:
        Global RideConfig instance
    """
    global _global_config
    if _global_config is None:
        _global_config = RideConfig()
    return _global_config


def set_config(config: RideConfig) -> None:
    """
    Set the global configuration instance
    
    Args:
        config: RideConfig instance to set as global
    """
    global _global_config
    _global_config = config


def update_config(**kwargs) -> RideConfig:
    """
    Update global configuration with new values
    
    Args:
        **kwargs: Configuration parameters to update
        
    Returns:
        Updated global configuration
    """
    global _global_config
    current_config = get_config()
    _global_config = current_config.update(**kwargs)
    return _global_config


def reset_config() -> RideConfig:
    """
    Reset configuration to defaults
    
    Returns:
        Reset global configuration
    """
    global _global_config
    _global_config = RideConfig()
    return _global_config


# Configuration file paths
def get_config_dir() -> Path:
    """Get the configuration directory"""
    config_dir = Path.home() / '.wrang'
    config_dir.mkdir(exist_ok=True)
    return config_dir


def get_default_config_path() -> Path:
    """Get the default configuration file path"""
    return get_config_dir() / 'config.json'


def load_user_config() -> RideConfig:
    """
    Load user configuration from default location
    
    Returns:
        User configuration if exists, otherwise default configuration
    """
    config_path = get_default_config_path()
    if config_path.exists():
        try:
            return RideConfig.from_file(config_path)
        except Exception as e:
            print(f"Warning: Could not load user config: {e}")
            print("Using default configuration.")
    
    return RideConfig()


def save_user_config(config: Optional[RideConfig] = None) -> None:
    """
    Save configuration to user's config file
    
    Args:
        config: Configuration to save. If None, saves current global config.
    """
    if config is None:
        config = get_config()
    
    config_path = get_default_config_path()
    config.to_file(config_path)


# Initialize global config on import
try:
    _global_config = load_user_config()
except Exception:
    _global_config = RideConfig()