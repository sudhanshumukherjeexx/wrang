#!/usr/bin/env python3
"""
wrang Utils - Exception Handling
Custom exceptions and error handling utilities
"""

from typing import Optional, Any, Tuple, Union
from pathlib import Path


class RideError(Exception):
    """Base exception for all wrang errors"""
    
    def __init__(self, message: str, details: Optional[str] = None, 
                 original_error: Optional[Exception] = None):
        self.message = message
        self.details = details
        self.original_error = original_error
        super().__init__(self.message)


class DataLoadError(RideError):
    """Exception raised when data loading fails"""
    
    def __init__(self, file_path: Union[str, Path], message: str, 
                 original_error: Optional[Exception] = None):
        self.file_path = Path(file_path)
        super().__init__(f"Failed to load {self.file_path.name}: {message}", 
                        original_error=original_error)


class DataValidationError(RideError):
    """Exception raised when data validation fails"""
    pass


class PreprocessingError(RideError):
    """Exception raised during data preprocessing"""

    def __init__(self, operation: str, message: str,
                 original_error: Optional[Exception] = None,
                 affected_columns: Optional[list] = None,
                 suggestions: Optional[list] = None):
        self.operation = operation
        self.affected_columns = affected_columns or []
        self.suggestions = suggestions or []
        super().__init__(f"Preprocessing error in {operation}: {message}",
                        original_error=original_error)


class ExportError(RideError):
    """Exception raised when data export fails"""
    
    def __init__(self, file_path: Union[str, Path], format_type: str, 
                 message: str, original_error: Optional[Exception] = None):
        self.file_path = Path(file_path)
        self.format_type = format_type
        super().__init__(f"Failed to export to {self.file_path.name} ({format_type}): {message}",
                        original_error=original_error)


class MemoryError(RideError):
    """Exception raised when memory requirements exceed limits"""
    
    def __init__(self, operation: str, required_memory_mb: float, 
                 available_memory_mb: float, dataset_size: Tuple[Optional[int], Optional[int]]):
        self.operation = operation
        self.required_memory_mb = required_memory_mb
        self.available_memory_mb = available_memory_mb
        self.dataset_size = dataset_size
        
        message = (f"Memory limit exceeded for {operation}: "
                  f"requires {required_memory_mb:.1f}MB, "
                  f"limit is {available_memory_mb:.1f}MB")
        super().__init__(message)


class UnsupportedOperationError(RideError):
    """Exception raised when an operation is not supported"""
    
    def __init__(self, operation: str, item: str, message: str,
                 supported_types: Optional[list] = None):
        self.operation = operation
        self.item = item
        self.supported_types = supported_types or []
        
        error_message = f"Unsupported {operation}: {item}. {message}"
        if self.supported_types:
            error_message += f" Supported: {', '.join(self.supported_types)}"
        
        super().__init__(error_message)


def handle_polars_error(error: Exception, context: str) -> RideError:
    """
    Convert Polars errors to appropriate wrang errors
    
    Args:
        error: Original Polars exception
        context: Context where the error occurred
        
    Returns:
        Appropriate wrang exception
    """
    error_str = str(error).lower()
    
    if 'memory' in error_str or 'out of memory' in error_str:
        return MemoryError(context, 0, 0, (None, None))
    elif 'file not found' in error_str or 'no such file' in error_str:
        return DataLoadError("unknown", f"File not found during {context}", error)
    elif 'parsing' in error_str or 'parse' in error_str:
        return DataLoadError("unknown", f"File parsing error during {context}", error)
    elif 'dtype' in error_str or 'data type' in error_str:
        return DataValidationError(f"Data type error during {context}: {error}")
    else:
        return RideError(f"Error during {context}: {error}", original_error=error)


def create_user_friendly_message(error: RideError) -> str:
    """
    Create user-friendly error messages from wrang exceptions
    
    Args:
        error: wrang exception
        
    Returns:
        User-friendly error message
    """
    if isinstance(error, DataLoadError):
        return f"❌ Could not load file '{error.file_path.name}'. Please check the file path and format."
    
    elif isinstance(error, MemoryError):
        return (f"⚠️ File too large for current memory settings ({error.required_memory_mb:.1f}MB required). "
                f"Try increasing memory limit in settings or use a smaller file.")
    
    elif isinstance(error, UnsupportedOperationError):
        if error.supported_types:
            return (f"❌ {error.item} is not supported for {error.operation}. "
                   f"Supported types: {', '.join(error.supported_types)}")
        else:
            return f"❌ {error.operation} is not supported for {error.item}"
    
    elif isinstance(error, ExportError):
        return f"❌ Could not save file '{error.file_path.name}' in {error.format_type} format."
    
    elif isinstance(error, PreprocessingError):
        return f"❌ Error during {error.operation}. Please check your data and try again."
    
    elif isinstance(error, DataValidationError):
        return f"❌ Data validation failed: {error.message}"
    
    else:
        return f"❌ {error.message}"


# Error recovery suggestions
ERROR_SUGGESTIONS = {
    DataLoadError: [
        "Check that the file path is correct",
        "Verify the file format is supported (CSV, Excel, Parquet, JSON)",
        "Try opening the file in another application first",
        "Check file permissions"
    ],
    
    MemoryError: [
        "Increase memory limit in Settings",
        "Use a smaller sample of your data",
        "Try loading the file in chunks",
        "Consider using a more powerful machine"
    ],
    
    UnsupportedOperationError: [
        "Check the supported file formats",
        "Consider converting your data to a supported format",
        "Update wrang to the latest version"
    ],
    
    ExportError: [
        "Check write permissions for the output directory",
        "Verify you have enough disk space",
        "Try a different output format",
        "Check that the output path is valid"
    ],
    
    PreprocessingError: [
        "Check your data for unexpected values",
        "Try a different preprocessing strategy",
        "Remove problematic columns temporarily",
        "Check for missing values or infinite numbers"
    ]
}


def get_error_suggestions(error: RideError) -> list:
    """Get suggestions for resolving an error"""
    error_type = type(error)
    return ERROR_SUGGESTIONS.get(error_type, [
        "Check your data and try again",
        "Report this issue if the problem persists"
    ])


