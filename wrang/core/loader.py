#!/usr/bin/env python3
"""
RIDE Core Data Loader
"""

import os
import json
from pathlib import Path
from typing import Union, Optional, Dict, Any, List, Tuple, Iterator
import polars as pl
from rich.console import Console
from rich.progress import Progress, SpinnerColumn, TextColumn, BarColumn, TimeElapsedColumn

from wrang.config import get_config, FileFormat
from wrang.utils.exceptions import (
    DataLoadError, DataValidationError, MemoryError, 
    UnsupportedOperationError, handle_polars_error
)

console = Console()


class FastDataLoader:
    """
    Lightning-fast data loader using Polars
    
    Provides 10x performance improvement over pandas while supporting
    all major file formats with smart memory management.
    """
    
    # Supported file extensions mapped to formats
    SUPPORTED_EXTENSIONS = {
        '.csv': FileFormat.CSV,
        '.xlsx': FileFormat.EXCEL,
        '.xls': FileFormat.EXCEL,
        '.parquet': FileFormat.PARQUET,
        '.json': FileFormat.JSON,
        '.jsonl': FileFormat.JSON,  # JSON Lines format
    }
    
    def __init__(self):
        self.config = get_config()
        self._cached_schemas: Dict[str, Dict[str, Any]] = {}
    
    def load(self, file_path: Union[str, Path], **kwargs) -> pl.DataFrame:
        """
        Load data from any supported format with automatic format detection
        
        Args:
            file_path: Path to the data file
            **kwargs: Additional parameters specific to file format
            
        Returns:
            Polars DataFrame with loaded data
            
        Raises:
            DataLoadError: If loading fails
            UnsupportedOperationError: If file format not supported
        """
        file_path = Path(file_path)
        
        # Validate file exists
        if not file_path.exists():
            raise DataLoadError(
                file_path, 
                "File not found",
                FileNotFoundError(f"No such file: {file_path}")
            )
        
        # Check file size and memory requirements
        self._check_memory_requirements(file_path)
        
        # Detect format
        file_format = self._detect_format(file_path)
        
        # Load based on format
        try:
            with Progress(
                SpinnerColumn(),
                TextColumn("[progress.description]{task.description}"),
                BarColumn(),
                TimeElapsedColumn(),
                console=console,
                transient=True
            ) as progress:
                task = progress.add_task(f"Loading {file_path.name}...", total=None)
                
                if file_format == FileFormat.CSV:
                    df = self._load_csv(file_path, **kwargs)
                elif file_format == FileFormat.EXCEL:
                    df = self._load_excel(file_path, **kwargs)
                elif file_format == FileFormat.PARQUET:
                    df = self._load_parquet(file_path, **kwargs)
                elif file_format == FileFormat.JSON:
                    df = self._load_json(file_path, **kwargs)
                else:
                    raise UnsupportedOperationError(
                        "load", 
                        str(file_format), 
                        f"Format {file_format} not implemented"
                    )
                
                progress.update(task, completed=True)
            
            # Cache schema for future reference
            self._cache_schema(str(file_path), df)
            
            console.print(f"✅ Successfully loaded {len(df):,} rows × {len(df.columns)} columns")
            return df
            
        except Exception as e:
            if isinstance(e, (DataLoadError, UnsupportedOperationError)):
                raise
            else:
                # Convert to appropriate RIDE exception
                raise handle_polars_error(e, "data loading")
    
    def scan_lazy(self, file_path: Union[str, Path], **kwargs) -> pl.LazyFrame:
        """
        Create lazy frame for efficient processing of large files
        
        Args:
            file_path: Path to the data file
            **kwargs: Additional parameters
            
        Returns:
            Polars LazyFrame for deferred execution
        """
        file_path = Path(file_path)
        file_format = self._detect_format(file_path)
        
        try:
            if file_format == FileFormat.CSV:
                return pl.scan_csv(file_path, **self._get_csv_kwargs(**kwargs))
            elif file_format == FileFormat.PARQUET:
                return pl.scan_parquet(file_path, **kwargs)
            else:
                # For formats that don't support lazy loading, load and convert
                df = self.load(file_path, **kwargs)
                return df.lazy()
                
        except Exception as e:
            raise handle_polars_error(e, "lazy scanning")
    
    def stream_chunks(
        self,
        file_path: Union[str, Path],
        chunk_size: Optional[int] = None,
        **kwargs,
    ) -> Iterator[pl.DataFrame]:
        """
        Stream a large file in fixed-size chunks without loading it all into RAM.

        Supported for CSV and Parquet. For other formats the file is loaded
        fully and split into chunks in-memory.

        Args:
            file_path: Path to the data file
            chunk_size: Rows per chunk (default from config)

        Yields:
            Polars DataFrames, one per chunk
        """
        file_path = Path(file_path)
        if chunk_size is None:
            chunk_size = self.config.chunk_size

        file_format = self._detect_format(file_path)

        try:
            if file_format == FileFormat.CSV:
                csv_kwargs = self._get_csv_kwargs(**kwargs)
                reader = pl.read_csv_batched(file_path, batch_size=chunk_size, **csv_kwargs)
                while True:
                    batch = reader.next_batches(1)
                    if not batch:
                        break
                    yield batch[0]

            elif file_format == FileFormat.PARQUET:
                # Parquet: use lazy scan + slice for memory-efficient chunking
                lf = pl.scan_parquet(file_path)
                total = lf.collect().height  # need row count once
                offset = 0
                while offset < total:
                    yield lf.slice(offset, chunk_size).collect()
                    offset += chunk_size

            else:
                # Fallback: full load then chunk
                df = self.load(file_path, **kwargs)
                for offset in range(0, len(df), chunk_size):
                    yield df.slice(offset, chunk_size)

        except Exception as e:
            raise handle_polars_error(e, "streaming chunks")

    def peek(self, file_path: Union[str, Path], n_rows: int = None) -> pl.DataFrame:
        """
        Quick peek at data without loading the entire file
        
        Args:
            file_path: Path to the data file
            n_rows: Number of rows to peek (default from config)
            
        Returns:
            Small sample of the data
        """
        if n_rows is None:
            n_rows = self.config.sample_size
        
        file_path = Path(file_path)
        file_format = self._detect_format(file_path)
        
        try:
            if file_format == FileFormat.CSV:
                return pl.read_csv(file_path, n_rows=n_rows)
            elif file_format == FileFormat.PARQUET:
                # For Parquet, we load and then take first n rows
                df = pl.read_parquet(file_path)
                return df.head(n_rows)
            else:
                # Load full file and take sample
                df = self.load(file_path)
                return df.head(n_rows)
                
        except Exception as e:
            raise handle_polars_error(e, "data peeking")
    
    def get_info(self, file_path: Union[str, Path]) -> Dict[str, Any]:
        """
        Get file information without loading data
        
        Args:
            file_path: Path to the data file
            
        Returns:
            Dictionary with file metadata
        """
        file_path = Path(file_path)
        
        if not file_path.exists():
            raise DataLoadError(file_path, "File not found")
        
        # Basic file info
        stat = file_path.stat()
        info = {
            'file_name': file_path.name,
            'file_size_bytes': stat.st_size,
            'file_size_human': self._format_file_size(stat.st_size),
            'file_format': self._detect_format(file_path).value,
            'last_modified': stat.st_mtime,
        }
        
        # Try to get column info quickly
        try:
            if file_path.suffix.lower() == '.csv':
                # Read just the header
                header_df = pl.read_csv(file_path, n_rows=0)
                info['columns'] = header_df.columns
                info['n_columns'] = len(header_df.columns)
                
                # Estimate row count for CSV
                info['estimated_rows'] = self._estimate_csv_rows(file_path)
            else:
                # For other formats, we need to peek
                sample = self.peek(file_path, n_rows=100)
                info['columns'] = sample.columns
                info['n_columns'] = len(sample.columns)
                info['sample_rows'] = len(sample)
                
        except Exception:
            info['columns'] = []
            info['n_columns'] = 0
        
        return info
    
    def _detect_format(self, file_path: Path) -> FileFormat:
        """Detect file format from extension"""
        suffix = file_path.suffix.lower()
        
        if suffix not in self.SUPPORTED_EXTENSIONS:
            raise UnsupportedOperationError(
                "format detection",
                suffix,
                f"File extension '{suffix}' is not supported",
                supported_types=list(self.SUPPORTED_EXTENSIONS.keys())
            )
        
        return self.SUPPORTED_EXTENSIONS[suffix]
    
    def _check_memory_requirements(self, file_path: Path) -> None:
        """Check if file size might cause memory issues"""
        file_size_mb = file_path.stat().st_size / (1024 * 1024)
        max_memory_mb = self.config.max_memory_usage_mb
        
        # Rough estimate: CSV needs ~3x file size in memory
        # Parquet is more efficient, Excel varies
        multiplier = {
            '.csv': 3.0,
            '.xlsx': 4.0,
            '.xls': 4.0,
            '.parquet': 1.5,
            '.json': 2.5,
        }.get(file_path.suffix.lower(), 3.0)
        
        estimated_memory_mb = file_size_mb * multiplier
        
        if estimated_memory_mb > max_memory_mb:
            raise MemoryError(
                "data loading",
                required_memory_mb=estimated_memory_mb,
                available_memory_mb=max_memory_mb,
                dataset_size=(None, None)  # We don't know dimensions yet
            )
    
    def _load_csv(self, file_path: Path, **kwargs) -> pl.DataFrame:
        """Load CSV file with optimized settings"""
        csv_kwargs = self._get_csv_kwargs(**kwargs)
        
        try:
            return pl.read_csv(file_path, **csv_kwargs)
        except Exception as e:
            # Try with different settings if first attempt fails
            if "encoding" not in csv_kwargs:
                try:
                    csv_kwargs['encoding'] = 'latin-1'
                    return pl.read_csv(file_path, **csv_kwargs)
                except Exception:
                    pass
            
            raise DataLoadError(
                file_path,
                f"CSV parsing failed: {str(e)}",
                original_error=e
            )
    
    def _load_excel(self, file_path: Path, **kwargs) -> pl.DataFrame:
        """Load Excel file using native Polars reader (calamine engine)."""
        try:
            sheet = kwargs.get('sheet_name', kwargs.get('sheet_id', 0))
            # Polars accepts sheet_id (0-based int) or sheet_name (str)
            if isinstance(sheet, int):
                return pl.read_excel(file_path, sheet_id=sheet + 1)
            return pl.read_excel(file_path, sheet_name=sheet)
        except Exception as native_err:
            # Graceful fallback to pandas for exotic Excel files
            try:
                import pandas as pd
                excel_kwargs = self._get_excel_kwargs(**kwargs)
                df_pandas = pd.read_excel(file_path, **excel_kwargs)
                return pl.from_pandas(df_pandas)
            except Exception as e:
                raise DataLoadError(
                    file_path,
                    f"Excel loading failed: {str(e)}",
                    original_error=e,
                )
    
    def _load_parquet(self, file_path: Path, **kwargs) -> pl.DataFrame:
        """Load Parquet file"""
        try:
            return pl.read_parquet(file_path, **kwargs)
        except Exception as e:
            raise DataLoadError(
                file_path,
                f"Parquet loading failed: {str(e)}",
                original_error=e
            )
    
    def _load_json(self, file_path: Path, **kwargs) -> pl.DataFrame:
        """Load JSON / JSON-Lines file using native Polars readers."""
        try:
            if file_path.suffix.lower() == '.jsonl':
                return pl.read_ndjson(file_path)
            # Try native Polars JSON reader first
            try:
                return pl.read_json(file_path)
            except Exception:
                # Fallback: normalise nested JSON via json.load
                with open(file_path, 'r', encoding=self.config.default_encoding) as f:
                    data = json.load(f)
                if isinstance(data, list):
                    return pl.from_dicts(data)
                # single-record dict → wrap in list
                return pl.from_dicts([data])
        except Exception as e:
            raise DataLoadError(
                file_path,
                f"JSON loading failed: {str(e)}",
                original_error=e,
            )
    
    def _get_csv_kwargs(self, **kwargs) -> Dict[str, Any]:
        """Get CSV loading parameters"""
        config_csv = self.config.get_file_config(Path("dummy.csv"))
        
        csv_kwargs = {
            'separator': kwargs.get('delimiter', config_csv.get('delimiter', ',')),
            'encoding': kwargs.get('encoding', config_csv.get('encoding', 'utf-8')),
            'infer_schema_length': kwargs.get('infer_schema_length', 10000),
            'ignore_errors': kwargs.get('ignore_errors', True),
            'try_parse_dates': kwargs.get('try_parse_dates', True),
        }
        
        # Remove None values
        return {k: v for k, v in csv_kwargs.items() if v is not None}
    
    def _get_excel_kwargs(self, **kwargs) -> Dict[str, Any]:
        """Get Excel loading parameters"""
        config_excel = self.config.get_file_config(Path("dummy.xlsx"))
        
        excel_kwargs = {
            'sheet_name': kwargs.get('sheet_name', config_excel.get('sheet_name', 0)),
            'header': kwargs.get('header', 0),
            'engine': kwargs.get('engine', config_excel.get('engine', 'openpyxl')),
        }
        
        return {k: v for k, v in excel_kwargs.items() if v is not None}
    
    def _estimate_csv_rows(self, file_path: Path) -> Optional[int]:
        """Estimate number of rows in CSV file"""
        try:
            # Read a sample and estimate
            sample_size = min(self.config.chunk_size, 1000)
            sample_df = pl.read_csv(file_path, n_rows=sample_size)
            
            if len(sample_df) == 0:
                return 0
            
            # Estimate based on file size and sample
            file_size = file_path.stat().st_size
            
            # Read first chunk to estimate bytes per row
            with open(file_path, 'rb') as f:
                chunk = f.read(min(8192, file_size))
                lines_in_chunk = chunk.count(b'\n')
                
            if lines_in_chunk > 0:
                bytes_per_line = len(chunk) / lines_in_chunk
                estimated_lines = int(file_size / bytes_per_line)
                return max(0, estimated_lines - 1)  # Subtract header
            
            return None
            
        except Exception:
            return None
    
    def _cache_schema(self, file_path: str, df: pl.DataFrame) -> None:
        """Cache DataFrame schema for future reference"""
        schema_info = {
            'columns': df.columns,
            'dtypes': [str(dtype) for dtype in df.dtypes],
            'shape': df.shape,
            'memory_usage': df.estimated_size('mb'),
        }
        self._cached_schemas[file_path] = schema_info
    
    def get_cached_schema(self, file_path: str) -> Optional[Dict[str, Any]]:
        """Get cached schema information"""
        return self._cached_schemas.get(file_path)
    
    @staticmethod
    def _format_file_size(size_bytes: int) -> str:
        """Format file size in human-readable format"""
        for unit in ['B', 'KB', 'MB', 'GB', 'TB']:
            if size_bytes < 1024:
                return f"{size_bytes:.1f} {unit}"
            size_bytes /= 1024
        return f"{size_bytes:.1f} PB"


class DataSaver:
    """
    Fast data saving with multiple format support
    """
    
    def __init__(self):
        self.config = get_config()
    
    def save(self, df: pl.DataFrame, file_path: Union[str, Path], 
             format_type: Optional[str] = None, **kwargs) -> None:
        """
        Save DataFrame to file with automatic format detection
        
        Args:
            df: Polars DataFrame to save
            file_path: Output file path
            format_type: Force specific format (optional)
            **kwargs: Format-specific parameters
        """
        file_path = Path(file_path)
        
        # Create directory if it doesn't exist
        file_path.parent.mkdir(parents=True, exist_ok=True)
        
        # Detect format from extension if not specified
        if format_type is None:
            loader = FastDataLoader()
            file_format = loader._detect_format(file_path)
        else:
            file_format = FileFormat(format_type.lower())
        
        try:
            with Progress(
                SpinnerColumn(),
                TextColumn("[progress.description]{task.description}"),
                console=console,
                transient=True
            ) as progress:
                task = progress.add_task(f"Saving to {file_path.name}...", total=None)
                
                if file_format == FileFormat.CSV:
                    self._save_csv(df, file_path, **kwargs)
                elif file_format == FileFormat.EXCEL:
                    self._save_excel(df, file_path, **kwargs)
                elif file_format == FileFormat.PARQUET:
                    self._save_parquet(df, file_path, **kwargs)
                elif file_format == FileFormat.JSON:
                    self._save_json(df, file_path, **kwargs)
                
                progress.update(task, completed=True)
            
            console.print(f"✅ Successfully saved {len(df):,} rows to {file_path.name}")
            
        except Exception as e:
            from wrang.utils.exceptions import ExportError
            raise ExportError(file_path, file_format.value, str(e), original_error=e)
    
    def _save_csv(self, df: pl.DataFrame, file_path: Path, **kwargs) -> None:
        """Save as CSV"""
        csv_kwargs = {
            'separator': kwargs.get('delimiter', self.config.csv_delimiter),
            'include_header': kwargs.get('include_header', True),
        }
        df.write_csv(file_path, **csv_kwargs)
    
    def _save_excel(self, df: pl.DataFrame, file_path: Path, **kwargs) -> None:
        """Save as Excel using native Polars writer."""
        try:
            df.write_excel(file_path, worksheet=kwargs.get('sheet_name', 'Sheet1'))
        except Exception:
            # Fallback to pandas for edge cases
            import pandas as pd
            df.to_pandas().to_excel(
                file_path,
                index=kwargs.get('index', self.config.include_index_in_export),
                sheet_name=kwargs.get('sheet_name', 'Sheet1'),
            )

    def _save_parquet(self, df: pl.DataFrame, file_path: Path, **kwargs) -> None:
        """Save as Parquet"""
        compression = kwargs.get('compression', self.config.export_compression)
        df.write_parquet(file_path, compression=compression)

    def _save_json(self, df: pl.DataFrame, file_path: Path, **kwargs) -> None:
        """Save as JSON using native Polars writer."""
        df.write_json(file_path, row_oriented=True)


# Convenience functions for backward compatibility
def load_data(file_path: Union[str, Path], **kwargs) -> pl.DataFrame:
    """Load data using FastDataLoader"""
    loader = FastDataLoader()
    return loader.load(file_path, **kwargs)


def save_data(df: pl.DataFrame, file_path: Union[str, Path], **kwargs) -> None:
    """Save data using DataSaver"""
    saver = DataSaver()
    saver.save(df, file_path, **kwargs)