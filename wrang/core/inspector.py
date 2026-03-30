#!/usr/bin/env python3
"""
RIDE Core Data Inspector
Smart data inspection and profiling with beautiful terminal output
"""

import polars as pl
import numpy as np
from typing import Dict, Any, List, Optional, Tuple, Union
from datetime import datetime
from collections import Counter
from rich.console import Console
from rich.table import Table
from rich.panel import Panel
from rich.columns import Columns
from rich.progress import track
from rich.text import Text
from joblib import Parallel, delayed

from wrang.config import get_config
from wrang.utils.exceptions import DataValidationError, handle_polars_error
from wrang.utils.constants import (
    numeric_columns, categorical_columns, datetime_columns, format_memory,
)

console = Console()


class DataInspector:
    """
    Comprehensive data inspection and profiling
    
    Provides detailed analysis of dataset structure, quality, and characteristics
    with beautiful terminal output using Rich.
    """
    
    def __init__(self, df: pl.DataFrame):
        """
        Initialize inspector with DataFrame
        
        Args:
            df: Polars DataFrame to inspect
        """
        if df is None or len(df) == 0:
            raise DataValidationError("Cannot inspect empty or None DataFrame")
        
        self.df = df
        self.config = get_config()
        self._basic_info: Optional[Dict[str, Any]] = None
        self._column_profiles: Optional[Dict[str, Dict[str, Any]]] = None
    
    def get_basic_info(self) -> Dict[str, Any]:
        """
        Get basic dataset information
        
        Returns:
            Dictionary with basic dataset metrics
        """
        if self._basic_info is None:
            self._basic_info = self._compute_basic_info()
        return self._basic_info
    
    def display_overview(self) -> None:
        """Display comprehensive dataset overview"""
        info = self.get_basic_info()
        
        # Create overview panel
        overview_content = self._create_overview_content(info)
        overview_panel = Panel(
            overview_content,
            title="📊 Dataset Overview",
            border_style="cyan",
            padding=(1, 2)
        )
        
        console.print(overview_panel)
        
        # Display column summary
        self.display_column_summary()
        
        # Display data quality metrics
        self.display_data_quality()
    
    def display_column_summary(self) -> None:
        """Display detailed column information"""
        columns_table = Table(title="📋 Column Details", show_header=True, header_style="bold cyan")
        
        # Add columns to table
        columns_table.add_column("Column", style="white", min_width=20)
        columns_table.add_column("Type", style="green", min_width=12)
        columns_table.add_column("Missing", style="red", min_width=8)
        columns_table.add_column("Missing %", style="red", min_width=10)
        columns_table.add_column("Unique", style="blue", min_width=8)
        columns_table.add_column("Unique %", style="blue", min_width=10)
        columns_table.add_column("Sample Values", style="yellow", min_width=25)
        
        # Get column profiles
        profiles = self.get_column_profiles()
        
        for col_name, profile in profiles.items():
            # Format missing percentage
            missing_pct = f"{profile['missing_percentage']:.1f}%" if profile['missing_percentage'] > 0 else "0%"
            
            # Format unique percentage
            unique_pct = f"{profile['unique_percentage']:.1f}%"
            
            # Format sample values
            sample_values = profile['sample_values'][:3]  # First 3 samples
            sample_str = ", ".join(str(v) for v in sample_values)
            if len(sample_str) > 30:
                sample_str = sample_str[:27] + "..."
            
            # Add color coding for missing values
            missing_style = "red" if profile['missing_percentage'] > 10 else "yellow" if profile['missing_percentage'] > 0 else "green"
            
            columns_table.add_row(
                col_name[:20],  # Truncate long column names
                str(profile['dtype']),
                str(profile['missing_count']),
                Text(missing_pct, style=missing_style),
                str(profile['unique_count']),
                unique_pct,
                sample_str
            )
        
        console.print(columns_table)
    
    def display_data_quality(self) -> None:
        """Display data quality metrics"""
        quality_metrics = self._analyze_data_quality()
        
        # Create quality summary panels
        panels = []
        
        # Missing values panel
        missing_panel = self._create_missing_values_panel(quality_metrics['missing_values'])
        panels.append(missing_panel)
        
        # Data types panel
        types_panel = self._create_data_types_panel(quality_metrics['data_types'])
        panels.append(types_panel)
        
        # Duplicates panel
        duplicates_panel = self._create_duplicates_panel(quality_metrics['duplicates'])
        panels.append(duplicates_panel)
        
        # Display panels in columns
        console.print(Columns(panels, equal=True, expand=True))
    
    def display_statistical_summary(self) -> None:
        """Display statistical summary for numeric columns"""
        numeric_cols = self._get_numeric_columns()
        
        if not numeric_cols:
            console.print("[yellow]📊 No numeric columns found for statistical summary[/yellow]")
            return
        
        # Create statistics table
        stats_table = Table(title="📈 Statistical Summary", show_header=True, header_style="bold cyan")
        stats_table.add_column("Column", style="white", min_width=15)
        stats_table.add_column("Count", style="blue", min_width=8)
        stats_table.add_column("Mean", style="green", min_width=10)
        stats_table.add_column("Std", style="green", min_width=10)
        stats_table.add_column("Min", style="magenta", min_width=10)
        stats_table.add_column("25%", style="cyan", min_width=10)
        stats_table.add_column("50%", style="cyan", min_width=10)
        stats_table.add_column("75%", style="cyan", min_width=10)
        stats_table.add_column("Max", style="magenta", min_width=10)
        
        for col in track(numeric_cols, description="Computing statistics..."):
            try:
                stats = self._compute_column_statistics(col)
                stats_table.add_row(
                    col[:15],
                    f"{stats['count']:,}",
                    f"{stats['mean']:.3f}" if stats['mean'] is not None else "N/A",
                    f"{stats['std']:.3f}" if stats['std'] is not None else "N/A",
                    f"{stats['min']:.3f}" if stats['min'] is not None else "N/A",
                    f"{stats['q25']:.3f}" if stats['q25'] is not None else "N/A",
                    f"{stats['median']:.3f}" if stats['median'] is not None else "N/A",
                    f"{stats['q75']:.3f}" if stats['q75'] is not None else "N/A",
                    f"{stats['max']:.3f}" if stats['max'] is not None else "N/A",
                )
            except Exception as e:
                # Skip columns that can't be processed
                console.print(f"[yellow]Warning: Could not compute statistics for {col}: {e}[/yellow]")
                continue
        
        console.print(stats_table)
    
    def get_column_profiles(self) -> Dict[str, Dict[str, Any]]:
        """
        Get detailed profiles for all columns
        
        Returns:
            Dictionary mapping column names to their profiles
        """
        if self._column_profiles is None:
            self._column_profiles = self._compute_column_profiles()
        return self._column_profiles
    
    def get_memory_usage(self) -> Dict[str, Any]:
        """
        Get memory usage information
        
        Returns:
            Dictionary with memory usage details
        """
        try:
            total_memory_mb = self.df.estimated_size('mb')
            
            # Calculate per-column memory usage
            column_memory = {}
            for col in self.df.columns:
                try:
                    col_memory = self.df.select(col).estimated_size('mb')
                    column_memory[col] = col_memory
                except Exception:
                    column_memory[col] = 0
            
            return {
                'total_memory_mb': total_memory_mb,
                'total_memory_human': self._format_memory(total_memory_mb),
                'column_memory': column_memory,
                'largest_columns': sorted(column_memory.items(), key=lambda x: x[1], reverse=True)[:5]
            }
        except Exception as e:
            return {
                'total_memory_mb': 0,
                'total_memory_human': 'Unknown',
                'column_memory': {},
                'largest_columns': [],
                'error': str(e)
            }
    
    def detect_potential_issues(self) -> List[Dict[str, Any]]:
        """
        Detect potential data quality issues
        
        Returns:
            List of detected issues with severity and recommendations
        """
        issues = []
        
        # Check for high missing value percentages
        profiles = self.get_column_profiles()
        for col_name, profile in profiles.items():
            missing_pct = profile['missing_percentage']
            
            if missing_pct > 50:
                issues.append({
                    'type': 'high_missing_values',
                    'severity': 'high',
                    'column': col_name,
                    'details': f'{missing_pct:.1f}% missing values',
                    'recommendation': 'Consider dropping this column or using advanced imputation'
                })
            elif missing_pct > 20:
                issues.append({
                    'type': 'moderate_missing_values',
                    'severity': 'medium',
                    'column': col_name,
                    'details': f'{missing_pct:.1f}% missing values',
                    'recommendation': 'Consider imputation strategies'
                })
        
        # Check for columns with single unique value
        for col_name, profile in profiles.items():
            if profile['unique_count'] == 1:
                issues.append({
                    'type': 'constant_column',
                    'severity': 'medium',
                    'column': col_name,
                    'details': 'Column has only one unique value',
                    'recommendation': 'Consider dropping this column as it provides no information'
                })
        
        # Check for potential ID columns (high uniqueness)
        for col_name, profile in profiles.items():
            if profile['unique_percentage'] > 95 and profile['unique_count'] > 100:
                issues.append({
                    'type': 'potential_id_column',
                    'severity': 'low',
                    'column': col_name,
                    'details': f'{profile["unique_percentage"]:.1f}% unique values',
                    'recommendation': 'This might be an ID column - consider removing for modeling'
                })
        
        # Check for duplicate rows
        duplicate_count = len(self.df) - len(self.df.unique())
        if duplicate_count > 0:
            duplicate_pct = (duplicate_count / len(self.df)) * 100
            severity = 'high' if duplicate_pct > 10 else 'medium' if duplicate_pct > 1 else 'low'
            issues.append({
                'type': 'duplicate_rows',
                'severity': severity,
                'column': 'entire_dataset',
                'details': f'{duplicate_count:,} duplicate rows ({duplicate_pct:.1f}%)',
                'recommendation': 'Consider removing duplicate rows'
            })
        
        return issues
    
    def _compute_basic_info(self) -> Dict[str, Any]:
        """Compute basic dataset information"""
        try:
            return {
                'n_rows': len(self.df),
                'n_columns': len(self.df.columns),
                'memory_usage': self.get_memory_usage(),
                'column_types': dict(zip(self.df.columns, [str(dtype) for dtype in self.df.dtypes])),
                'missing_values_total': self.df.null_count().sum_horizontal().item(),
                'duplicate_rows': len(self.df) - len(self.df.unique()),
                'numeric_columns': len(self._get_numeric_columns()),
                'categorical_columns': len(self._get_categorical_columns()),
                'datetime_columns': len(self._get_datetime_columns()),
            }
        except Exception as e:
            raise handle_polars_error(e, "basic info computation")
    
    def _compute_column_profiles(self) -> Dict[str, Dict[str, Any]]:
        """Compute detailed profiles for all columns using parallel workers."""

        def _safe_profile(col: str) -> Tuple[str, Dict[str, Any]]:
            try:
                return col, self._profile_single_column(col)
            except Exception as exc:
                return col, {
                    'dtype': str(self.df[col].dtype),
                    'missing_count': 0,
                    'missing_percentage': 0,
                    'unique_count': 0,
                    'unique_percentage': 0,
                    'sample_values': [],
                    'error': str(exc),
                }

        n_jobs = min(self.config.max_workers or -1, len(self.df.columns))
        results = Parallel(n_jobs=n_jobs, prefer="threads")(
            delayed(_safe_profile)(col) for col in self.df.columns
        )
        return dict(results)
    
    def _profile_single_column(self, col: str) -> Dict[str, Any]:
        """Profile a single column"""
        series = self.df[col]
        n_rows = len(self.df)
        
        # Basic statistics
        missing_count = series.null_count()
        missing_percentage = (missing_count / n_rows) * 100 if n_rows > 0 else 0
        
        # Unique values (handle potential memory issues)
        try:
            unique_count = series.n_unique()
            unique_percentage = (unique_count / n_rows) * 100 if n_rows > 0 else 0
        except Exception:
            unique_count = -1  # Indicate computation failed
            unique_percentage = -1
        
        # Sample values
        try:
            sample_values = series.drop_nulls().head(5).to_list()
        except Exception:
            sample_values = []
        
        profile = {
            'dtype': str(series.dtype),
            'missing_count': missing_count,
            'missing_percentage': missing_percentage,
            'unique_count': unique_count,
            'unique_percentage': unique_percentage,
            'sample_values': sample_values,
        }
        
        # Add type-specific information
        if self._is_numeric_column(col):
            try:
                stats = self._compute_column_statistics(col)
                profile.update(stats)
            except Exception:
                pass
        
        elif self._is_categorical_column(col):
            try:
                # Value counts for categorical
                value_counts = series.value_counts().head(10)
                profile['top_values'] = value_counts.to_dict() if hasattr(value_counts, 'to_dict') else {}
            except Exception:
                profile['top_values'] = {}
        
        return profile
    
    def _compute_column_statistics(self, col: str) -> Dict[str, Any]:
        """Compute statistical measures for numeric column"""
        series = self.df[col].drop_nulls()
        
        if len(series) == 0:
            return {
                'count': 0, 'mean': None, 'std': None, 'min': None,
                'q25': None, 'median': None, 'q75': None, 'max': None
            }
        
        try:
            return {
                'count': len(series),
                'mean': float(series.mean()) if series.mean() is not None else None,
                'std': float(series.std()) if series.std() is not None else None,
                'min': float(series.min()) if series.min() is not None else None,
                'q25': float(series.quantile(0.25)) if series.quantile(0.25) is not None else None,
                'median': float(series.median()) if series.median() is not None else None,
                'q75': float(series.quantile(0.75)) if series.quantile(0.75) is not None else None,
                'max': float(series.max()) if series.max() is not None else None,
            }
        except Exception:
            return {
                'count': len(series), 'mean': None, 'std': None, 'min': None,
                'q25': None, 'median': None, 'q75': None, 'max': None
            }
    
    def _analyze_data_quality(self) -> Dict[str, Any]:
        """Analyze overall data quality"""
        profiles = self.get_column_profiles()
        
        # Missing values analysis
        missing_analysis = {
            'columns_with_missing': sum(1 for p in profiles.values() if p['missing_count'] > 0),
            'total_missing_cells': sum(p['missing_count'] for p in profiles.values()),
            'missing_percentage': (sum(p['missing_count'] for p in profiles.values()) / 
                                 (len(self.df) * len(self.df.columns))) * 100,
            'worst_columns': sorted(
                [(col, p['missing_percentage']) for col, p in profiles.items() if p['missing_count'] > 0],
                key=lambda x: x[1], reverse=True
            )[:5]
        }
        
        # Data types analysis
        dtype_counts = Counter(p['dtype'] for p in profiles.values())
        types_analysis = {
            'type_distribution': dict(dtype_counts),
            'numeric_columns': len(self._get_numeric_columns()),
            'categorical_columns': len(self._get_categorical_columns()),
            'datetime_columns': len(self._get_datetime_columns()),
        }
        
        # Duplicates analysis
        duplicate_count = len(self.df) - len(self.df.unique())
        duplicates_analysis = {
            'duplicate_rows': duplicate_count,
            'duplicate_percentage': (duplicate_count / len(self.df)) * 100 if len(self.df) > 0 else 0,
            'unique_rows': len(self.df.unique()),
        }
        
        return {
            'missing_values': missing_analysis,
            'data_types': types_analysis,
            'duplicates': duplicates_analysis,
        }
    
    def _create_overview_content(self, info: Dict[str, Any]) -> str:
        """Create overview panel content"""
        memory_info = info['memory_usage']
        
        content = f"""[bold]Dataset Dimensions:[/bold]
• Rows: {info['n_rows']:,}
• Columns: {info['n_columns']:,}
• Memory Usage: {memory_info['total_memory_human']}

[bold]Column Types:[/bold]
• Numeric: {info['numeric_columns']}
• Categorical: {info['categorical_columns']}
• DateTime: {info['datetime_columns']}

[bold]Data Quality:[/bold]
• Missing Values: {info['missing_values_total']:,} cells
• Duplicate Rows: {info['duplicate_rows']:,}"""
        
        return content
    
    def _create_missing_values_panel(self, missing_info: Dict[str, Any]) -> Panel:
        """Create missing values summary panel"""
        content = f"""[bold]Missing Values Summary[/bold]

• Columns with missing: {missing_info['columns_with_missing']}
• Total missing cells: {missing_info['total_missing_cells']:,}
• Overall missing %: {missing_info['missing_percentage']:.1f}%

[bold]Worst Columns:[/bold]"""
        
        for col, pct in missing_info['worst_columns'][:3]:
            content += f"\n• {col[:15]}: {pct:.1f}%"
        
        return Panel(content, title="🚫 Missing Values", border_style="red")
    
    def _create_data_types_panel(self, types_info: Dict[str, Any]) -> Panel:
        """Create data types summary panel"""
        content = f"""[bold]Data Types Distribution[/bold]

• Numeric: {types_info['numeric_columns']}
• Categorical: {types_info['categorical_columns']}
• DateTime: {types_info['datetime_columns']}

[bold]Type Details:[/bold]"""
        
        for dtype, count in types_info['type_distribution'].items():
            short_type = dtype.split('(')[0]  # Remove Polars type parameters
            content += f"\n• {short_type}: {count}"
        
        return Panel(content, title="🔧 Data Types", border_style="green")
    
    def _create_duplicates_panel(self, dup_info: Dict[str, Any]) -> Panel:
        """Create duplicates summary panel"""
        content = f"""[bold]Duplicate Analysis[/bold]

• Duplicate rows: {dup_info['duplicate_rows']:,}
• Duplicate %: {dup_info['duplicate_percentage']:.1f}%
• Unique rows: {dup_info['unique_rows']:,}

[bold]Status:[/bold]"""
        
        if dup_info['duplicate_rows'] == 0:
            content += "\n✅ No duplicates found"
        elif dup_info['duplicate_percentage'] < 1:
            content += "\n⚠️ Low duplication"
        elif dup_info['duplicate_percentage'] < 10:
            content += "\n⚠️ Moderate duplication"
        else:
            content += "\n🚨 High duplication"
        
        return Panel(content, title="🔄 Duplicates", border_style="yellow")
    
    def _get_numeric_columns(self) -> List[str]:
        return numeric_columns(self.df)

    def _get_categorical_columns(self) -> List[str]:
        return categorical_columns(self.df)

    def _get_datetime_columns(self) -> List[str]:
        return datetime_columns(self.df)

    def _is_numeric_column(self, col: str) -> bool:
        from wrang.utils.constants import is_numeric
        return is_numeric(self.df[col])

    def _is_categorical_column(self, col: str) -> bool:
        from wrang.utils.constants import is_categorical
        return is_categorical(self.df[col])

    def _is_datetime_column(self, col: str) -> bool:
        from wrang.utils.constants import is_datetime
        return is_datetime(self.df[col])

    @staticmethod
    def _format_memory(memory_mb: float) -> str:
        return format_memory(memory_mb)


# Convenience function
def inspect_data(df: pl.DataFrame) -> DataInspector:
    """Create and return DataInspector instance"""
    return DataInspector(df)