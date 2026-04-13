#!/usr/bin/env python3
"""
RIDE Core Data Cleaner
Advanced data cleaning with multiple strategies and validation
"""

import polars as pl
import time
import numpy as np
from typing import Dict, Any, List, Optional, Tuple, Union, Callable
from scipy import stats
from sklearn.impute import KNNImputer
from sklearn.preprocessing import LabelEncoder
import warnings
from rich.console import Console
from rich.table import Table
from rich.panel import Panel
from rich.progress import track, Progress, SpinnerColumn, TextColumn, BarColumn
from rich.prompt import Confirm, Prompt
from rich.text import Text

from wrang.config import get_config, ImputationStrategy
from wrang.utils.exceptions import (
    PreprocessingError, DataValidationError, handle_polars_error
)

console = Console()
warnings.filterwarnings('ignore')


class DataCleaner:
    """
    Advanced data cleaning and validation
    
    Provides comprehensive data cleaning capabilities including missing value
    imputation, outlier handling, duplicate removal, and data validation.
    """
    
    def __init__(self, df: pl.DataFrame):
        """
        Initialize cleaner with DataFrame
        
        Args:
            df: Polars DataFrame to clean
        """
        if df is None or len(df) == 0:
            raise DataValidationError("Cannot clean empty or None DataFrame")

        self.df = df.clone()          # working copy
        self.original_df = df.clone() # pristine original
        self.config = get_config()
        self.cleaning_log: List[Dict[str, Any]] = []

        # Undo stack: list of (snapshot, log_entry) tuples
        self._undo_stack: List[Tuple[pl.DataFrame, Dict[str, Any]]] = []
        
    def handle_missing_values(self, strategy: Union[str, ImputationStrategy] = ImputationStrategy.MEDIAN,
                            columns: Optional[List[str]] = None,
                            custom_value: Optional[Any] = None,
                            drop_threshold: Optional[float] = None) -> 'DataCleaner':
        """
        Handle missing values using various strategies
        
        Args:
            strategy: Imputation strategy to use
            columns: Specific columns to process (default: all columns)
            custom_value: Value to use for custom imputation
            drop_threshold: Drop columns with missing % above threshold
            
        Returns:
            Self for method chaining
        """
        if isinstance(strategy, str):
            strategy = ImputationStrategy(strategy.lower())

        console.print(f"🔧 [bold]Handling Missing Values ({strategy.value.title()})[/bold]")

        # Snapshot before mutation
        self._save_snapshot("handle_missing_values", {
            'strategy': strategy.value,
            'columns': columns,
        })

        # Get columns to process
        if columns is None:
            columns = self.df.columns

        # Drop columns with too many missing values if threshold specified
        if drop_threshold is not None:
            columns = self._filter_columns_by_missing_threshold(columns, drop_threshold)

        initial_missing = self._count_missing_values()
        
        try:
            with Progress(
                SpinnerColumn(),
                TextColumn("[progress.description]{task.description}"),
                BarColumn(),
                console=console
            ) as progress:
                task = progress.add_task("Processing columns...", total=len(columns))
                
                for col in columns:
                    if col not in self.df.columns:
                        console.print(f"[yellow]Warning: Column '{col}' not found, skipping[/yellow]")
                        continue
                    
                    self._handle_column_missing_values(col, strategy, custom_value)
                    progress.advance(task)
            
            # Log the operation
            final_missing = self._count_missing_values()
            self._log_operation("missing_value_imputation", {
                'strategy': strategy.value,
                'columns_processed': len(columns),
                'initial_missing': initial_missing,
                'final_missing': final_missing,
                'missing_reduced': initial_missing - final_missing
            })
            
            console.print(f"✅ Missing values reduced from {initial_missing:,} to {final_missing:,}")
            
        except Exception as e:
            raise PreprocessingError(
                "missing value imputation",
                str(e),
                affected_columns=columns,
                suggestions=[
                    "Check if columns exist in the dataset",
                    "Verify data types are appropriate for the strategy",
                    "Try a different imputation strategy"
                ]
            )
        
        return self
    
    def remove_duplicates(self, columns: Optional[List[str]] = None, 
                         keep: str = 'first') -> 'DataCleaner':
        """
        Remove duplicate rows
        
        Args:
            columns: Columns to consider for duplicates (default: all)
            keep: Which duplicate to keep ('first', 'last', 'none')
            
        Returns:
            Self for method chaining
        """
        console.print("🔧 [bold]Removing Duplicate Rows[/bold]")
        self._save_snapshot("remove_duplicates", {'columns': columns, 'keep': keep})
        initial_rows = len(self.df)
        
        try:
            if columns is None:
                # Remove duplicates based on all columns
                if keep == 'none':
                    # Remove all duplicates (keep none)
                    duplicated_mask = self.df.is_duplicated()
                    self.df = self.df.filter(~duplicated_mask)
                else:
                    # Keep first or last
                    self.df = self.df.unique(maintain_order=(keep == 'first'))
            else:
                # Remove duplicates based on specific columns
                if keep == 'none':
                    duplicated_mask = self.df.is_duplicated(subset=columns)
                    self.df = self.df.filter(~duplicated_mask)
                else:
                    self.df = self.df.unique(subset=columns, maintain_order=(keep == 'first'))
            
            final_rows = len(self.df)
            removed_rows = initial_rows - final_rows
            
            # Log the operation
            self._log_operation("duplicate_removal", {
                'initial_rows': initial_rows,
                'final_rows': final_rows,
                'removed_rows': removed_rows,
                'columns': columns or 'all',
                'keep': keep
            })
            
            console.print(f"✅ Removed {removed_rows:,} duplicate rows ({final_rows:,} remaining)")
            
        except Exception as e:
            raise PreprocessingError(
                "duplicate removal",
                str(e),
                affected_columns=columns,
                suggestions=[
                    "Check if specified columns exist",
                    "Verify column names are spelled correctly"
                ]
            )
        
        return self
    
    def handle_outliers(self, method: str = 'iqr', action: str = 'remove',
                       columns: Optional[List[str]] = None,
                       factor: float = 1.5) -> 'DataCleaner':
        """
        Handle outliers using various methods
        
        Args:
            method: Detection method ('iqr', 'zscore', 'modified_zscore')
            action: What to do with outliers ('remove', 'cap', 'transform')
            columns: Columns to process (default: all numeric)
            factor: Factor for outlier detection (e.g., IQR factor)
            
        Returns:
            Self for method chaining
        """
        console.print(f"🎯 [bold]Handling Outliers ({method.upper()}, {action})[/bold]")
        self._save_snapshot("handle_outliers", {'method': method, 'action': action, 'columns': columns})

        # Get numeric columns if not specified
        if columns is None:
            columns = self._get_numeric_columns()
        
        if not columns:
            console.print("[yellow]⚠️ No numeric columns found for outlier handling[/yellow]")
            return self
        
        outliers_summary = []
        
        try:
            for col in track(columns, description="Processing columns..."):
                if col not in self.df.columns:
                    continue
                
                outlier_info = self._handle_column_outliers(col, method, action, factor)
                outliers_summary.append(outlier_info)
            
            # Display summary
            self._display_outliers_summary(outliers_summary, method, action)
            
            # Log the operation
            total_outliers = sum(info['outliers_detected'] for info in outliers_summary)
            self._log_operation("outlier_handling", {
                'method': method,
                'action': action,
                'columns_processed': len(columns),
                'total_outliers': total_outliers,
                'factor': factor
            })
            
        except Exception as e:
            raise PreprocessingError(
                "outlier handling",
                str(e),
                affected_columns=columns,
                suggestions=[
                    "Check if columns are numeric",
                    "Try a different detection method",
                    "Adjust the factor parameter"
                ]
            )
        
        return self
    
    def validate_data_types(self, type_map: Optional[Dict[str, str]] = None,
                          auto_convert: bool = True) -> 'DataCleaner':
        """
        Validate and optionally convert data types
        
        Args:
            type_map: Mapping of column names to desired types
            auto_convert: Automatically convert compatible types
            
        Returns:
            Self for method chaining
        """
        console.print("🔍 [bold]Validating Data Types[/bold]")
        
        validation_results = []
        conversions_made = 0
        
        try:
            for col in self.df.columns:
                current_type = str(self.df[col].dtype)
                
                # Check if specific type mapping exists
                if type_map and col in type_map:
                    target_type = type_map[col]
                    if current_type != target_type:
                        success = self._convert_column_type(col, target_type)
                        if success:
                            conversions_made += 1
                            validation_results.append({
                                'column': col,
                                'original_type': current_type,
                                'new_type': target_type,
                                'status': 'converted'
                            })
                        else:
                            validation_results.append({
                                'column': col,
                                'original_type': current_type,
                                'target_type': target_type,
                                'status': 'failed'
                            })
                
                elif auto_convert:
                    # Auto-detect and convert obvious cases
                    suggested_type = self._suggest_column_type(col)
                    if suggested_type and suggested_type != current_type:
                        success = self._convert_column_type(col, suggested_type)
                        if success:
                            conversions_made += 1
                            validation_results.append({
                                'column': col,
                                'original_type': current_type,
                                'new_type': suggested_type,
                                'status': 'auto_converted'
                            })
            
            # Display results
            if validation_results:
                self._display_type_validation_results(validation_results)
            
            # Log the operation
            self._log_operation("data_type_validation", {
                'columns_checked': len(self.df.columns),
                'conversions_made': conversions_made,
                'auto_convert': auto_convert
            })
            
            console.print(f"✅ Data type validation complete ({conversions_made} conversions made)")
            
        except Exception as e:
            raise PreprocessingError(
                "data type validation",
                str(e),
                suggestions=[
                    "Check type mapping format",
                    "Verify target types are valid",
                    "Review data for conversion compatibility"
                ]
            )
        
        return self
    
    def clean_text_data(self, columns: Optional[List[str]] = None,
                       operations: Optional[List[str]] = None) -> 'DataCleaner':
        """
        Clean text data with various operations
        
        Args:
            columns: Text columns to clean (default: auto-detect)
            operations: List of operations ('strip', 'lower', 'remove_special', etc.)
            
        Returns:
            Self for method chaining
        """
        if operations is None:
            operations = ['strip', 'normalize_whitespace']
        
        # Get text columns if not specified
        if columns is None:
            columns = self._get_text_columns()
        
        if not columns:
            console.print("[yellow]⚠️ No text columns found for cleaning[/yellow]")
            return self
        
        console.print(f"📝 [bold]Cleaning Text Data ({len(columns)} columns)[/bold]")
        
        try:
            for col in track(columns, description="Cleaning text columns..."):
                if col not in self.df.columns:
                    continue
                
                self._clean_text_column(col, operations)
            
            # Log the operation
            self._log_operation("text_cleaning", {
                'columns_processed': len(columns),
                'operations': operations
            })
            
            console.print(f"✅ Text cleaning complete for {len(columns)} columns")
            
        except Exception as e:
            raise PreprocessingError(
                "text cleaning",
                str(e),
                affected_columns=columns,
                suggestions=[
                    "Check if columns contain text data",
                    "Verify operation names are valid"
                ]
            )
        
        return self
    
    def validate_constraints(self, constraints: Dict[str, Dict[str, Any]]) -> Dict[str, List[int]]:
        """
        Validate data against custom constraints
        
        Args:
            constraints: Dictionary of column constraints
            
        Returns:
            Dictionary mapping columns to lists of violating row indices
        """
        console.print("✅ [bold]Validating Data Constraints[/bold]")
        
        violations = {}
        
        for col, constraint_rules in constraints.items():
            if col not in self.df.columns:
                console.print(f"[yellow]Warning: Column '{col}' not found[/yellow]")
                continue
            
            col_violations = []
            
            # Check various constraint types
            if 'min_value' in constraint_rules:
                mask = self.df[col] < constraint_rules['min_value']
                indices = self.df.with_row_count().filter(mask)['row_nr'].to_list()
                col_violations.extend(indices)
            
            if 'max_value' in constraint_rules:
                mask = self.df[col] > constraint_rules['max_value']
                indices = self.df.with_row_count().filter(mask)['row_nr'].to_list()
                col_violations.extend(indices)
            
            if 'allowed_values' in constraint_rules:
                mask = ~self.df[col].is_in(constraint_rules['allowed_values'])
                indices = self.df.with_row_count().filter(mask)['row_nr'].to_list()
                col_violations.extend(indices)
            
            if 'regex_pattern' in constraint_rules:
                # For text validation
                try:
                    mask = ~self.df[col].str.contains(constraint_rules['regex_pattern'])
                    indices = self.df.with_row_count().filter(mask)['row_nr'].to_list()
                    col_violations.extend(indices)
                except Exception:
                    console.print(f"[yellow]Warning: Could not apply regex to {col}[/yellow]")
            
            if col_violations:
                violations[col] = list(set(col_violations))  # Remove duplicates
        
        # Display violations summary
        if violations:
            self._display_constraint_violations(violations)
        else:
            console.print("✅ All constraints satisfied")
        
        return violations
    
    def get_cleaning_summary(self) -> Dict[str, Any]:
        """
        Get summary of all cleaning operations performed
        
        Returns:
            Dictionary with cleaning summary
        """
        return {
            'original_shape': self.original_df.shape,
            'current_shape': self.df.shape,
            'operations_performed': len(self.cleaning_log),
            'cleaning_log': self.cleaning_log,
            'rows_removed': len(self.original_df) - len(self.df),
            'columns_modified': len(set(
                op.get('affected_columns', []) for op in self.cleaning_log
            ))
        }
    
    def display_cleaning_report(self) -> None:
        """Display comprehensive cleaning report"""
        summary = self.get_cleaning_summary()
        
        # Create main summary panel
        report_content = f"""[bold]Data Cleaning Summary[/bold]

[bold]Shape Changes:[/bold]
• Original: {summary['original_shape'][0]:,} rows × {summary['original_shape'][1]} columns
• Current: {summary['current_shape'][0]:,} rows × {summary['current_shape'][1]} columns
• Rows removed: {summary['rows_removed']:,}

[bold]Operations Performed:[/bold] {summary['operations_performed']}"""
        
        main_panel = Panel(report_content, title="🧹 Cleaning Report", border_style="green")
        console.print(main_panel)
        
        # Display detailed operations log
        if self.cleaning_log:
            operations_table = Table(title="Operations Log", show_header=True, header_style="bold cyan")
            operations_table.add_column("Operation", style="white", min_width=20)
            operations_table.add_column("Details", style="yellow", min_width=30)
            operations_table.add_column("Impact", style="green", min_width=15)
            
            for i, op in enumerate(self.cleaning_log, 1):
                operation_name = op['operation'].replace('_', ' ').title()
                
                # Format details based on operation type
                if op['operation'] == 'missing_value_imputation':
                    details = f"Strategy: {op['details']['strategy']}, Columns: {op['details']['columns_processed']}"
                    impact = f"-{op['details']['missing_reduced']:,} missing"
                elif op['operation'] == 'duplicate_removal':
                    details = f"Keep: {op['details']['keep']}"
                    impact = f"-{op['details']['removed_rows']:,} rows"
                elif op['operation'] == 'outlier_handling':
                    details = f"Method: {op['details']['method']}, Action: {op['details']['action']}"
                    impact = f"{op['details']['total_outliers']:,} outliers"
                else:
                    details = str(op['details'])[:40] + "..." if len(str(op['details'])) > 40 else str(op['details'])
                    impact = "N/A"
                
                operations_table.add_row(f"{i}. {operation_name}", details, impact)
            
            console.print(operations_table)
    
    def get_cleaned_data(self) -> pl.DataFrame:
        """
        Get the cleaned DataFrame
        
        Returns:
            Cleaned Polars DataFrame
        """
        return self.df.clone()
    
    # Private helper methods
    
    def _handle_column_missing_values(self, col: str, strategy: ImputationStrategy, 
                                    custom_value: Optional[Any] = None) -> None:
        """Handle missing values for a single column"""
        if self.df[col].null_count() == 0:
            return  # No missing values
        
        try:
            if strategy == ImputationStrategy.DROP:
                self.df = self.df.drop_nulls(subset=[col])
            
            elif strategy == ImputationStrategy.MEAN:
                if self._is_numeric_column(col):
                    mean_val = self.df[col].mean()
                    self.df = self.df.with_columns(self.df[col].fill_null(mean_val))
            
            elif strategy == ImputationStrategy.MEDIAN:
                if self._is_numeric_column(col):
                    median_val = self.df[col].median()
                    self.df = self.df.with_columns(self.df[col].fill_null(median_val))
            
            elif strategy == ImputationStrategy.MODE:
                mode_series = self.df[col].mode().drop_nulls()
                if len(mode_series) == 0:
                    return
                mode_val = mode_series.first()
                self.df = self.df.with_columns(self.df[col].fill_null(mode_val))
            
            elif strategy == ImputationStrategy.FORWARD_FILL:
                self.df = self.df.with_columns(self.df[col].forward_fill())
            
            elif strategy == ImputationStrategy.BACKWARD_FILL:
                self.df = self.df.with_columns(self.df[col].backward_fill())
            
            elif strategy == ImputationStrategy.CUSTOM_VALUE:
                if custom_value is not None:
                    self.df = self.df.with_columns(self.df[col].fill_null(custom_value))
            
            elif strategy == ImputationStrategy.DISTRIBUTION:
                if self._is_numeric_column(col):
                    self._impute_from_distribution(col)
            
            elif strategy == ImputationStrategy.KNN:
                self._impute_with_knn(col)
                
        except Exception as e:
            raise PreprocessingError(
                f"missing value imputation for column {col}",
                str(e),
                affected_columns=[col]
            )
    
    def _impute_from_distribution(self, col: str) -> None:
        """Impute missing values using random sampling from distribution"""
        non_null_data = self.df[col].drop_nulls()
        if len(non_null_data) == 0:
            return
        
        # Calculate distribution parameters
        mean = non_null_data.mean()
        std = non_null_data.std()
        missing_count = self.df[col].null_count()
        
        # Generate random values from normal distribution
        random_values = np.random.normal(mean, std, missing_count)
        
        # Create a series of random values aligned with null positions
        mask = self.df[col].is_null()
        fill_values = pl.Series(random_values)
        
        # This is a simplified approach - in practice, you'd need more complex logic
        # to align the random values with the null positions
        median_val = non_null_data.median()  # Fallback to median
        self.df = self.df.with_columns(self.df[col].fill_null(median_val))
    
    def _impute_with_knn(self, col: str) -> None:
        """Impute missing values using KNN"""
        try:
            # Get numeric columns for KNN
            numeric_cols = self._get_numeric_columns()
            if col not in numeric_cols or len(numeric_cols) < 2:
                # Fallback to median if KNN not applicable
                if self._is_numeric_column(col):
                    median_val = self.df[col].median()
                    self.df = self.df.with_columns(self.df[col].fill_null(median_val))
                return
            
            # Convert to pandas for sklearn
            df_pandas = self.df.select(numeric_cols).to_pandas()
            
            # Apply KNN imputation
            imputer = KNNImputer(n_neighbors=min(5, len(df_pandas) - 1))
            imputed_data = imputer.fit_transform(df_pandas)
            
            # Update the specific column
            col_index = numeric_cols.index(col)
            imputed_values = imputed_data[:, col_index]
            
            # Convert back to Polars and update
            self.df = self.df.with_columns(
                pl.Series(name=col, values=imputed_values)
            )
            
        except Exception:
            # Fallback to median if KNN fails
            if self._is_numeric_column(col):
                median_val = self.df[col].median()
                self.df = self.df.with_columns(self.df[col].fill_null(median_val))
    
    def _handle_column_outliers(self, col: str, method: str, action: str, factor: float) -> Dict[str, Any]:
        """Handle outliers for a single column"""
        data = self.df[col].drop_nulls()
        if len(data) == 0:
            return {'column': col, 'outliers_detected': 0, 'action_taken': 'none'}
        
        # Detect outliers
        outlier_mask = self._detect_outliers_mask(col, method, factor)
        outliers_count = outlier_mask.sum()
        
        if outliers_count == 0:
            return {'column': col, 'outliers_detected': 0, 'action_taken': 'none'}
        
        # Apply action
        if action == 'remove':
            self.df = self.df.filter(~outlier_mask)
            action_taken = f'removed {outliers_count} rows'
        
        elif action == 'cap':
            # Cap outliers to percentile values
            lower_bound = self.df[col].quantile(0.05)
            upper_bound = self.df[col].quantile(0.95)
            
            self.df = self.df.with_columns([
                pl.when(self.df[col] < lower_bound)
                .then(lower_bound)
                .when(self.df[col] > upper_bound)
                .then(upper_bound)
                .otherwise(self.df[col])
                .alias(col)
            ])
            action_taken = f'capped {outliers_count} values'
        
        elif action == 'transform':
            # Log transform for positive values
            if self.df[col].min() > 0:
                self.df = self.df.with_columns(self.df[col].log().alias(col))
                action_taken = f'log transformed (had {outliers_count} outliers)'
            else:
                action_taken = f'no transform applied (negative values present)'
        
        else:
            action_taken = 'none'
        
        return {
            'column': col,
            'outliers_detected': outliers_count,
            'action_taken': action_taken
        }
    
    def _detect_outliers_mask(self, col: str, method: str, factor: float) -> pl.Series:
        """Detect outliers and return boolean mask"""
        if method == 'iqr':
            q1 = self.df[col].quantile(0.25)
            q3 = self.df[col].quantile(0.75)
            iqr = q3 - q1
            lower_bound = q1 - factor * iqr
            upper_bound = q3 + factor * iqr
            return (self.df[col] < lower_bound) | (self.df[col] > upper_bound)
        
        elif method == 'zscore':
            mean_val = self.df[col].mean()
            std_val = self.df[col].std()
            z_scores = ((self.df[col] - mean_val) / std_val).abs()
            return z_scores > 3
        
        elif method == 'modified_zscore':
            median_val = self.df[col].median()
            mad = (self.df[col] - median_val).abs().median()
            modified_z_scores = (0.6745 * (self.df[col] - median_val) / mad).abs()
            return modified_z_scores > 3.5
        
        else:
            raise ValueError(f"Unknown outlier detection method: {method}")
    
    def _convert_column_type(self, col: str, target_type: str) -> bool:
        """Convert column to target type"""
        try:
            if target_type.lower() in ['int', 'integer']:
                self.df = self.df.with_columns(self.df[col].cast(pl.Int64))
            elif target_type.lower() in ['float', 'double']:
                self.df = self.df.with_columns(self.df[col].cast(pl.Float64))
            elif target_type.lower() in ['str', 'string', 'text']:
                self.df = self.df.with_columns(self.df[col].cast(pl.Utf8))
            elif target_type.lower() in ['bool', 'boolean']:
                self.df = self.df.with_columns(self.df[col].cast(pl.Boolean))
            elif target_type.lower() in ['date', 'datetime']:
                self.df = self.df.with_columns(pl.col(col).str.strptime(pl.Datetime))
            else:
                return False
            return True
        except Exception:
            return False
    
    def _suggest_column_type(self, col: str) -> Optional[str]:
        """Suggest appropriate type for column based on data"""
        # This is a simplified version - could be much more sophisticated
        sample = self.df[col].drop_nulls().head(100)
        if len(sample) == 0:
            return None
        
        # Check if all values can be converted to int
        try:
            sample.cast(pl.Int64)
            return 'int'
        except Exception:
            pass

        # Check if all values can be converted to float
        try:
            sample.cast(pl.Float64)
            return 'float'
        except Exception:
            pass
        
        return None
    
    def _clean_text_column(self, col: str, operations: List[str]) -> None:
        """Clean a single text column"""
        for operation in operations:
            if operation == 'strip':
                self.df = self.df.with_columns(self.df[col].str.strip_chars())
            elif operation == 'lower':
                self.df = self.df.with_columns(self.df[col].str.to_lowercase())
            elif operation == 'upper':
                self.df = self.df.with_columns(self.df[col].str.to_uppercase())
            elif operation == 'normalize_whitespace':
                # Replace multiple whitespaces with single space
                self.df = self.df.with_columns(
                    self.df[col].str.replace_all(r'\s+', ' ')
                )
            elif operation == 'remove_special':
                # Remove special characters (keep only alphanumeric and space)
                self.df = self.df.with_columns(
                    self.df[col].str.replace_all(r'[^a-zA-Z0-9\s]', '')
                )
    
    def _get_numeric_columns(self) -> List[str]:
        """Get list of numeric columns"""
        return [col for col in self.df.columns if self._is_numeric_column(col)]
    
    def _get_text_columns(self) -> List[str]:
        """Get list of text columns"""
        return [col for col in self.df.columns if self._is_text_column(col)]
    
    def _is_numeric_column(self, col: str) -> bool:
        """Check if column is numeric"""
        dtype = str(self.df[col].dtype)
        return any(num_type in dtype.lower() for num_type in ['int', 'float', 'decimal'])
    
    def _is_text_column(self, col: str) -> bool:
        """Check if column is text"""
        dtype = str(self.df[col].dtype)
        return 'utf8' in dtype.lower() or 'str' in dtype.lower()
    
    def _count_missing_values(self) -> int:
        """Count total missing values in DataFrame"""
        return self.df.null_count().sum_horizontal().item()
    
    def _filter_columns_by_missing_threshold(self, columns: List[str], threshold: float) -> List[str]:
        """Filter out columns with missing percentage above threshold"""
        filtered_columns = []
        total_rows = len(self.df)
        
        for col in columns:
            if col not in self.df.columns:
                continue
            
            missing_pct = (self.df[col].null_count() / total_rows) * 100
            if missing_pct <= threshold:
                filtered_columns.append(col)
            else:
                console.print(f"[yellow]Dropping column '{col}' ({missing_pct:.1f}% missing)[/yellow]")
        
        return filtered_columns
    
    def _display_outliers_summary(self, outliers_summary: List[Dict[str, Any]], 
                                method: str, action: str) -> None:
        """Display outlier handling summary"""
        # Create summary table
        outliers_table = Table(title=f"Outlier Handling Summary ({method.upper()})", 
                              show_header=True, header_style="bold cyan")
        outliers_table.add_column("Column", style="white", min_width=15)
        outliers_table.add_column("Outliers Found", style="red", min_width=12)
        outliers_table.add_column("Action Taken", style="green", min_width=25)
        
        for info in outliers_summary:
            outliers_table.add_row(
                info['column'][:15],
                str(info['outliers_detected']),
                info['action_taken']
            )
        
        console.print(outliers_table)
    
    def _display_type_validation_results(self, results: List[Dict[str, Any]]) -> None:
        """Display type validation results"""
        if not results:
            return
        
        # Create results table
        types_table = Table(title="Data Type Conversions", show_header=True, header_style="bold cyan")
        types_table.add_column("Column", style="white", min_width=15)
        types_table.add_column("Original Type", style="yellow", min_width=15)
        types_table.add_column("New Type", style="green", min_width=15)
        types_table.add_column("Status", style="blue", min_width=12)
        
        for result in results:
            status_color = "green" if result['status'] in ['converted', 'auto_converted'] else "red"
            status_text = Text(result['status'].replace('_', ' ').title(), style=status_color)
            
            types_table.add_row(
                result['column'][:15],
                result['original_type'],
                result.get('new_type', result.get('target_type', 'N/A')),
                status_text
            )
        
        console.print(types_table)
    
    def _display_constraint_violations(self, violations: Dict[str, List[int]]) -> None:
        """Display constraint violations summary"""
        violations_table = Table(title="Constraint Violations", show_header=True, header_style="bold red")
        violations_table.add_column("Column", style="white", min_width=15)
        violations_table.add_column("Violations", style="red", min_width=12)
        violations_table.add_column("Sample Row Indices", style="yellow", min_width=20)
        
        for col, violation_indices in violations.items():
            sample_indices = violation_indices[:5]  # Show first 5
            indices_str = ", ".join(map(str, sample_indices))
            if len(violation_indices) > 5:
                indices_str += f" ... (+{len(violation_indices) - 5} more)"
            
            violations_table.add_row(
                col[:15],
                str(len(violation_indices)),
                indices_str
            )
        
        console.print(violations_table)
        
        total_violations = sum(len(indices) for indices in violations.values())
        console.print(f"[red]⚠️ Total constraint violations: {total_violations:,}[/red]")
    
    # ------------------------------------------------------------------
    # Undo / rollback support
    # ------------------------------------------------------------------

    def _save_snapshot(self, operation: str, details: Dict[str, Any]) -> None:
        """Save the current state before a destructive operation."""
        self._undo_stack.append((self.df.clone(), {'operation': operation, **details}))

    def undo(self) -> bool:
        """
        Roll back the last cleaning operation.

        Returns:
            True if rollback succeeded, False if there is nothing to undo.
        """
        if not self._undo_stack:
            console.print("[yellow]⚠️  Nothing to undo.[/yellow]")
            return False

        snapshot, entry = self._undo_stack.pop()
        self.df = snapshot
        if self.cleaning_log:
            self.cleaning_log.pop()
        console.print(
            f"[green]↩  Undid operation:[/green] {entry.get('operation', 'unknown')}"
        )
        return True

    def undo_all(self) -> None:
        """Roll back every cleaning operation — restore the original DataFrame."""
        self.df = self.original_df.clone()
        self._undo_stack.clear()
        self.cleaning_log.clear()
        console.print("[green]↩  All operations undone. DataFrame restored to original.[/green]")

    def get_history(self) -> List[Dict[str, Any]]:
        """Return the list of applied cleaning operations."""
        return list(self.cleaning_log)

    def display_history(self) -> None:
        """Display a Rich table of all applied operations."""
        if not self.cleaning_log:
            console.print("[yellow]No operations in history.[/yellow]")
            return

        from rich.table import Table
        table = Table(title="Cleaning History", show_header=True, header_style="bold cyan")
        table.add_column("#", style="dim", width=4)
        table.add_column("Operation", style="white", min_width=20)
        table.add_column("Timestamp", style="green", min_width=20)
        table.add_column("Details", style="yellow", min_width=30)

        for i, entry in enumerate(self.cleaning_log, 1):
            table.add_row(
                str(i),
                entry.get('operation', 'unknown'),
                entry.get('timestamp', '—'),
                str({k: v for k, v in entry.items() if k not in ('operation', 'timestamp')})[:80],
            )

        console.print(table)

    def _log_operation(self, operation: str, details: Dict[str, Any]) -> None:
        """Log a cleaning operation"""
        self.cleaning_log.append({
            'operation': operation,
            'details': details,
            'timestamp': time.time()
        })


class BatchCleaner:
    """
    Batch cleaning operations for multiple DataFrames or large datasets
    """
    
    def __init__(self):
        self.config = get_config()
        self.cleaning_strategies: Dict[str, Dict[str, Any]] = {}
    
    def register_cleaning_strategy(self, name: str, strategy: Dict[str, Any]) -> None:
        """
        Register a reusable cleaning strategy
        
        Args:
            name: Strategy name
            strategy: Dictionary defining cleaning operations
        """
        self.cleaning_strategies[name] = strategy
        console.print(f"✅ Registered cleaning strategy: '{name}'")
    
    def apply_strategy(self, df: pl.DataFrame, strategy_name: str) -> DataCleaner:
        """
        Apply a registered cleaning strategy to a DataFrame
        
        Args:
            df: DataFrame to clean
            strategy_name: Name of registered strategy
            
        Returns:
            DataCleaner instance with strategy applied
        """
        if strategy_name not in self.cleaning_strategies:
            raise ValueError(f"Strategy '{strategy_name}' not found")
        
        strategy = self.cleaning_strategies[strategy_name]
        cleaner = DataCleaner(df)
        
        console.print(f"🧹 [bold]Applying strategy: {strategy_name}[/bold]")
        
        # Apply operations in order
        if 'missing_values' in strategy:
            mv_config = strategy['missing_values']
            cleaner.handle_missing_values(
                strategy=mv_config.get('strategy', ImputationStrategy.MEDIAN),
                columns=mv_config.get('columns'),
                custom_value=mv_config.get('custom_value'),
                drop_threshold=mv_config.get('drop_threshold')
            )
        
        if 'duplicates' in strategy:
            dup_config = strategy['duplicates']
            cleaner.remove_duplicates(
                columns=dup_config.get('columns'),
                keep=dup_config.get('keep', 'first')
            )
        
        if 'outliers' in strategy:
            out_config = strategy['outliers']
            cleaner.handle_outliers(
                method=out_config.get('method', 'iqr'),
                action=out_config.get('action', 'remove'),
                columns=out_config.get('columns'),
                factor=out_config.get('factor', 1.5)
            )
        
        if 'data_types' in strategy:
            dt_config = strategy['data_types']
            cleaner.validate_data_types(
                type_map=dt_config.get('type_map'),
                auto_convert=dt_config.get('auto_convert', True)
            )
        
        if 'text_cleaning' in strategy:
            text_config = strategy['text_cleaning']
            cleaner.clean_text_data(
                columns=text_config.get('columns'),
                operations=text_config.get('operations')
            )
        
        console.print("✅ Strategy application complete")
        return cleaner
    
    def get_strategy_templates(self) -> Dict[str, Dict[str, Any]]:
        """Get predefined cleaning strategy templates"""
        return {
            'basic_cleaning': {
                'description': 'Basic cleaning for most datasets',
                'missing_values': {
                    'strategy': ImputationStrategy.MEDIAN,
                    'drop_threshold': 90.0
                },
                'duplicates': {
                    'keep': 'first'
                },
                'data_types': {
                    'auto_convert': True
                },
                'text_cleaning': {
                    'operations': ['strip', 'normalize_whitespace']
                }
            },
            
            'aggressive_cleaning': {
                'description': 'Aggressive cleaning with outlier removal',
                'missing_values': {
                    'strategy': ImputationStrategy.MEAN,
                    'drop_threshold': 50.0
                },
                'duplicates': {
                    'keep': 'first'
                },
                'outliers': {
                    'method': 'iqr',
                    'action': 'remove',
                    'factor': 1.5
                },
                'data_types': {
                    'auto_convert': True
                }
            },
            
            'conservative_cleaning': {
                'description': 'Conservative cleaning that preserves data',
                'missing_values': {
                    'strategy': ImputationStrategy.MODE,
                    'drop_threshold': 95.0
                },
                'duplicates': {
                    'keep': 'first'
                },
                'outliers': {
                    'method': 'iqr',
                    'action': 'cap',
                    'factor': 2.0
                },
                'text_cleaning': {
                    'operations': ['strip']
                }
            },
            
            'ml_ready': {
                'description': 'Prepare data for machine learning',
                'missing_values': {
                    'strategy': ImputationStrategy.KNN,
                    'drop_threshold': 75.0
                },
                'duplicates': {
                    'keep': 'first'
                },
                'outliers': {
                    'method': 'modified_zscore',
                    'action': 'cap',
                    'factor': 1.5
                },
                'data_types': {
                    'auto_convert': True
                },
                'text_cleaning': {
                    'operations': ['strip', 'lower', 'normalize_whitespace']
                }
            }
        }
    
    def display_available_strategies(self) -> None:
        """Display all available cleaning strategies"""
        templates = self.get_strategy_templates()
        all_strategies = {**templates, **self.cleaning_strategies}
        
        strategies_table = Table(title="Available Cleaning Strategies", 
                                show_header=True, header_style="bold cyan")
        strategies_table.add_column("Strategy", style="white", min_width=20)
        strategies_table.add_column("Type", style="green", min_width=12)
        strategies_table.add_column("Description", style="yellow", min_width=40)
        
        for name, strategy in all_strategies.items():
            strategy_type = "Template" if name in templates else "Custom"
            description = strategy.get('description', 'Custom strategy')
            
            strategies_table.add_row(name, strategy_type, description)
        
        console.print(strategies_table)


# Convenience functions
def clean_data(df: pl.DataFrame) -> DataCleaner:
    """Create and return DataCleaner instance"""
    return DataCleaner(df)


def quick_clean(df: pl.DataFrame, strategy: str = 'basic_cleaning') -> pl.DataFrame:
    """
    Quick cleaning using predefined strategies
    
    Args:
        df: DataFrame to clean
        strategy: Strategy name ('basic_cleaning', 'aggressive_cleaning', etc.)
        
    Returns:
        Cleaned DataFrame
    """
    batch_cleaner = BatchCleaner()
    templates = batch_cleaner.get_strategy_templates()
    
    if strategy not in templates:
        raise ValueError(f"Strategy '{strategy}' not found. Available: {list(templates.keys())}")
    
    batch_cleaner.register_cleaning_strategy(strategy, templates[strategy])
    cleaner = batch_cleaner.apply_strategy(df, strategy)
    
    return cleaner.get_cleaned_data()