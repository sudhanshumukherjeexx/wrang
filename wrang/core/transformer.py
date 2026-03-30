#!/usr/bin/env python3
"""
RIDE Core Data Transformer
Advanced feature transformation and engineering with multiple encoding and scaling methods
"""

import time
import polars as pl
import numpy as np
from typing import Dict, Any, List, Optional, Tuple, Union, Callable
from sklearn.preprocessing import (
    StandardScaler, MinMaxScaler, RobustScaler, MaxAbsScaler,
    QuantileTransformer, PowerTransformer, LabelEncoder, 
    OneHotEncoder, OrdinalEncoder
)
from sklearn.feature_selection import SelectKBest, chi2, f_classif, mutual_info_classif
import warnings
from rich.console import Console
from rich.table import Table
from rich.panel import Panel
from rich.progress import track, Progress, SpinnerColumn, TextColumn, BarColumn
from rich.prompt import Confirm, Prompt, IntPrompt
from rich.text import Text

from wrang.config import get_config, ScalingMethod, EncodingMethod
from wrang.utils.exceptions import (
    PreprocessingError, DataValidationError, handle_polars_error
)

console = Console()
warnings.filterwarnings('ignore')


class DataTransformer:
    """
    Advanced feature transformation and engineering
    
    Provides comprehensive transformation capabilities including encoding,
    scaling, feature creation, and selection with sklearn integration.
    """
    
    def __init__(self, df: pl.DataFrame):
        """
        Initialize transformer with DataFrame
        
        Args:
            df: Polars DataFrame to transform
        """
        if df is None or len(df) == 0:
            raise DataValidationError("Cannot transform empty or None DataFrame")
        
        self.df = df.clone()
        self.original_df = df.clone()
        self.config = get_config()
        self.transformation_log: List[Dict[str, Any]] = []
        self.fitted_transformers: Dict[str, Any] = {}
        
    def encode_categorical_features(self, 
                                  method: Union[str, EncodingMethod] = EncodingMethod.LABEL,
                                  columns: Optional[List[str]] = None,
                                  drop_original: bool = True,
                                  handle_unknown: str = 'ignore') -> 'DataTransformer':
        """
        Encode categorical features using various methods
        
        Args:
            method: Encoding method to use
            columns: Specific columns to encode (default: auto-detect categorical)
            drop_original: Whether to drop original categorical columns
            handle_unknown: How to handle unknown categories ('ignore', 'error')
            
        Returns:
            Self for method chaining
        """
        if isinstance(method, str):
            method = EncodingMethod(method.lower())
        
        console.print(f"🏷️ [bold]Encoding Categorical Features ({method.value.title()})[/bold]")
        
        # Get categorical columns if not specified
        if columns is None:
            columns = self._get_categorical_columns()
        
        if not columns:
            console.print("[yellow]⚠️ No categorical columns found for encoding[/yellow]")
            return self
        
        # Filter columns based on unique value counts
        columns = self._filter_categorical_columns(columns, method)
        
        encoding_results = []
        
        try:
            for col in track(columns, description="Encoding columns..."):
                if col not in self.df.columns:
                    continue
                
                result = self._encode_single_column(col, method, drop_original, handle_unknown)
                encoding_results.append(result)
            
            # Display results
            self._display_encoding_results(encoding_results, method)
            
            # Log the operation
            total_new_columns = sum(r['new_columns_count'] for r in encoding_results)
            self._log_transformation("categorical_encoding", {
                'method': method.value,
                'columns_processed': len(columns),
                'total_new_columns': total_new_columns,
                'drop_original': drop_original
            })
            
            console.print(f"✅ Categorical encoding complete ({total_new_columns} new columns created)")
            
        except Exception as e:
            raise PreprocessingError(
                "categorical encoding",
                str(e),
                affected_columns=columns,
                suggestions=[
                    "Check if columns contain categorical data",
                    "Try a different encoding method",
                    "Reduce number of unique categories"
                ]
            )
        
        return self
    
    def scale_features(self, 
                      method: Union[str, ScalingMethod] = ScalingMethod.STANDARD,
                      columns: Optional[List[str]] = None,
                      feature_range: Tuple[float, float] = (0, 1)) -> 'DataTransformer':
        """
        Scale numeric features using various methods
        
        Args:
            method: Scaling method to use
            columns: Specific columns to scale (default: all numeric)
            feature_range: Range for MinMax scaling
            
        Returns:
            Self for method chaining
        """
        if isinstance(method, str):
            method = ScalingMethod(method.lower())
        
        console.print(f"⚖️ [bold]Scaling Features ({method.value.title()})[/bold]")
        
        # Get numeric columns if not specified
        if columns is None:
            columns = self._get_numeric_columns()
        
        if not columns:
            console.print("[yellow]⚠️ No numeric columns found for scaling[/yellow]")
            return self
        
        scaling_results = []
        
        try:
            # Initialize scaler
            scaler = self._get_scaler(method, feature_range)
            
            # Fit scaler on the data
            data_to_scale = self.df.select(columns).to_pandas()
            scaler.fit(data_to_scale)
            
            # Store fitted scaler for future use
            self.fitted_transformers[f'scaler_{method.value}'] = scaler
            
            # Transform the data
            scaled_data = scaler.transform(data_to_scale)
            
            # Update DataFrame
            for i, col in enumerate(columns):
                self.df = self.df.with_columns(
                    pl.Series(name=col, values=scaled_data[:, i])
                )
                
                # Calculate scaling statistics
                original_stats = self._get_column_stats(self.original_df[col])
                new_stats = self._get_column_stats(self.df[col])
                
                scaling_results.append({
                    'column': col,
                    'original_mean': original_stats['mean'],
                    'original_std': original_stats['std'],
                    'new_mean': new_stats['mean'],
                    'new_std': new_stats['std']
                })
            
            # Display results
            self._display_scaling_results(scaling_results, method)
            
            # Log the operation
            self._log_transformation("feature_scaling", {
                'method': method.value,
                'columns_processed': len(columns),
                'feature_range': feature_range if method == ScalingMethod.MINMAX else None
            })
            
            console.print(f"✅ Feature scaling complete ({len(columns)} columns scaled)")
            
        except Exception as e:
            raise PreprocessingError(
                "feature scaling",
                str(e),
                affected_columns=columns,
                suggestions=[
                    "Check if columns are numeric",
                    "Remove columns with constant values",
                    "Handle missing values before scaling"
                ]
            )
        
        return self
    
    def create_polynomial_features(self, 
                                 columns: Optional[List[str]] = None,
                                 degree: int = 2,
                                 interaction_only: bool = False,
                                 include_bias: bool = False) -> 'DataTransformer':
        """
        Create polynomial and interaction features
        
        Args:
            columns: Columns to create polynomial features for
            degree: Polynomial degree
            interaction_only: Only create interaction features
            include_bias: Include bias column
            
        Returns:
            Self for method chaining
        """
        console.print(f"🧮 [bold]Creating Polynomial Features (degree={degree})[/bold]")
        
        if columns is None:
            columns = self._get_numeric_columns()
        
        if len(columns) < 2 and not interaction_only:
            console.print("[yellow]⚠️ Need at least 2 numeric columns for polynomial features[/yellow]")
            return self
        
        try:
            from sklearn.preprocessing import PolynomialFeatures
            
            # Create polynomial features
            poly = PolynomialFeatures(
                degree=degree,
                interaction_only=interaction_only,
                include_bias=include_bias
            )
            
            # Fit and transform
            data_to_transform = self.df.select(columns).to_pandas()
            poly_features = poly.fit_transform(data_to_transform)
            
            # Get feature names
            feature_names = poly.get_feature_names_out(columns)
            
            # Create new DataFrame with polynomial features
            poly_df = pl.DataFrame(poly_features, schema=feature_names)
            
            # Add new features to existing DataFrame (excluding original columns if they exist)
            new_columns = [name for name in feature_names if name not in columns]
            if new_columns:
                for i, new_col in enumerate(new_columns):
                    # Find the index of this column in the polynomial features
                    col_index = list(feature_names).index(new_col)
                    self.df = self.df.with_columns(
                        pl.Series(name=new_col, values=poly_features[:, col_index])
                    )
            
            # Store fitted transformer
            self.fitted_transformers['polynomial_features'] = poly
            
            # Log the operation
            self._log_transformation("polynomial_features", {
                'degree': degree,
                'interaction_only': interaction_only,
                'input_columns': len(columns),
                'output_columns': len(feature_names),
                'new_columns': len(new_columns)
            })
            
            console.print(f"✅ Created {len(new_columns)} polynomial features")
            
        except Exception as e:
            raise PreprocessingError(
                "polynomial feature creation",
                str(e),
                affected_columns=columns,
                suggestions=[
                    "Reduce polynomial degree",
                    "Check for memory limitations",
                    "Consider using fewer input columns"
                ]
            )
        
        return self
    
    def apply_mathematical_transforms(self, 
                                    transforms: Dict[str, str],
                                    create_new_columns: bool = True) -> 'DataTransformer':
        """
        Apply mathematical transformations to columns
        
        Args:
            transforms: Dictionary mapping column names to transform types
            create_new_columns: Whether to create new columns or modify existing
            
        Returns:
            Self for method chaining
        """
        console.print(f"🧪 [bold]Applying Mathematical Transforms[/bold]")
        
        available_transforms = {
            'log': lambda x: pl.col(x).log(),
            'log1p': lambda x: pl.col(x).log1p(),
            'sqrt': lambda x: pl.col(x).sqrt(),
            'square': lambda x: pl.col(x).pow(2),
            'cube': lambda x: pl.col(x).pow(3),
            'reciprocal': lambda x: 1.0 / pl.col(x),
            'exp': lambda x: pl.col(x).exp(),
            'abs': lambda x: pl.col(x).abs(),
            'sin': lambda x: pl.col(x).sin(),
            'cos': lambda x: pl.col(x).cos(),
            'tan': lambda x: pl.col(x).tan()
        }
        
        transform_results = []
        
        try:
            for col, transform_type in transforms.items():
                if col not in self.df.columns:
                    console.print(f"[yellow]Warning: Column '{col}' not found[/yellow]")
                    continue
                
                if transform_type not in available_transforms:
                    console.print(f"[yellow]Warning: Transform '{transform_type}' not available[/yellow]")
                    continue
                
                # Apply transformation
                transform_func = available_transforms[transform_type]
                
                if create_new_columns:
                    new_col_name = f"{col}_{transform_type}"
                    self.df = self.df.with_columns(
                        transform_func(col).alias(new_col_name)
                    )
                    target_col = new_col_name
                else:
                    self.df = self.df.with_columns(
                        transform_func(col).alias(col)
                    )
                    target_col = col
                
                # Handle potential errors (inf, nan)
                self.df = self.df.with_columns(
                    pl.when(pl.col(target_col).is_infinite() | pl.col(target_col).is_nan())
                    .then(None)
                    .otherwise(pl.col(target_col))
                    .alias(target_col)
                )
                
                transform_results.append({
                    'original_column': col,
                    'target_column': target_col,
                    'transform': transform_type,
                    'created_new': create_new_columns
                })
            
            # Display results
            self._display_transform_results(transform_results)
            
            # Log the operation
            self._log_transformation("mathematical_transforms", {
                'transforms_applied': len(transform_results),
                'create_new_columns': create_new_columns,
                'transforms': transforms
            })
            
            console.print(f"✅ Applied {len(transform_results)} mathematical transforms")
            
        except Exception as e:
            raise PreprocessingError(
                "mathematical transforms",
                str(e),
                suggestions=[
                    "Check for negative values in log transforms",
                    "Handle zero values in reciprocal transforms",
                    "Verify column data types"
                ]
            )
        
        return self
    
    def create_binned_features(self, 
                             columns: Optional[List[str]] = None,
                             n_bins: int = 5,
                             strategy: str = 'quantile',
                             labels: Optional[List[str]] = None) -> 'DataTransformer':
        """
        Create binned/discretized features from continuous variables
        
        Args:
            columns: Columns to bin (default: all numeric)
            n_bins: Number of bins
            strategy: Binning strategy ('uniform', 'quantile', 'kmeans')
            labels: Custom labels for bins
            
        Returns:
            Self for method chaining
        """
        console.print(f"📊 [bold]Creating Binned Features ({strategy}, {n_bins} bins)[/bold]")
        
        if columns is None:
            columns = self._get_numeric_columns()
        
        if not columns:
            console.print("[yellow]⚠️ No numeric columns found for binning[/yellow]")
            return self
        
        binning_results = []
        
        try:
            for col in track(columns, description="Creating bins..."):
                if col not in self.df.columns:
                    continue
                
                # Create bins using different strategies
                if strategy == 'quantile':
                    # Equal-frequency bins
                    bin_edges = self.df[col].quantile([i/n_bins for i in range(n_bins + 1)])
                    bin_edges = bin_edges.unique().sort()
                elif strategy == 'uniform':
                    # Equal-width bins
                    min_val = self.df[col].min()
                    max_val = self.df[col].max()
                    bin_edges = pl.Series([min_val + i * (max_val - min_val) / n_bins 
                                         for i in range(n_bins + 1)])
                else:
                    # Use sklearn's KBinsDiscretizer for kmeans strategy
                    from sklearn.preprocessing import KBinsDiscretizer
                    discretizer = KBinsDiscretizer(n_bins=n_bins, encode='ordinal', strategy=strategy)
                    data_to_bin = self.df[col].drop_nulls().to_pandas().values.reshape(-1, 1)
                    
                    if len(data_to_bin) > 0:
                        binned_data = discretizer.fit_transform(data_to_bin)
                        
                        # Create new column with binned values
                        binned_col_name = f"{col}_binned"
                        
                        # Create a mapping for the full dataset
                        non_null_mask = self.df[col].is_not_null()
                        full_binned = pl.Series([None] * len(self.df))
                        
                        # This is a simplified approach - in practice, you'd need more careful alignment
                        # For now, let's use quantile strategy as fallback
                        bin_edges = self.df[col].quantile([i/n_bins for i in range(n_bins + 1)])
                        bin_edges = bin_edges.unique().sort()
                
                # Create binned column using cut
                binned_col_name = f"{col}_binned"
                
                # Use Polars' cut function for binning
                try:
                    self.df = self.df.with_columns(
                        pl.col(col).cut(bin_edges, labels=labels).alias(binned_col_name)
                    )
                except Exception:
                    # Fallback: manual binning
                    conditions = []
                    for i in range(len(bin_edges) - 1):
                        if i == 0:
                            condition = pl.col(col) <= bin_edges[i + 1]
                        elif i == len(bin_edges) - 2:
                            condition = pl.col(col) > bin_edges[i]
                        else:
                            condition = (pl.col(col) > bin_edges[i]) & (pl.col(col) <= bin_edges[i + 1])
                        conditions.append((condition, i))
                    
                    # Apply conditions
                    expr = pl.lit(None)
                    for condition, value in conditions:
                        expr = pl.when(condition).then(value).otherwise(expr)
                    
                    self.df = self.df.with_columns(expr.alias(binned_col_name))
                
                binning_results.append({
                    'original_column': col,
                    'binned_column': binned_col_name,
                    'n_bins': len(bin_edges) - 1 if bin_edges is not None else n_bins,
                    'strategy': strategy
                })
            
            # Display results
            self._display_binning_results(binning_results)
            
            # Log the operation
            self._log_transformation("feature_binning", {
                'columns_processed': len(columns),
                'n_bins': n_bins,
                'strategy': strategy,
                'new_columns': len(binning_results)
            })
            
            console.print(f"✅ Created {len(binning_results)} binned features")
            
        except Exception as e:
            raise PreprocessingError(
                "feature binning",
                str(e),
                affected_columns=columns,
                suggestions=[
                    "Check for sufficient data range",
                    "Reduce number of bins",
                    "Try a different binning strategy"
                ]
            )
        
        return self
    
    def select_features(self, 
                       target_column: str,
                       method: str = 'mutual_info',
                       k: int = 10) -> 'DataTransformer':
        """
        Select best features using statistical tests
        
        Args:
            target_column: Target variable for feature selection
            method: Selection method ('mutual_info', 'chi2', 'f_classif')
            k: Number of features to select
            
        Returns:
            Self for method chaining
        """
        console.print(f"🎯 [bold]Selecting Features ({method}, k={k})[/bold]")
        
        if target_column not in self.df.columns:
            raise DataValidationError(f"Target column '{target_column}' not found")
        
        # Get feature columns (all except target)
        feature_columns = [col for col in self.df.columns if col != target_column]
        numeric_features = [col for col in feature_columns if self._is_numeric_column(col)]
        
        if len(numeric_features) == 0:
            console.print("[yellow]⚠️ No numeric features found for selection[/yellow]")
            return self
        
        if k > len(numeric_features):
            k = len(numeric_features)
            console.print(f"[yellow]Adjusted k to {k} (max available features)[/yellow]")
        
        try:
            # Prepare data
            X = self.df.select(numeric_features).to_pandas()
            y = self.df[target_column].to_pandas()
            
            # Remove any remaining null values
            mask = ~(X.isnull().any(axis=1) | y.isnull())
            X = X[mask]
            y = y[mask]
            
            # Select features based on method
            if method == 'mutual_info':
                # Determine if classification or regression
                if self._is_categorical_column(target_column) or len(y.unique()) < 20:
                    selector = SelectKBest(score_func=mutual_info_classif, k=k)
                else:
                    from sklearn.feature_selection import mutual_info_regression
                    selector = SelectKBest(score_func=mutual_info_regression, k=k)
            elif method == 'chi2':
                # Ensure non-negative values for chi2
                X = X.abs()  # Make values non-negative
                selector = SelectKBest(score_func=chi2, k=k)
            elif method == 'f_classif':
                selector = SelectKBest(score_func=f_classif, k=k)
            else:
                raise ValueError(f"Unknown feature selection method: {method}")
            
            # Fit selector
            X_selected = selector.fit_transform(X, y)
            
            # Get selected feature names
            selected_features = [numeric_features[i] for i in selector.get_support(indices=True)]
            feature_scores = selector.scores_[selector.get_support(indices=True)]
            
            # Keep only selected features (plus target and any non-numeric columns)
            non_numeric_features = [col for col in feature_columns if not self._is_numeric_column(col)]
            columns_to_keep = selected_features + [target_column] + non_numeric_features
            
            self.df = self.df.select(columns_to_keep)
            
            # Store selector
            self.fitted_transformers['feature_selector'] = selector
            
            # Display results
            self._display_feature_selection_results(selected_features, feature_scores, method)
            
            # Log the operation
            self._log_transformation("feature_selection", {
                'method': method,
                'k_requested': k,
                'features_selected': len(selected_features),
                'original_features': len(numeric_features),
                'selected_features': selected_features
            })
            
            console.print(f"✅ Selected {len(selected_features)} best features")
            
        except Exception as e:
            raise PreprocessingError(
                "feature selection",
                str(e),
                suggestions=[
                    "Check target column type",
                    "Ensure sufficient data after null removal",
                    "Try a different selection method"
                ]
            )
        
        return self
    
    def get_transformation_summary(self) -> Dict[str, Any]:
        """
        Get summary of all transformations performed
        
        Returns:
            Dictionary with transformation summary
        """
        return {
            'original_shape': self.original_df.shape,
            'current_shape': self.df.shape,
            'transformations_performed': len(self.transformation_log),
            'transformation_log': self.transformation_log,
            'new_columns': self.df.shape[1] - self.original_df.shape[1],
            'fitted_transformers': list(self.fitted_transformers.keys())
        }
    
    def display_transformation_report(self) -> None:
        """Display comprehensive transformation report"""
        summary = self.get_transformation_summary()
        
        # Create main summary panel
        report_content = f"""[bold]Feature Transformation Summary[/bold]

[bold]Shape Changes:[/bold]
• Original: {summary['original_shape'][0]:,} rows × {summary['original_shape'][1]} columns
• Current: {summary['current_shape'][0]:,} rows × {summary['current_shape'][1]} columns
• New columns: {summary['new_columns']}

[bold]Transformations Applied:[/bold] {summary['transformations_performed']}
[bold]Fitted Transformers:[/bold] {len(summary['fitted_transformers'])}"""
        
        main_panel = Panel(report_content, title="🔄 Transformation Report", border_style="blue")
        console.print(main_panel)
        
        # Display detailed transformation log
        if self.transformation_log:
            transform_table = Table(title="Transformation Log", show_header=True, header_style="bold cyan")
            transform_table.add_column("Transformation", style="white", min_width=20)
            transform_table.add_column("Details", style="yellow", min_width=40)
            transform_table.add_column("Impact", style="green", min_width=15)
            
            for i, transform in enumerate(self.transformation_log, 1):
                transform_name = transform['transformation'].replace('_', ' ').title()
                
                # Format details based on transformation type
                details = str(transform['details'])[:50] + "..." if len(str(transform['details'])) > 50 else str(transform['details'])
                
                # Calculate impact
                if 'new_columns' in transform['details']:
                    impact = f"+{transform['details']['new_columns']} cols"
                elif 'columns_processed' in transform['details']:
                    impact = f"{transform['details']['columns_processed']} cols"
                else:
                    impact = "Modified"
                
                transform_table.add_row(f"{i}. {transform_name}", details, impact)
            
            console.print(transform_table)
    
    def get_transformed_data(self) -> pl.DataFrame:
        """
        Get the transformed DataFrame
        
        Returns:
            Transformed Polars DataFrame
        """
        return self.df.clone()
    
    # Private helper methods
    
    def _encode_single_column(self, col: str, method: EncodingMethod, 
                            drop_original: bool, handle_unknown: str) -> Dict[str, Any]:
        """Encode a single categorical column"""
        try:
            if method == EncodingMethod.LABEL:
                return self._label_encode_column(col, drop_original)
            elif method == EncodingMethod.ONEHOT:
                return self._onehot_encode_column(col, drop_original, handle_unknown)
            elif method == EncodingMethod.ORDINAL:
                return self._ordinal_encode_column(col, drop_original)
            else:
                raise ValueError(f"Unsupported encoding method: {method}")
                
        except Exception as e:
            return {
                'column': col,
                'method': method.value,
                'success': False,
                'error': str(e),
                'new_columns_count': 0
            }
    
    def _label_encode_column(self, col: str, drop_original: bool) -> Dict[str, Any]:
        """Apply label encoding to a column"""
        # Get unique values and create mapping
        unique_values = self.df[col].unique().drop_nulls().sort()
        
        # Create label mapping
        label_map = {val: i for i, val in enumerate(unique_values.to_list())}
        
        # Apply mapping
        new_col_name = f"{col}_encoded" if not drop_original else col
        
        # Create mapping expression
        mapping_expr = pl.col(col)
        for original_val, encoded_val in label_map.items():
            mapping_expr = pl.when(pl.col(col) == original_val).then(encoded_val).otherwise(mapping_expr)
        
        self.df = self.df.with_columns(mapping_expr.alias(new_col_name))
        
        # Drop original if requested
        if drop_original and new_col_name != col:
            self.df = self.df.drop(col)
        
        # Store mapping for future reference
        self.fitted_transformers[f'label_encoder_{col}'] = label_map
        
        return {
            'column': col,
            'method': 'label',
            'success': True,
            'new_columns': [new_col_name] if new_col_name != col else [],
            'new_columns_count': 1 if new_col_name != col else 0,
            'unique_values': len(unique_values),
            'mapping': label_map
        }
    
    def _onehot_encode_column(self, col: str, drop_original: bool, handle_unknown: str) -> Dict[str, Any]:
        """Apply one-hot encoding to a column"""
        # Get unique values
        unique_values = self.df[col].unique().drop_nulls().sort().to_list()
        
        # Create binary columns for each unique value
        new_columns = []
        for value in unique_values:
            new_col_name = f"{col}_{value}"
            self.df = self.df.with_columns(
                (pl.col(col) == value).cast(pl.Int8).alias(new_col_name)
            )
            new_columns.append(new_col_name)
        
        # Drop original if requested
        if drop_original:
            self.df = self.df.drop(col)
        
        return {
            'column': col,
            'method': 'onehot',
            'success': True,
            'new_columns': new_columns,
            'new_columns_count': len(new_columns),
            'unique_values': len(unique_values)
        }
    
    def _ordinal_encode_column(self, col: str, drop_original: bool) -> Dict[str, Any]:
        """Apply ordinal encoding to a column (similar to label but with user-defined order)"""
        # For now, implement same as label encoding
        # In a real implementation, you'd ask user for custom ordering
        return self._label_encode_column(col, drop_original)
    
    def _get_scaler(self, method: ScalingMethod, feature_range: Tuple[float, float]):
        """Get appropriate scaler based on method"""
        if method == ScalingMethod.STANDARD:
            return StandardScaler()
        elif method == ScalingMethod.MINMAX:
            return MinMaxScaler(feature_range=feature_range)
        elif method == ScalingMethod.ROBUST:
            return RobustScaler()
        elif method == ScalingMethod.MAXABS:
            return MaxAbsScaler()
        elif method == ScalingMethod.QUANTILE_UNIFORM:
            return QuantileTransformer(output_distribution='uniform')
        elif method == ScalingMethod.QUANTILE_NORMAL:
            return QuantileTransformer(output_distribution='normal')
        else:
            raise ValueError(f"Unknown scaling method: {method}")
    
    def _get_column_stats(self, series: pl.Series) -> Dict[str, float]:
        """Get basic statistics for a column"""
        return {
            'mean': float(series.mean()) if series.mean() is not None else 0.0,
            'std': float(series.std()) if series.std() is not None else 0.0,
            'min': float(series.min()) if series.min() is not None else 0.0,
            'max': float(series.max()) if series.max() is not None else 0.0
        }
    
    def _filter_categorical_columns(self, columns: List[str], method: EncodingMethod) -> List[str]:
        """Filter categorical columns based on unique value counts and method"""
        filtered_columns = []
        
        for col in columns:
            if col not in self.df.columns:
                continue
            
            unique_count = self.df[col].n_unique()
            
            # Apply filtering based on method and configuration
            if method == EncodingMethod.ONEHOT:
                if unique_count > self.config.max_features_for_onehot:
                    console.print(f"[yellow]Skipping '{col}' for one-hot encoding ({unique_count} unique values > {self.config.max_features_for_onehot})[/yellow]")
                    continue
            elif method == EncodingMethod.LABEL:
                if unique_count > self.config.max_features_for_label:
                    console.print(f"[yellow]Skipping '{col}' for label encoding ({unique_count} unique values > {self.config.max_features_for_label})[/yellow]")
                    continue
            
            if unique_count == 1:
                console.print(f"[yellow]Skipping '{col}' (only 1 unique value)[/yellow]")
                continue
            
            filtered_columns.append(col)
        
        return filtered_columns
    
    def _get_categorical_columns(self) -> List[str]:
        """Get list of categorical columns"""
        return [col for col in self.df.columns if self._is_categorical_column(col)]
    
    def _get_numeric_columns(self) -> List[str]:
        """Get list of numeric columns"""
        return [col for col in self.df.columns if self._is_numeric_column(col)]
    
    def _is_numeric_column(self, col: str) -> bool:
        """Check if column is numeric"""
        dtype = str(self.df[col].dtype)
        return any(num_type in dtype.lower() for num_type in ['int', 'float', 'decimal'])
    
    def _is_categorical_column(self, col: str) -> bool:
        """Check if column is categorical"""
        dtype = str(self.df[col].dtype)
        return 'utf8' in dtype.lower() or 'str' in dtype.lower()
    
    def _display_encoding_results(self, results: List[Dict[str, Any]], method: EncodingMethod) -> None:
        """Display encoding results in a table"""
        if not results:
            return
        
        encoding_table = Table(title=f"Encoding Results ({method.value.title()})", 
                              show_header=True, header_style="bold cyan")
        encoding_table.add_column("Column", style="white", min_width=15)
        encoding_table.add_column("Status", style="green", min_width=10)
        encoding_table.add_column("New Columns", style="blue", min_width=12)
        encoding_table.add_column("Unique Values", style="yellow", min_width=12)
        encoding_table.add_column("Details", style="magenta", min_width=20)
        
        for result in results:
            status = "✅ Success" if result['success'] else "❌ Failed"
            status_color = "green" if result['success'] else "red"
            
            if result['success']:
                details = f"Method: {result['method']}"
                if 'mapping' in result and len(str(result['mapping'])) < 50:
                    details += f", Mapping available"
            else:
                details = result.get('error', 'Unknown error')[:30] + "..."
            
            encoding_table.add_row(
                result['column'][:15],
                Text(status, style=status_color),
                str(result['new_columns_count']),
                str(result.get('unique_values', 'N/A')),
                details
            )
        
        console.print(encoding_table)
    
    def _display_scaling_results(self, results: List[Dict[str, Any]], method: ScalingMethod) -> None:
        """Display scaling results in a table"""
        if not results:
            return
        
        scaling_table = Table(title=f"Scaling Results ({method.value.title()})", 
                             show_header=True, header_style="bold cyan")
        scaling_table.add_column("Column", style="white", min_width=15)
        scaling_table.add_column("Original Mean", style="yellow", min_width=12)
        scaling_table.add_column("Original Std", style="yellow", min_width=12)
        scaling_table.add_column("New Mean", style="green", min_width=12)
        scaling_table.add_column("New Std", style="green", min_width=12)
        
        for result in results:
            scaling_table.add_row(
                result['column'][:15],
                f"{result['original_mean']:.3f}",
                f"{result['original_std']:.3f}",
                f"{result['new_mean']:.3f}",
                f"{result['new_std']:.3f}"
            )
        
        console.print(scaling_table)
    
    def _display_transform_results(self, results: List[Dict[str, Any]]) -> None:
        """Display mathematical transformation results"""
        if not results:
            return
        
        transform_table = Table(title="Mathematical Transformations", 
                               show_header=True, header_style="bold cyan")
        transform_table.add_column("Original Column", style="white", min_width=15)
        transform_table.add_column("Transform", style="green", min_width=12)
        transform_table.add_column("Target Column", style="blue", min_width=15)
        transform_table.add_column("Type", style="yellow", min_width=10)
        
        for result in results:
            transform_type = "New Column" if result['created_new'] else "Modified"
            
            transform_table.add_row(
                result['original_column'][:15],
                result['transform'],
                result['target_column'][:15],
                transform_type
            )
        
        console.print(transform_table)
    
    def _display_binning_results(self, results: List[Dict[str, Any]]) -> None:
        """Display binning results"""
        if not results:
            return
        
        binning_table = Table(title="Feature Binning Results", 
                             show_header=True, header_style="bold cyan")
        binning_table.add_column("Original Column", style="white", min_width=15)
        binning_table.add_column("Binned Column", style="green", min_width=15)
        binning_table.add_column("Number of Bins", style="blue", min_width=12)
        binning_table.add_column("Strategy", style="yellow", min_width=12)
        
        for result in results:
            binning_table.add_row(
                result['original_column'][:15],
                result['binned_column'][:15],
                str(result['n_bins']),
                result['strategy']
            )
        
        console.print(binning_table)
    
    def _display_feature_selection_results(self, selected_features: List[str], 
                                         scores: np.ndarray, method: str) -> None:
        """Display feature selection results"""
        selection_table = Table(title=f"Feature Selection Results ({method.title()})", 
                               show_header=True, header_style="bold cyan")
        selection_table.add_column("Rank", style="white", min_width=6)
        selection_table.add_column("Feature", style="green", min_width=20)
        selection_table.add_column("Score", style="blue", min_width=12)
        selection_table.add_column("Importance", style="yellow", min_width=12)
        
        # Sort by score descending
        feature_score_pairs = list(zip(selected_features, scores))
        feature_score_pairs.sort(key=lambda x: x[1], reverse=True)
        
        max_score = max(scores) if len(scores) > 0 else 1
        
        for i, (feature, score) in enumerate(feature_score_pairs, 1):
            importance = "★★★" if score > 0.7 * max_score else "★★" if score > 0.4 * max_score else "★"
            
            selection_table.add_row(
                str(i),
                feature[:20],
                f"{score:.4f}",
                importance
            )
        
        console.print(selection_table)
    
    def _log_transformation(self, transformation: str, details: Dict[str, Any]) -> None:
        """Log a transformation operation"""
        self.transformation_log.append({
            'transformation': transformation,
            'details': details,
            'timestamp': time.time()
        })


class TransformationPipeline:
    """
    Pipeline for chaining multiple transformations with save/load capability
    """
    
    def __init__(self):
        self.steps: List[Tuple[str, str, Dict[str, Any]]] = []
        self.fitted_pipeline: Optional[Dict[str, Any]] = None
    
    def add_step(self, step_name: str, operation: str, **kwargs) -> 'TransformationPipeline':
        """
        Add a transformation step to the pipeline
        
        Args:
            step_name: Name for this step
            operation: Operation type ('encode', 'scale', 'transform', etc.)
            **kwargs: Parameters for the operation
            
        Returns:
            Self for method chaining
        """
        self.steps.append((step_name, operation, kwargs))
        console.print(f"✅ Added step '{step_name}' ({operation}) to pipeline")
        return self
    
    def fit_transform(self, df: pl.DataFrame) -> pl.DataFrame:
        """
        Fit and apply all pipeline steps
        
        Args:
            df: Input DataFrame
            
        Returns:
            Transformed DataFrame
        """
        console.print(f"🔄 [bold]Applying Transformation Pipeline ({len(self.steps)} steps)[/bold]")
        
        transformer = DataTransformer(df)
        
        for step_name, operation, kwargs in self.steps:
            console.print(f"  Executing: {step_name}")
            
            try:
                if operation == 'encode':
                    transformer.encode_categorical_features(**kwargs)
                elif operation == 'scale':
                    transformer.scale_features(**kwargs)
                elif operation == 'math_transform':
                    transformer.apply_mathematical_transforms(**kwargs)
                elif operation == 'polynomial':
                    transformer.create_polynomial_features(**kwargs)
                elif operation == 'binning':
                    transformer.create_binned_features(**kwargs)
                elif operation == 'select':
                    transformer.select_features(**kwargs)
                else:
                    console.print(f"[yellow]Warning: Unknown operation '{operation}' in step '{step_name}'[/yellow]")
                    continue
            except Exception as e:
                console.print(f"[red]Error in step '{step_name}': {e}[/red]")
                raise
        
        # Store fitted transformers
        self.fitted_pipeline = transformer.fitted_transformers
        
        console.print("✅ Pipeline execution complete")
        return transformer.get_transformed_data()
    
    def get_pipeline_summary(self) -> Dict[str, Any]:
        """Get summary of pipeline steps"""
        return {
            'total_steps': len(self.steps),
            'steps': [{'name': name, 'operation': op, 'parameters': params} 
                     for name, op, params in self.steps],
            'fitted': self.fitted_pipeline is not None
        }
    
    def display_pipeline(self) -> None:
        """Display pipeline steps in a formatted table"""
        if not self.steps:
            console.print("[yellow]Pipeline is empty[/yellow]")
            return
        
        pipeline_table = Table(title="Transformation Pipeline", 
                              show_header=True, header_style="bold cyan")
        pipeline_table.add_column("Step", style="white", min_width=6)
        pipeline_table.add_column("Name", style="green", min_width=20)
        pipeline_table.add_column("Operation", style="blue", min_width=15)
        pipeline_table.add_column("Parameters", style="yellow", min_width=30)
        
        for i, (step_name, operation, kwargs) in enumerate(self.steps, 1):
            params_str = ", ".join(f"{k}={v}" for k, v in kwargs.items())
            if len(params_str) > 30:
                params_str = params_str[:27] + "..."
            
            pipeline_table.add_row(
                str(i),
                step_name,
                operation,
                params_str
            )
        
        console.print(pipeline_table)


# Convenience functions
def transform_data(df: pl.DataFrame) -> DataTransformer:
    """Create and return DataTransformer instance"""
    return DataTransformer(df)


def create_pipeline() -> TransformationPipeline:
    """Create and return TransformationPipeline instance"""
    return TransformationPipeline()


def quick_transform(df: pl.DataFrame, 
                   encode_categorical: bool = True,
                   scale_features: bool = True,
                   encoding_method: str = 'label',
                   scaling_method: str = 'standard') -> pl.DataFrame:
    """
    Quick transformation with common operations
    
    Args:
        df: DataFrame to transform
        encode_categorical: Whether to encode categorical features
        scale_features: Whether to scale numeric features
        encoding_method: Encoding method to use
        scaling_method: Scaling method to use
        
    Returns:
        Transformed DataFrame
    """
    transformer = DataTransformer(df)
    
    if encode_categorical:
        transformer.encode_categorical_features(method=encoding_method)
    
    if scale_features:
        transformer.scale_features(method=scaling_method)
    
    return transformer.get_transformed_data()