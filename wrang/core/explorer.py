#!/usr/bin/env python3
"""
RIDE Core Data Explorer
Statistical exploration and EDA with advanced analytics
"""

import polars as pl
import numpy as np
from typing import Dict, Any, List, Optional, Tuple, Union
from scipy import stats
from scipy.stats import pearsonr, spearmanr, chi2_contingency
import plotext as plt
from rich.console import Console
from rich.table import Table
from rich.panel import Panel
from rich.columns import Columns
from rich.progress import track
from rich.text import Text

from wrang.config import get_config
from wrang.utils.exceptions import DataValidationError, handle_polars_error

console = Console()


class DataExplorer:
    """
    Advanced statistical exploration and EDA
    
    Provides correlation analysis, distribution analysis, outlier detection,
    and statistical tests with beautiful visualizations.
    """
    
    def __init__(self, df: pl.DataFrame):
        """
        Initialize explorer with DataFrame
        
        Args:
            df: Polars DataFrame to explore
        """
        if df is None or len(df) == 0:
            raise DataValidationError("Cannot explore empty or None DataFrame")
        
        self.df = df
        self.config = get_config()
        self._numeric_columns: Optional[List[str]] = None
        self._categorical_columns: Optional[List[str]] = None
        
    def analyze_correlations(self, method: str = 'pearson', 
                           min_correlation: float = 0.1) -> Dict[str, Any]:
        """
        Comprehensive correlation analysis
        
        Args:
            method: Correlation method ('pearson', 'spearman', 'kendall')
            min_correlation: Minimum correlation to display
            
        Returns:
            Dictionary with correlation results
        """
        numeric_cols = self._get_numeric_columns()
        
        if len(numeric_cols) < 2:
            console.print("[yellow]⚠️ Need at least 2 numeric columns for correlation analysis[/yellow]")
            return {'correlations': [], 'matrix': None, 'message': 'Insufficient numeric columns'}
        
        try:
            # Compute correlation matrix
            if method == 'pearson':                
                #corr_matrix = self.df.select(numeric_cols).corr()
                # Drop rows with nulls in the numeric columns (to handle missing data)
                cleaned_df = self.df.select(numeric_cols).drop_nulls()
                
                # Now compute the Pearson correlation matrix
                corr_matrix = cleaned_df.corr()
            else:
                # For Spearman/Kendall, convert to pandas temporarily
                df_pandas = self.df.select(numeric_cols).to_pandas()
                corr_matrix = df_pandas.corr(method=method)
                corr_matrix = pl.from_pandas(corr_matrix)
            
            # Extract significant correlations
            correlations = self._extract_correlations(corr_matrix, numeric_cols, min_correlation)
            
            # Display results
            self._display_correlation_results(correlations, method)
            
            return {
                'correlations': correlations,
                'matrix': corr_matrix,
                'method': method,
                'min_threshold': min_correlation
            }
            
        except Exception as e:
            raise handle_polars_error(e, "correlation analysis")
    
    def analyze_distributions(self, columns: Optional[List[str]] = None) -> Dict[str, Any]:
        """
        Analyze distributions of numeric columns
        
        Args:
            columns: Specific columns to analyze (default: all numeric)
            
        Returns:
            Dictionary with distribution analysis results
        """
        if columns is None:
            columns = self._get_numeric_columns()
        
        if not columns:
            console.print("[yellow]⚠️ No numeric columns found for distribution analysis[/yellow]")
            return {'distributions': {}, 'message': 'No numeric columns'}
        
        distributions = {}
        
        console.print("📊 [bold]Distribution Analysis[/bold]")
        
        for col in track(columns, description="Analyzing distributions..."):
            try:
                dist_analysis = self._analyze_single_distribution(col)
                distributions[col] = dist_analysis
                
                # Display individual results
                self._display_distribution_summary(col, dist_analysis)
                
            except Exception as e:
                console.print(f"[yellow]Warning: Could not analyze distribution for {col}: {e}[/yellow]")
                distributions[col] = {'error': str(e)}
        
        return {'distributions': distributions}
    
    def detect_outliers(self, method: str = 'iqr', columns: Optional[List[str]] = None) -> Dict[str, Any]:
        """
        Detect outliers using various methods
        
        Args:
            method: Detection method ('iqr', 'zscore', 'modified_zscore')
            columns: Specific columns to analyze (default: all numeric)
            
        Returns:
            Dictionary with outlier detection results
        """
        if columns is None:
            columns = self._get_numeric_columns()
        
        if not columns:
            console.print("[yellow]⚠️ No numeric columns found for outlier detection[/yellow]")
            return {'outliers': {}, 'message': 'No numeric columns'}
        
        outliers_results = {}
        
        console.print(f"🎯 [bold]Outlier Detection ({method.upper()})[/bold]")
        
        # Create summary table
        outliers_table = Table(title="Outlier Summary", show_header=True, header_style="bold cyan")
        outliers_table.add_column("Column", style="white", min_width=15)
        outliers_table.add_column("Total Values", style="blue", min_width=12)
        outliers_table.add_column("Outliers", style="red", min_width=10)
        outliers_table.add_column("Outlier %", style="red", min_width=10)
        outliers_table.add_column("Method", style="green", min_width=12)
        
        for col in track(columns, description="Detecting outliers..."):
            try:
                outlier_info = self._detect_column_outliers(col, method)
                outliers_results[col] = outlier_info
                
                # Add to summary table
                outlier_pct = (outlier_info['outlier_count'] / outlier_info['total_count']) * 100
                outliers_table.add_row(
                    col[:15],
                    str(outlier_info['total_count']),
                    str(outlier_info['outlier_count']),
                    f"{outlier_pct:.1f}%",
                    method.upper()
                )
                
            except Exception as e:
                console.print(f"[yellow]Warning: Could not detect outliers for {col}: {e}[/yellow]")
                outliers_results[col] = {'error': str(e)}
        
        console.print(outliers_table)
        return {'outliers': outliers_results, 'method': method}
    
    def analyze_categorical_variables(self, max_categories: Optional[int] = None) -> Dict[str, Any]:
        """
        Analyze categorical variables
        
        Args:
            max_categories: Maximum categories to display per variable
            
        Returns:
            Dictionary with categorical analysis results
        """
        categorical_cols = self._get_categorical_columns()
        
        if not categorical_cols:
            console.print("[yellow]⚠️ No categorical columns found[/yellow]")
            return {'categorical_analysis': {}, 'message': 'No categorical columns'}
        
        if max_categories is None:
            max_categories = self.config.max_categories_plot
        
        analysis_results = {}
        
        console.print("📋 [bold]Categorical Variables Analysis[/bold]")
        
        for col in track(categorical_cols, description="Analyzing categorical variables..."):
            try:
                cat_analysis = self._analyze_single_categorical(col, max_categories)
                analysis_results[col] = cat_analysis
                
                # Display results
                self._display_categorical_summary(col, cat_analysis)
                
            except Exception as e:
                console.print(f"[yellow]Warning: Could not analyze categorical variable {col}: {e}[/yellow]")
                analysis_results[col] = {'error': str(e)}
        
        return {'categorical_analysis': analysis_results}
    
    def plot_correlation_heatmap(self, method: str = 'pearson') -> None:
        """
        Display correlation heatmap in terminal
        
        Args:
            method: Correlation method
        """
        numeric_cols = self._get_numeric_columns()
        
        if len(numeric_cols) < 2:
            console.print("[yellow]⚠️ Need at least 2 numeric columns for heatmap[/yellow]")
            return
        
        try:
            # Compute correlation matrix
            if method == 'pearson':
                corr_matrix = self.df.select(numeric_cols).corr()
            else:
                df_pandas = self.df.select(numeric_cols).to_pandas()
                corr_matrix = df_pandas.corr(method=method)
            
            # Display as formatted table with color coding
            self._display_correlation_heatmap(corr_matrix, numeric_cols)
            
        except Exception as e:
            console.print(f"[red]Error creating heatmap: {e}[/red]")
    
    def plot_histogram(self, column: str, bins: int = 20) -> None:
        """
        Plot histogram for a numeric column
        
        Args:
            column: Column name to plot
            bins: Number of histogram bins
        """
        if column not in self.df.columns:
            console.print(f"[red]❌ Column '{column}' not found[/red]")
            return
        
        if not self._is_numeric_column(column):
            console.print(f"[yellow]⚠️ Column '{column}' is not numeric[/yellow]")
            return
        
        try:
            # Get data and remove nulls
            data = self.df[column].drop_nulls()
            
            if len(data) == 0:
                console.print(f"[yellow]⚠️ No non-null data in column '{column}'[/yellow]")
                return
            
            # Create histogram
            plt.clear_data()
            plt.theme('dark')
            plt.plotsize(self.config.plot_width, self.config.plot_height)
            
            values = data.to_list()
            plt.hist(values, bins=bins, color='cyan')
            plt.title(f"Distribution of {column}")
            plt.xlabel(column)
            plt.ylabel("Frequency")
            
            console.print(f"\n📊 [bold]Histogram: {column}[/bold]")
            plt.show()
            plt.clear_data()
            
            # Show basic statistics
            self._show_column_stats(column, data)
            
        except Exception as e:
            console.print(f"[red]Error plotting histogram: {e}[/red]")
    
    def plot_scatter(self, x_column: str, y_column: str, sample_size: Optional[int] = None) -> None:
        """
        Plot scatter plot for two numeric columns
        
        Args:
            x_column: X-axis column
            y_column: Y-axis column
            sample_size: Maximum number of points to plot
        """
        if x_column not in self.df.columns:
            console.print(f"[red]❌ Column '{x_column}' not found[/red]")
            return
        
        if y_column not in self.df.columns:
            console.print(f"[red]❌ Column '{y_column}' not found[/red]")
            return
        
        if not (self._is_numeric_column(x_column) and self._is_numeric_column(y_column)):
            console.print("[yellow]⚠️ Both columns must be numeric for scatter plot[/yellow]")
            return
        
        try:
            # Get data and remove nulls
            scatter_df = self.df.select([x_column, y_column]).drop_nulls()
            
            if len(scatter_df) == 0:
                console.print("[yellow]⚠️ No non-null data pairs found[/yellow]")
                return
            
            # Sample if dataset is too large
            if sample_size and len(scatter_df) > sample_size:
                scatter_df = scatter_df.sample(n=sample_size)
                console.print(f"[blue]ℹ️ Showing random sample of {sample_size:,} points[/blue]")
            
            # Create scatter plot
            plt.clear_data()
            plt.theme('dark')
            plt.plotsize(self.config.plot_width, self.config.plot_height)
            
            x_values = scatter_df[x_column].to_list()
            y_values = scatter_df[y_column].to_list()
            
            plt.scatter(x_values, y_values, color='green', marker='•')
            plt.title(f"Scatter Plot: {x_column} vs {y_column}")
            plt.xlabel(x_column)
            plt.ylabel(y_column)
            
            console.print(f"\n📊 [bold]Scatter Plot: {x_column} vs {y_column}[/bold]")
            plt.show()
            plt.clear_data()
            
            # Calculate and show correlation
            try:
                correlation = self.df.select([
                    pl.corr(x_column, y_column).alias('correlation')
                ]).item()
                
                console.print(f"\n📈 [bold]Correlation:[/bold] {correlation:.4f}")
                self._interpret_correlation(correlation)
                
            except Exception:
                console.print("[yellow]Could not calculate correlation[/yellow]")
            
        except Exception as e:
            console.print(f"[red]Error plotting scatter: {e}[/red]")
    
    def test_normality(self, columns: Optional[List[str]] = None, alpha: float = 0.05) -> Dict[str, Any]:
        """
        Test normality of distributions using Shapiro-Wilk test
        
        Args:
            columns: Columns to test (default: all numeric)
            alpha: Significance level
            
        Returns:
            Dictionary with normality test results
        """
        if columns is None:
            columns = self._get_numeric_columns()
        
        if not columns:
            console.print("[yellow]⚠️ No numeric columns found for normality testing[/yellow]")
            return {'normality_tests': {}, 'message': 'No numeric columns'}
        
        results = {}
        
        # Create results table
        normality_table = Table(title="Normality Tests (Shapiro-Wilk)", show_header=True, header_style="bold cyan")
        normality_table.add_column("Column", style="white", min_width=15)
        normality_table.add_column("Statistic", style="blue", min_width=12)
        normality_table.add_column("P-value", style="green", min_width=12)
        normality_table.add_column("Normal?", style="magenta", min_width=10)
        normality_table.add_column("Interpretation", style="yellow", min_width=20)
        
        for col in track(columns, description="Testing normality..."):
            try:
                # Get non-null data
                data = self.df[col].drop_nulls().to_numpy()
                
                if len(data) < 3:
                    results[col] = {'error': 'Insufficient data'}
                    continue
                
                # Shapiro-Wilk test (sample if too large)
                if len(data) > 5000:
                    sample_data = np.random.choice(data, 5000, replace=False)
                    console.print(f"[blue]ℹ️ Sampling 5000 points for {col}[/blue]")
                else:
                    sample_data = data
                
                statistic, p_value = stats.shapiro(sample_data)
                is_normal = p_value > alpha
                
                results[col] = {
                    'statistic': statistic,
                    'p_value': p_value,
                    'is_normal': is_normal,
                    'alpha': alpha
                }
                
                # Interpret result
                if is_normal:
                    interpretation = "Likely normal"
                    status_color = "green"
                    status_text = "Yes"
                else:
                    interpretation = "Not normal"
                    status_color = "red"  
                    status_text = "No"
                
                normality_table.add_row(
                    col[:15],
                    f"{statistic:.4f}",
                    f"{p_value:.4f}",
                    Text(status_text, style=status_color),
                    interpretation
                )
                
            except Exception as e:
                console.print(f"[yellow]Warning: Could not test normality for {col}: {e}[/yellow]")
                results[col] = {'error': str(e)}
        
        console.print(normality_table)
        return {'normality_tests': results, 'alpha': alpha}
    
    def _get_numeric_columns(self) -> List[str]:
        """Get list of numeric columns"""
        if self._numeric_columns is None:
            self._numeric_columns = [col for col in self.df.columns if self._is_numeric_column(col)]
        return self._numeric_columns
    
    def _get_categorical_columns(self) -> List[str]:
        """Get list of categorical columns"""
        if self._categorical_columns is None:
            self._categorical_columns = [col for col in self.df.columns if self._is_categorical_column(col)]
        return self._categorical_columns
    
    def _is_numeric_column(self, col: str) -> bool:
        """Check if column is numeric"""
        dtype = str(self.df[col].dtype)
        return any(num_type in dtype.lower() for num_type in ['int', 'float', 'decimal'])
    
    def _is_categorical_column(self, col: str) -> bool:
        """Check if column is categorical"""
        dtype = str(self.df[col].dtype)
        return 'utf8' in dtype.lower() or 'str' in dtype.lower()
    
    def _extract_correlations(self, corr_matrix: pl.DataFrame, columns: List[str], 
                            min_correlation: float) -> List[Dict[str, Any]]:
        """Extract significant correlations from matrix"""
        correlations = []
        
        for i, col1 in enumerate(columns):
            for j, col2 in enumerate(columns[i+1:], i+1):
                try:
                    corr_value = corr_matrix[col1][j]  # Get correlation value
                    
                    if abs(corr_value) >= min_correlation:
                        correlations.append({
                            'column1': col1,
                            'column2': col2,
                            'correlation': corr_value,
                            'abs_correlation': abs(corr_value),
                            'strength': self._classify_correlation_strength(abs(corr_value))
                        })
                except Exception:
                    continue
        
        # Sort by absolute correlation
        correlations.sort(key=lambda x: x['abs_correlation'], reverse=True)
        return correlations
    
    def _classify_correlation_strength(self, abs_corr: float) -> str:
        """Classify correlation strength"""
        if abs_corr >= 0.9:
            return "Very Strong"
        elif abs_corr >= 0.7:
            return "Strong"
        elif abs_corr >= 0.5:
            return "Moderate"
        elif abs_corr >= 0.3:
            return "Weak"
        else:
            return "Very Weak"
    
    def _display_correlation_results(self, correlations: List[Dict[str, Any]], method: str) -> None:
        """Display correlation analysis results"""
        if not correlations:
            console.print("[yellow]⚠️ No significant correlations found[/yellow]")
            return
        
        # Create correlation table
        corr_table = Table(title=f"Correlation Analysis ({method.title()})", show_header=True, header_style="bold cyan")
        corr_table.add_column("Column 1", style="white", min_width=15)
        corr_table.add_column("Column 2", style="white", min_width=15)
        corr_table.add_column("Correlation", style="blue", min_width=12)
        corr_table.add_column("Strength", style="green", min_width=12)
        corr_table.add_column("Direction", style="yellow", min_width=10)
        
        for corr in correlations[:20]:  # Show top 20
            # Color code correlation value
            corr_val = corr['correlation']
            if corr_val > 0:
                corr_color = "green"
                direction = "Positive"
            else:
                corr_color = "red"
                direction = "Negative"
            
            corr_table.add_row(
                corr['column1'][:15],
                corr['column2'][:15],
                Text(f"{corr_val:.4f}", style=corr_color),
                corr['strength'],
                direction
            )
        
        console.print(corr_table)
    
    def _analyze_single_distribution(self, column: str) -> Dict[str, Any]:
        """Analyze distribution of a single column"""
        data = self.df[column].drop_nulls()
        
        if len(data) == 0:
            return {'error': 'No non-null data'}
        
        data_array = data.to_numpy()
        
        return {
            'count': len(data),
            'mean': float(np.mean(data_array)),
            'median': float(np.median(data_array)),
            'std': float(np.std(data_array)),
            'skewness': float(stats.skew(data_array)),
            'kurtosis': float(stats.kurtosis(data_array)),
            'min': float(np.min(data_array)),
            'max': float(np.max(data_array)),
            'range': float(np.max(data_array) - np.min(data_array)),
            'q25': float(np.percentile(data_array, 25)),
            'q75': float(np.percentile(data_array, 75)),
            'iqr': float(np.percentile(data_array, 75) - np.percentile(data_array, 25))
        }
    
    def _display_distribution_summary(self, column: str, analysis: Dict[str, Any]) -> None:
        """Display distribution analysis summary"""
        if 'error' in analysis:
            console.print(f"[red]❌ {column}: {analysis['error']}[/red]")
            return
        
        # Create summary text
        summary = f"""[bold]{column}[/bold]
• Count: {analysis['count']:,}
• Mean: {analysis['mean']:.3f}, Median: {analysis['median']:.3f}
• Std: {analysis['std']:.3f}, Range: {analysis['range']:.3f}
• Skewness: {analysis['skewness']:.3f}, Kurtosis: {analysis['kurtosis']:.3f}"""
        
        # Interpret skewness and kurtosis
        if abs(analysis['skewness']) > 1:
            skew_interpretation = "highly skewed"
        elif abs(analysis['skewness']) > 0.5:
            skew_interpretation = "moderately skewed"
        else:
            skew_interpretation = "approximately symmetric"
        
        summary += f"\n• Distribution: {skew_interpretation}"
        
        panel = Panel(summary, border_style="blue", padding=(0, 1))
        console.print(panel)
    
    def _detect_column_outliers(self, column: str, method: str) -> Dict[str, Any]:
        """Detect outliers in a single column"""
        data = self.df[column].drop_nulls()
        data_array = data.to_numpy()
        
        if len(data_array) == 0:
            return {'error': 'No non-null data', 'outlier_count': 0, 'total_count': 0}
        
        if method == 'iqr':
            q1, q3 = np.percentile(data_array, [25, 75])
            iqr = q3 - q1
            lower_bound = q1 - 1.5 * iqr
            upper_bound = q3 + 1.5 * iqr
            outliers = (data_array < lower_bound) | (data_array > upper_bound)
            
        elif method == 'zscore':
            z_scores = np.abs(stats.zscore(data_array))
            outliers = z_scores > 3
            
        elif method == 'modified_zscore':
            median = np.median(data_array)
            mad = stats.median_abs_deviation(data_array)
            modified_z_scores = 0.6745 * (data_array - median) / mad
            outliers = np.abs(modified_z_scores) > 3.5
            
        else:
            raise ValueError(f"Unknown outlier detection method: {method}")
        
        return {
            'outlier_count': int(np.sum(outliers)),
            'total_count': len(data_array),
            'outlier_indices': np.where(outliers)[0].tolist(),
            'method': method,
            'bounds': locals().get('lower_bound', None) and locals().get('upper_bound', None)
        }
    
    def _analyze_single_categorical(self, column: str, max_categories: int) -> Dict[str, Any]:
        """Analyze a single categorical column"""
        try:
            value_counts = self.df[column].value_counts().head(max_categories)
            total_count = len(self.df[column].drop_nulls())
            
            return {
                'unique_count': self.df[column].n_unique(),
                'total_count': total_count,
                'top_categories': value_counts.to_dict() if hasattr(value_counts, 'to_dict') else {},
                'missing_count': self.df[column].null_count()
            }
        except Exception as e:
            return {'error': str(e)}
    
    def _display_categorical_summary(self, column: str, analysis: Dict[str, Any]) -> None:
        """Display categorical analysis summary"""
        if 'error' in analysis:
            console.print(f"[red]❌ {column}: {analysis['error']}[/red]")
            return
        
        summary = f"""[bold]{column}[/bold]
• Unique values: {analysis['unique_count']:,}
• Total count: {analysis['total_count']:,}
• Missing: {analysis['missing_count']:,}"""
        
        # Show top categories if available
        if 'top_categories' in analysis and analysis['top_categories']:
            summary += "\n• Top categories:"
            for cat, count in list(analysis['top_categories'].items())[:5]:
                pct = (count / analysis['total_count']) * 100
                summary += f"\n  - {cat}: {count:,} ({pct:.1f}%)"
        
        panel = Panel(summary, border_style="green", padding=(0, 1))
        console.print(panel)
    
    def _display_correlation_heatmap(self, corr_matrix: Union[pl.DataFrame, any], columns: List[str]) -> None:
        """Display correlation matrix as formatted table"""
        console.print("\n🔥 [bold]Correlation Heatmap[/bold]")
        
        # Create heatmap table
        heatmap_table = Table(show_header=True, header_style="bold cyan")
        heatmap_table.add_column("Column", style="white", min_width=12)
        
        # Add column headers
        for col in columns[:10]:  # Limit to 10 columns for readability
            heatmap_table.add_column(col[:8], min_width=8)
        
        # Add rows
        for i, row_col in enumerate(columns[:10]):
            row_data = [row_col[:12]]
            
            for j, col_col in enumerate(columns[:10]):
                try:
                    if hasattr(corr_matrix, 'item'):  # Polars DataFrame
                        corr_val = corr_matrix[i, j]
                    else:  # Pandas DataFrame
                        corr_val = corr_matrix.iloc[i, j]
                    
                    # Color code correlation
                    if abs(corr_val) >= 0.7:
                        color = "red" if corr_val > 0 else "blue"
                        style = "bold"
                    elif abs(corr_val) >= 0.3:
                        color = "yellow" if corr_val > 0 else "cyan"
                        style = None
                    else:
                        color = "white"
                        style = "dim"
                    
                    if i == j:
                        cell_text = Text("1.00", style="white bold")
                    else:
                        cell_text = Text(f"{corr_val:.2f}", style=f"{color} {style or ''}".strip())
                    
                    row_data.append(cell_text)
                    
                except Exception:
                    row_data.append("N/A")
            
            heatmap_table.add_row(*row_data)
        
        console.print(heatmap_table)
    
    def _show_column_stats(self, column: str, data: pl.Series) -> None:
        """Show basic statistics for a column"""
        data_array = data.to_numpy()
        
        stats_text = f"""📈 [bold]Statistics for {column}:[/bold]
• Count: {len(data_array):,}
• Mean: {np.mean(data_array):.3f}
• Median: {np.median(data_array):.3f}
• Std Dev: {np.std(data_array):.3f}
• Min: {np.min(data_array):.3f}
• Max: {np.max(data_array):.3f}"""
        
        console.print(stats_text)
    
    def _interpret_correlation(self, correlation: float) -> None:
        """Interpret correlation value"""
        abs_corr = abs(correlation)
        
        if abs_corr >= 0.9:
            strength = "very strong"
        elif abs_corr >= 0.7:
            strength = "strong"
        elif abs_corr >= 0.5:
            strength = "moderate"
        elif abs_corr >= 0.3:
            strength = "weak"
        else:
            strength = "very weak"
        
        direction = "positive" if correlation > 0 else "negative"
        
        console.print(f"💡 [bold]Interpretation:[/bold] {strength.title()} {direction} correlation")


# Convenience function
def explore_data(df: pl.DataFrame) -> DataExplorer:
    """Create and return DataExplorer instance"""
    return DataExplorer(df)