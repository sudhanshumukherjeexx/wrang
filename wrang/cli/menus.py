#!/usr/bin/env python3
"""
wrang Menu System
Interactive menu handlers that connect the UI with powerful core modules
"""

import polars as pl
import time
from pathlib import Path
from typing import Dict, Any, List, Optional, Tuple
from rich.prompt import Prompt, IntPrompt, FloatPrompt
from rich.console import Console
from rich.panel import Panel

from wrang.config import get_config, update_config, ImputationStrategy, ScalingMethod, EncodingMethod
from wrang.core.loader import FastDataLoader, DataSaver
from wrang.core.inspector import DataInspector
from wrang.core.explorer import DataExplorer
from wrang.core.cleaner import DataCleaner, quick_clean
from wrang.core.transformer import DataTransformer, create_pipeline, quick_transform
from wrang.core.validator import DataSchema, ColumnSchema, DataValidator, infer_schema
from wrang.cli.formatters import get_formatter, wait_for_input, clear_screen
from wrang.utils.exceptions import RideError, create_user_friendly_message

console = Console()


class MenuHandler:
    """
    Main menu handler for wrang
    
    Manages the interactive menu system and coordinates between
    the UI layer and core data processing modules.
    """
    
    def __init__(self):
        self.config = get_config()
        self.formatter = get_formatter()
        self.current_df: Optional[pl.DataFrame] = None
        self.current_file: Optional[str] = None
        self.loader = FastDataLoader()
        self.saver = DataSaver()
        self.operation_history: List[Dict[str, Any]] = []
    
    def run_main_menu(self) -> None:
        """Run the main menu loop"""
        while True:
            try:
                clear_screen()
                self.formatter.display_welcome_banner()
                self.formatter.display_main_menu(self.current_file)
                
                choice = Prompt.ask(
                    f"\n[{self.formatter.colors['primary']}]Enter your choice[/{self.formatter.colors['primary']}]",
                    console=console
                ).strip().lower()
                
                if choice in ['q', 'quit', 'exit']:
                    self._handle_exit()
                    break
                elif choice == '1':
                    self.handle_load_dataset()
                elif choice == '2':
                    self.handle_inspect_data()
                elif choice == '3':
                    self.handle_explore_data()
                elif choice == '4':
                    self.handle_clean_data()
                elif choice == '5':
                    self.handle_transform_data()
                elif choice == '6':
                    self.handle_visualize_data()
                elif choice == '7':
                    self.handle_export_data()
                elif choice == '8':
                    self.handle_settings()
                elif choice == '9':
                    self.handle_sql_query()
                elif choice == '10':
                    self.handle_html_profile()
                elif choice == '11':
                    self.handle_validate_data()
                elif choice == '$':
                    self.handle_quick_export()
                else:
                    console.print(f"[{self.formatter.colors['error']}]Invalid choice. Please try again.[/{self.formatter.colors['error']}]")
                    wait_for_input()
                
            except KeyboardInterrupt:
                console.print(f"\n[{self.formatter.colors['warning']}]Operation cancelled by user[/{self.formatter.colors['warning']}]")
                wait_for_input()
            except Exception as e:
                self._handle_unexpected_error(e)
    
    def handle_load_dataset(self) -> None:
        """Handle dataset loading menu"""
        self.formatter.display_section_header(
            "Load Dataset",
            "Import your data from various file formats",
            "📂"
        )
        
        if not self._check_no_dataset_warning():
            return
        
        file_path = self.formatter.prompt_file_path("Enter the path to your dataset file")
        
        if not file_path.strip():
            console.print(f"[{self.formatter.colors['warning']}]No file path provided[/{self.formatter.colors['warning']}]")
            wait_for_input()
            return
        
        file_path = Path(file_path.strip())
        
        try:
            start_time = time.time()
            
            # Load the dataset
            self.current_df = self.loader.load(file_path)
            self.current_file = file_path.name
            
            duration = time.time() - start_time
            
            # Display success and summary
            self.formatter.display_operation_result(
                "Dataset Loading",
                True,
                f"Loaded {len(self.current_df):,} rows × {len(self.current_df.columns)} columns",
                duration
            )
            
            # Show data summary
            console.print()
            self.formatter.display_data_summary(self.current_df, self.current_file)
            
            # Log operation
            self._log_operation("load_dataset", {
                'file_path': str(file_path),
                'rows': len(self.current_df),
                'columns': len(self.current_df.columns),
                'duration': duration
            })
            
        except Exception as e:
            self._handle_operation_error("Dataset Loading", e)
        
        wait_for_input()
    
    def handle_inspect_data(self) -> None:
        """Handle data inspection menu"""
        if not self._check_dataset_loaded():
            return
        
        self.formatter.display_section_header(
            "Data Inspection",
            "Explore dataset structure, quality, and characteristics",
            "🔍"
        )
        
        try:
            inspector = DataInspector(self.current_df)
            
            while True:
                # Create inspection menu
                choices = [
                    "Dataset Overview",
                    "Column Details", 
                    "Statistical Summary",
                    "Data Quality Report",
                    "Memory Usage Analysis",
                    "Detect Data Issues",
                    "Change Column Data Type",  # NEW OPTION
                    "Back to Main Menu"
                ]
                
                choice = self.formatter.prompt_user_choice(
                    "What would you like to inspect?",
                    choices
                )
                
                if not choice or choice == "Back to Main Menu":
                    break
                
                console.print()
                start_time = time.time()
                
                if choice == "Dataset Overview":
                    inspector.display_overview()
                
                elif choice == "Column Details":
                    detailed = self.formatter.prompt_confirmation(
                        "Show detailed column analysis?", 
                        default=False
                    )
                    self.formatter.display_column_info(self.current_df, detailed=detailed)
                
                elif choice == "Statistical Summary":
                    inspector.display_statistical_summary()
                
                elif choice == "Data Quality Report":
                    inspector.display_data_quality()
                
                elif choice == "Memory Usage Analysis":
                    memory_info = inspector.get_memory_usage()
                    self._display_memory_analysis(memory_info)
                
                elif choice == "Detect Data Issues":
                    issues = inspector.detect_potential_issues()
                    self._display_data_issues(issues)
                
                elif choice == "Change Column Data Type":
                    self._handle_datatype_conversion()
                
                duration = time.time() - start_time
                console.print(f"\n[{self.formatter.colors['dim']}]Analysis completed in {duration:.2f} seconds[/{self.formatter.colors['dim']}]")
                wait_for_input()
        
        except Exception as e:
            self._handle_operation_error("Data Inspection", e)
            wait_for_input()
    
    def _handle_datatype_conversion(self) -> None:
        """Handle data type conversion for individual columns"""
        console.print(f"\n[{self.formatter.colors['primary']}]📝 Column Data Type Conversion[/{self.formatter.colors['primary']}]")
        console.print(f"[{self.formatter.colors['info']}]Convert columns to appropriate data types one at a time[/{self.formatter.colors['info']}]")
        
        # Important note about datetime conversion
        console.print(f"\n[{self.formatter.colors['warning']}]ℹ️  Note: Datetime conversion is not yet supported. Coming in future updates.[/{self.formatter.colors['warning']}]")
        
        # Display current column types
        console.print(f"\n[bold]Current Column Types:[/bold]")
        
        from rich.table import Table
        type_table = Table(show_header=True, header_style="bold cyan")
        type_table.add_column("#", style="cyan", width=4)
        type_table.add_column("Column", style="white", min_width=20)
        type_table.add_column("Current Type", style="green", min_width=15)
        type_table.add_column("Sample Values", style="yellow", min_width=30)
        
        column_map = {}
        for i, col in enumerate(self.current_df.columns, 1):
            dtype = str(self.current_df[col].dtype)
            samples = self.current_df[col].drop_nulls().head(3).to_list()
            sample_str = ", ".join(str(x)[:15] for x in samples)
            if len(sample_str) > 30:
                sample_str = sample_str[:27] + "..."
            
            type_table.add_row(str(i), col[:20], dtype, sample_str)
            column_map[i] = col
        
        console.print(type_table)
        
        # Select column to convert
        console.print(f"\n[{self.formatter.colors['info']}]Select a column to convert (or 0 to cancel):[/{self.formatter.colors['info']}]")
        
        try:
            col_choice = IntPrompt.ask("Column number", console=console)
            
            if col_choice == 0:
                console.print(f"[{self.formatter.colors['warning']}]Conversion cancelled[/{self.formatter.colors['warning']}]")
                return
            
            if col_choice not in column_map:
                console.print(f"[{self.formatter.colors['error']}]Invalid column number[/{self.formatter.colors['error']}]")
                return
            
            selected_col = column_map[col_choice]
            current_dtype = str(self.current_df[selected_col].dtype)
            
            console.print(f"\n[bold]Selected Column:[/bold] {selected_col}")
            console.print(f"[bold]Current Type:[/bold] {current_dtype}")
            
            # Show available conversions
            console.print(f"\n[bold]Available Data Types:[/bold]")
            available_types = [
                ("1", "String (text)", "Utf8", "Convert to text/string format"),
                ("2", "Integer (whole numbers)", "Int64", "Convert to integer numbers"),
                ("3", "Float (decimal numbers)", "Float64", "Convert to decimal numbers"),
                ("4", "Boolean (True/False)", "Boolean", "Convert to true/false values"),
                ("5", "Cancel", None, "Return without changes")
            ]
            
            for num, name, dtype_name, desc in available_types:
                if dtype_name:
                    console.print(f"  {num}. [cyan]{name}[/cyan] ({dtype_name})")
                    console.print(f"     [dim]{desc}[/dim]")
                else:
                    console.print(f"  {num}. [yellow]{name}[/yellow]")
            
            type_choice = Prompt.ask("\nSelect target data type", choices=["1", "2", "3", "4", "5"], console=console)
            
            if type_choice == "5":
                console.print(f"[{self.formatter.colors['warning']}]Conversion cancelled[/{self.formatter.colors['warning']}]")
                return
            
            # Map choice to target type
            type_map = {
                "1": ("Utf8", pl.Utf8),
                "2": ("Int64", pl.Int64),
                "3": ("Float64", pl.Float64),
                "4": ("Boolean", pl.Boolean)
            }
            
            target_name, target_type = type_map[type_choice]
            
            # Perform conversion with error handling
            console.print(f"\n[{self.formatter.colors['info']}]Converting {selected_col} to {target_name}...[/{self.formatter.colors['info']}]")
            
            try:
                # Try conversion
                if target_type == pl.Utf8:
                    # String conversion (always safe)
                    converted_col = self.current_df[selected_col].cast(pl.Utf8)
                
                elif target_type == pl.Int64:
                    # Integer conversion
                    # Check if current type is string
                    if str(self.current_df[selected_col].dtype) == "Utf8":
                        # Try to parse as integer
                        try:
                            converted_col = self.current_df[selected_col].str.strip_chars().cast(pl.Int64, strict=False)
                        except Exception:
                            raise ValueError("Column contains non-numeric text that cannot be converted to integers")
                    else:
                        # Numeric to integer
                        converted_col = self.current_df[selected_col].cast(pl.Int64, strict=False)
                
                elif target_type == pl.Float64:
                    # Float conversion
                    if str(self.current_df[selected_col].dtype) == "Utf8":
                        # Try to parse as float
                        try:
                            converted_col = self.current_df[selected_col].str.strip_chars().cast(pl.Float64, strict=False)
                        except Exception:
                            raise ValueError("Column contains non-numeric text that cannot be converted to decimals")
                    else:
                        converted_col = self.current_df[selected_col].cast(pl.Float64, strict=False)
                
                elif target_type == pl.Boolean:
                    # Boolean conversion
                    if str(self.current_df[selected_col].dtype) == "Utf8":
                        # Convert common boolean strings
                        converted_col = self.current_df[selected_col].str.to_lowercase().map_elements(
                            lambda x: True if x in ['true', '1', 'yes', 'y', 't'] 
                            else False if x in ['false', '0', 'no', 'n', 'f'] 
                            else None,
                            return_dtype=pl.Boolean
                        )
                    else:
                        converted_col = self.current_df[selected_col].cast(pl.Boolean, strict=False)
                
                # Check for conversion issues
                null_count_before = self.current_df[selected_col].null_count()
                null_count_after = converted_col.null_count()
                new_nulls = null_count_after - null_count_before
                
                if new_nulls > 0:
                    console.print(f"\n[{self.formatter.colors['warning']}]⚠️  Warning: Conversion created {new_nulls} new null values[/{self.formatter.colors['warning']}]")
                    console.print(f"[{self.formatter.colors['warning']}]This indicates {new_nulls} values couldn't be converted[/{self.formatter.colors['warning']}]")
                    
                    proceed = self.formatter.prompt_confirmation(
                        "Do you want to proceed with this conversion?",
                        default=False
                    )
                    
                    if not proceed:
                        console.print(f"[{self.formatter.colors['warning']}]Conversion cancelled[/{self.formatter.colors['warning']}]")
                        return
                
                # Show preview of conversion
                console.print(f"\n[bold]Conversion Preview:[/bold]")
                preview_table = Table(show_header=True, header_style="bold cyan")
                preview_table.add_column("Before", style="yellow")
                preview_table.add_column("After", style="green")
                preview_table.add_column("Status", style="white")
                
                for i in range(min(5, len(self.current_df))):
                    before_val = self.current_df[selected_col][i]
                    after_val = converted_col[i]
                    
                    if before_val is None and after_val is None:
                        status = "🔵 Null → Null"
                    elif before_val is not None and after_val is None:
                        status = "🔴 Lost in conversion"
                    elif str(before_val) != str(after_val):
                        status = "🟢 Converted"
                    else:
                        status = "🟢 Unchanged"
                    
                    preview_table.add_row(
                        str(before_val) if before_val is not None else "NULL",
                        str(after_val) if after_val is not None else "NULL",
                        status
                    )
                
                console.print(preview_table)
                
                # Confirm conversion
                confirm = self.formatter.prompt_confirmation(
                    f"Apply conversion to column '{selected_col}'?",
                    default=True
                )
                
                if confirm:
                    # Apply conversion
                    self.current_df = self.current_df.with_columns(converted_col.alias(selected_col))
                    
                    console.print(f"\n[{self.formatter.colors['success']}]✅ Successfully converted {selected_col} to {target_name}[/{self.formatter.colors['success']}]")
                    
                    # Log the operation
                    self._log_operation("datatype_conversion", {
                        'column': selected_col,
                        'from_type': current_dtype,
                        'to_type': target_name,
                        'new_nulls': new_nulls
                    })
                else:
                    console.print(f"[{self.formatter.colors['warning']}]Conversion cancelled[/{self.formatter.colors['warning']}]")
            
            except Exception as e:
                # Handle conversion errors
                console.print(f"\n[{self.formatter.colors['error']}]❌ Conversion Failed[/{self.formatter.colors['error']}]")
                console.print(f"[{self.formatter.colors['error']}]Error: {str(e)}[/{self.formatter.colors['error']}]")
                
                # Show helpful suggestions
                console.print(f"\n[{self.formatter.colors['info']}]💡 Suggestions:[/{self.formatter.colors['info']}]")
                
                if "non-numeric" in str(e).lower():
                    console.print(f"  • Column contains text values that cannot be converted to numbers")
                    console.print(f"  • Try cleaning the data first to remove non-numeric characters")
                    console.print(f"  • Check for special characters, currency symbols, or text mixed with numbers")
                elif "overflow" in str(e).lower():
                    console.print(f"  • Some values are too large for the target data type")
                    console.print(f"  • Try using Float64 instead of Int64 for large numbers")
                elif target_type == pl.Boolean:
                    console.print(f"  • Boolean conversion failed - check for unexpected values")
                    console.print(f"  • Accepted values: true/false, 1/0, yes/no, y/n, t/f")
                else:
                    console.print(f"  • Check your data for inconsistencies")
                    console.print(f"  • Try inspecting the column first to identify issues")
                    console.print(f"  • Consider cleaning the data before conversion")
                
                # Show problematic values
                try:
                    console.print(f"\n[{self.formatter.colors['info']}]Sample values in column:[/{self.formatter.colors['info']}]")
                    sample_values = self.current_df[selected_col].drop_nulls().head(10).to_list()
                    for val in sample_values[:5]:
                        console.print(f"  • {val} (type: {type(val).__name__})")
                except Exception:
                    pass
        
        except KeyboardInterrupt:
            console.print(f"\n[{self.formatter.colors['warning']}]Conversion cancelled by user[/{self.formatter.colors['warning']}]")
        except Exception as e:
            console.print(f"[{self.formatter.colors['error']}]Unexpected error: {e}[/{self.formatter.colors['error']}]")
    
    def handle_explore_data(self) -> None:
        """Handle data exploration menu"""
        if not self._check_dataset_loaded():
            return
        
        self.formatter.display_section_header(
            "Data Exploration",
            "Statistical analysis, correlations, and distributions",
            "📊"
        )
        
        try:
            explorer = DataExplorer(self.current_df)
            
            while True:
                choices = [
                    "Correlation Analysis",
                    "Distribution Analysis",
                    "Outlier Detection",
                    "Categorical Variables Analysis",
                    "Normality Testing",
                    "Plot Histogram",
                    "Plot Scatter Plot",
                    "Plot Correlation Heatmap",
                    "Back to Main Menu"
                ]
                
                choice = self.formatter.prompt_user_choice(
                    "Choose exploration method:",
                    choices
                )
                
                if not choice or choice == "Back to Main Menu":
                    break
                
                console.print()
                start_time = time.time()
                
                if choice == "Correlation Analysis":
                    method = self._prompt_correlation_method()
                    min_corr = FloatPrompt.ask(
                        "Minimum correlation to display",
                        default=0.1,
                        console=console
                    )
                    explorer.analyze_correlations(method=method, min_correlation=min_corr)
                
                elif choice == "Distribution Analysis":
                    columns = self._prompt_column_selection("numeric")
                    explorer.analyze_distributions(columns=columns)
                
                elif choice == "Outlier Detection":
                    method = self._prompt_outlier_method()
                    columns = self._prompt_column_selection("numeric")
                    explorer.detect_outliers(method=method, columns=columns)
                
                elif choice == "Categorical Variables Analysis":
                    max_cats = IntPrompt.ask(
                        "Maximum categories to display per variable",
                        default=20,
                        console=console
                    )
                    explorer.analyze_categorical_variables(max_categories=max_cats)
                
                elif choice == "Normality Testing":
                    columns = self._prompt_column_selection("numeric")
                    alpha = FloatPrompt.ask(
                        "Significance level (alpha)",
                        default=0.05,
                        console=console
                    )
                    explorer.test_normality(columns=columns, alpha=alpha)
                
                elif choice == "Plot Histogram":
                    column = self._prompt_single_column_selection("numeric")
                    if column:
                        bins = IntPrompt.ask("Number of bins", default=20, console=console)
                        explorer.plot_histogram(column, bins=bins)
                
                elif choice == "Plot Scatter Plot":
                    columns = self._prompt_two_column_selection("numeric")
                    if columns and len(columns) == 2:
                        sample_size = IntPrompt.ask(
                            "Maximum points to plot (for performance)",
                            default=1000,
                            console=console
                        )
                        explorer.plot_scatter(columns[0], columns[1], sample_size=sample_size)
                
                elif choice == "Plot Correlation Heatmap":
                    method = self._prompt_correlation_method()
                    explorer.plot_correlation_heatmap(method=method)
                
                duration = time.time() - start_time
                console.print(f"\n[{self.formatter.colors['dim']}]Analysis completed in {duration:.2f} seconds[/{self.formatter.colors['dim']}]")
                wait_for_input()
        
        except Exception as e:
            self._handle_operation_error("Data Exploration", e)
            wait_for_input()
    
    def handle_clean_data(self) -> None:
        """Handle data cleaning menu"""
        if not self._check_dataset_loaded():
            return
        
        self.formatter.display_section_header(
            "Data Cleaning",
            "Clean and validate your data for analysis",
            "🧹"
        )
        
        try:
            # Show cleaning strategy options
            strategies = [
                "Custom Cleaning (Step by Step)",
                "Quick Clean - Basic",
                "Quick Clean - Aggressive",
                "Quick Clean - Conservative",
                "Quick Clean - ML Ready",
                "Undo Last Cleaning Step",
                "Undo All Cleaning (Restore Original)",
                "View Cleaning History",
                "Back to Main Menu"
            ]

            choice = self.formatter.prompt_user_choice(
                "Choose cleaning approach:",
                strategies
            )

            if not choice or choice == "Back to Main Menu":
                return

            # ---- Rollback options ----------------------------------------
            if choice == "Undo Last Cleaning Step":
                self._handle_undo_cleaning()
                return

            if choice == "Undo All Cleaning (Restore Original)":
                self._handle_undo_all_cleaning()
                return

            if choice == "View Cleaning History":
                self._handle_view_cleaning_history()
                return

            # ---- Cleaning operations -------------------------------------
            start_time = time.time()
            original_shape = self.current_df.shape

            if choice == "Custom Cleaning (Step by Step)":
                self._handle_custom_cleaning()
            else:
                # Quick cleaning strategies
                strategy_map = {
                    "Quick Clean - Basic": "basic_cleaning",
                    "Quick Clean - Aggressive": "aggressive_cleaning",
                    "Quick Clean - Conservative": "conservative_cleaning",
                    "Quick Clean - ML Ready": "ml_ready"
                }

                strategy_name = strategy_map[choice]
                console.print(f"\n[{self.formatter.colors['info']}]Applying {choice}...[/{self.formatter.colors['info']}]")

                cleaned_df = quick_clean(self.current_df, strategy=strategy_name)

                # Show before/after comparison
                self._show_cleaning_results(original_shape, cleaned_df.shape, time.time() - start_time)

                # Ask if user wants to keep changes
                if self.formatter.prompt_confirmation("Keep these changes?", default=True):
                    self.current_df = cleaned_df
                    self._log_operation("quick_clean", {
                        'strategy': strategy_name,
                        'original_shape': original_shape,
                        'final_shape': cleaned_df.shape
                    })
        
        except Exception as e:
            self._handle_operation_error("Data Cleaning", e)
        
        wait_for_input()
    
    def handle_transform_data(self) -> None:
        """Handle data transformation menu"""
        if not self._check_dataset_loaded():
            return
        
        self.formatter.display_section_header(
            "Data Transformation",
            "Encode, scale, and engineer features",
            "🔄"
        )
        
        try:
            while True:
                choices = [
                    "Encode Categorical Features",
                    "Scale Numeric Features", 
                    "Mathematical Transformations",
                    "Create Polynomial Features",
                    "Create Binned Features",
                    "Feature Selection",
                    "Quick Transform",
                    "Build Custom Pipeline",
                    "View Transformation Summary",
                    "Back to Main Menu"
                ]
                
                choice = self.formatter.prompt_user_choice(
                    "Choose transformation:",
                    choices
                )
                
                if not choice or choice == "Back to Main Menu":
                    break
                
                start_time = time.time()
                
                if choice == "Encode Categorical Features":
                    self._handle_categorical_encoding()
                
                elif choice == "Scale Numeric Features":
                    self._handle_feature_scaling()
                
                elif choice == "Mathematical Transformations":
                    self._handle_mathematical_transforms()
                
                elif choice == "Create Polynomial Features":
                    self._handle_polynomial_features()
                
                elif choice == "Create Binned Features":
                    self._handle_feature_binning()
                
                elif choice == "Feature Selection":
                    self._handle_feature_selection()
                
                elif choice == "Quick Transform":
                    self._handle_quick_transform()
                
                elif choice == "Build Custom Pipeline":
                    self._handle_transformation_pipeline()
                
                elif choice == "View Transformation Summary":
                    # This would show a summary if we had a transformer instance
                    console.print("[yellow]No active transformation session[/yellow]")
                
                duration = time.time() - start_time
                if duration > 1:  # Only show timing for longer operations
                    console.print(f"\n[{self.formatter.colors['dim']}]Transformation completed in {duration:.2f} seconds[/{self.formatter.colors['dim']}]")
                
                wait_for_input()
        
        except Exception as e:
            self._handle_operation_error("Data Transformation", e)
            wait_for_input()
    
    def handle_visualize_data(self) -> None:
        """Handle data visualization menu"""
        if not self._check_dataset_loaded():
            return
        
        self.formatter.display_section_header(
            "Data Visualization",
            "Create plots and charts for data exploration",
            "📈"
        )
        
        try:
            explorer = DataExplorer(self.current_df)
            
            while True:
                choices = [
                    "Histogram",
                    "Scatter Plot",
                    "Correlation Heatmap",
                    "Box Plot (Text-based)",
                    "Data Distribution Summary",
                    "Back to Main Menu"
                ]
                
                choice = self.formatter.prompt_user_choice(
                    "Choose visualization:",
                    choices
                )
                
                if not choice or choice == "Back to Main Menu":
                    break
                
                console.print()
                
                if choice == "Histogram":
                    column = self._prompt_single_column_selection("numeric")
                    if column:
                        bins = IntPrompt.ask("Number of bins", default=20, console=console)
                        explorer.plot_histogram(column, bins=bins)
                
                elif choice == "Scatter Plot":
                    columns = self._prompt_two_column_selection("numeric")
                    if columns and len(columns) == 2:
                        sample_size = IntPrompt.ask(
                            "Maximum points to plot",
                            default=1000,
                            console=console
                        )
                        explorer.plot_scatter(columns[0], columns[1], sample_size=sample_size)
                
                elif choice == "Correlation Heatmap":
                    method = self._prompt_correlation_method()
                    explorer.plot_correlation_heatmap(method=method)
                
                elif choice == "Box Plot (Text-based)":
                    console.print("[yellow]Box plot visualization coming in future update[/yellow]")
                
                elif choice == "Data Distribution Summary":
                    explorer.distribution_summary()
                
                wait_for_input()
        
        except Exception as e:
            self._handle_operation_error("Data Visualization", e)
            wait_for_input()
    
    def handle_export_data(self) -> None:
        """Handle data export menu"""
        if not self._check_dataset_loaded():
            return
        
        self.formatter.display_section_header(
            "Export Data",
            "Save your processed data in various formats",
            "💾"
        )
        
        try:
            # Show export options
            formats = ["CSV", "Excel", "Parquet", "JSON"]
            
            format_choice = self.formatter.prompt_user_choice(
                "Choose export format:",
                formats
            )
            
            if not format_choice:
                return
            
            # Get output path
            default_name = f"{Path(self.current_file).stem}_processed" if self.current_file else "processed_data"
            extension = {
                "CSV": ".csv",
                "Excel": ".xlsx", 
                "Parquet": ".parquet",
                "JSON": ".json"
            }[format_choice]
            
            output_path = self.formatter.prompt_file_path(
                f"Enter output path (default: {default_name}{extension})"
            )
            
            if not output_path.strip():
                output_path = default_name + extension
            
            # Ensure correct extension
            output_path = Path(output_path)
            if output_path.suffix.lower() != extension:
                output_path = output_path.with_suffix(extension)
            
            start_time = time.time()
            
            # Export the data
            self.saver.save(self.current_df, output_path, format_type=format_choice.lower())
            
            duration = time.time() - start_time
            
            self.formatter.display_operation_result(
                "Data Export",
                True,
                f"Saved {len(self.current_df):,} rows to {output_path.name}",
                duration
            )
            
            self._log_operation("export_data", {
                'format': format_choice,
                'output_path': str(output_path),
                'rows': len(self.current_df),
                'columns': len(self.current_df.columns)
            })
        
        except Exception as e:
            self._handle_operation_error("Data Export", e)

        wait_for_input()

    # ------------------------------------------------------------------
    # HTML Profile Report (menu option 10)
    # ------------------------------------------------------------------

    def handle_html_profile(self) -> None:
        """Generate a self-contained HTML data-profile report."""
        if not self._check_dataset_loaded():
            return

        self.formatter.display_section_header(
            "HTML Data Profile Report",
            "Generate a self-contained HTML report for the current dataset",
            "📊"
        )

        try:
            default_name = (
                Path(self.current_file).stem + "_profile.html"
                if self.current_file
                else "data_profile.html"
            )
            out_path = Prompt.ask(
                f"[{self.formatter.colors['primary']}]Output file path[/{self.formatter.colors['primary']}]",
                default=default_name,
                console=console,
            ).strip()

            from wrang.viz.export_utils import generate_html_report
            result = generate_html_report(
                self.current_df,
                output_path=out_path,
                title=f"RIDE Profile — {self.current_file or 'dataset'}",
            )
            console.print(
                f"[{self.formatter.colors['success']}]✅  Report saved to: {result}[/{self.formatter.colors['success']}]"
            )
        except Exception as e:
            self._handle_operation_error("HTML Profile Report", e)

        wait_for_input()

    # ------------------------------------------------------------------
    # DuckDB SQL Query (menu option 9)
    # ------------------------------------------------------------------

    def handle_sql_query(self) -> None:
        """Execute a DuckDB SQL query against the current dataset."""
        if not self._check_dataset_loaded():
            return

        self.formatter.display_section_header(
            "SQL Query Mode",
            "Query your dataset with DuckDB SQL (table name: 'data')",
            "🗄️"
        )

        try:
            import duckdb  # type: ignore
        except ImportError:
            console.print(
                f"[{self.formatter.colors['error']}]"
                "duckdb is not installed. Run: pip install duckdb"
                f"[/{self.formatter.colors['error']}]"
            )
            wait_for_input()
            return

        try:
            import duckdb

            con = duckdb.connect(database=":memory:")
            con.register("data", self.current_df.to_arrow())

            console.print(
                f"[{self.formatter.colors['dim']}]"
                "Enter SQL queries (table: 'data'). Type 'exit' to return."
                f"[/{self.formatter.colors['dim']}]"
            )
            console.print(
                f"[{self.formatter.colors['dim']}]"
                "Example: SELECT * FROM data LIMIT 10"
                f"[/{self.formatter.colors['dim']}]\n"
            )

            while True:
                try:
                    query = Prompt.ask(
                        f"[{self.formatter.colors['primary']}]SQL[/{self.formatter.colors['primary']}]",
                        console=console,
                    ).strip()
                except (EOFError, KeyboardInterrupt):
                    break

                if query.lower() in {"exit", "quit", "q", ""}:
                    break

                try:
                    result = con.execute(query).pl()

                    from rich.table import Table
                    from rich import box

                    tbl = Table(box=box.SIMPLE_HEAD, show_lines=False)
                    for col in result.columns:
                        tbl.add_column(col, style="cyan", no_wrap=True)
                    for row in result.iter_rows():
                        tbl.add_row(*[str(v) if v is not None else "" for v in row])

                    console.print(tbl)
                    console.print(
                        f"[{self.formatter.colors['dim']}]{len(result)} row(s)[/{self.formatter.colors['dim']}]\n"
                    )

                    if self.formatter.prompt_confirmation("Replace current dataset with this result?", default=False):
                        self.current_df = result
                        console.print(
                            f"[{self.formatter.colors['success']}]✅  Dataset updated: {self.current_df.shape}[/{self.formatter.colors['success']}]"
                        )

                except Exception as qe:
                    console.print(f"[{self.formatter.colors['error']}]Query error: {qe}[/{self.formatter.colors['error']}]\n")

        except Exception as e:
            self._handle_operation_error("SQL Query", e)

        wait_for_input()

    # ------------------------------------------------------------------
    # Schema Validation (menu option 11)
    # ------------------------------------------------------------------

    def handle_validate_data(self) -> None:
        """Validate the current dataset against a schema."""
        if not self._check_dataset_loaded():
            return

        self.formatter.display_section_header(
            "Validate Data",
            "Check your dataset against a schema — dtypes, nulls, ranges, allowed values",
            "✅"
        )

        while True:
            choices = [
                "Infer schema from current data",
                "Validate with inferred schema",
                "Validate from JSON schema file",
                "Save inferred schema to JSON",
                "Back to Main Menu",
            ]

            choice = self.formatter.prompt_user_choice(
                "What would you like to do?",
                choices
            )

            if not choice or choice == "Back to Main Menu":
                break

            console.print()

            if choice == "Infer schema from current data":
                self._handle_infer_schema()

            elif choice == "Validate with inferred schema":
                self._handle_validate_inferred()

            elif choice == "Validate from JSON schema file":
                self._handle_validate_from_file()

            elif choice == "Save inferred schema to JSON":
                self._handle_save_schema()

            wait_for_input()

    def _handle_infer_schema(self) -> None:
        """Display the schema inferred from the current dataframe."""
        from rich.table import Table
        from rich import box

        schema = infer_schema(self.current_df)

        tbl = Table(
            title="Inferred Schema",
            box=box.SIMPLE_HEAD,
            show_lines=False,
            title_style=f"bold {self.formatter.colors['primary']}",
        )
        tbl.add_column("Column", style="cyan", no_wrap=True)
        tbl.add_column("dtype", style="magenta")
        tbl.add_column("nullable", justify="center")
        tbl.add_column("max_missing_%", justify="right")
        tbl.add_column("unique", justify="center")

        for cs in schema.columns:
            tbl.add_row(
                cs.name,
                cs.dtype or "—",
                "✓" if cs.nullable else "✗",
                f"{cs.max_missing_pct:.1f}%",
                "✓" if cs.unique else "—",
            )

        console.print(tbl)
        console.print(
            f"[{self.formatter.colors['dim']}]"
            f"{len(schema.columns)} columns inferred. "
            "Use 'Save inferred schema to JSON' to export this as an editable file."
            f"[/{self.formatter.colors['dim']}]"
        )

    def _handle_validate_inferred(self) -> None:
        """Validate the current dataframe against its own inferred schema."""
        schema = infer_schema(self.current_df)
        result = DataValidator(schema).validate(self.current_df)
        result.display()
        summary = result.to_dict()
        console.print(
            f"\n[{self.formatter.colors['dim']}]"
            f"Errors: {summary['error_count']}  |  Warnings: {summary['warning_count']}"
            f"[/{self.formatter.colors['dim']}]"
        )

    def _handle_validate_from_file(self) -> None:
        """Load a JSON schema file and validate the current dataframe against it."""
        schema_path = self.formatter.prompt_file_path(
            "Path to JSON schema file"
        ).strip()

        if not schema_path:
            console.print(
                f"[{self.formatter.colors['warning']}]No path provided.[/{self.formatter.colors['warning']}]"
            )
            return

        try:
            schema = DataSchema.from_json(schema_path)
        except FileNotFoundError:
            console.print(
                f"[{self.formatter.colors['error']}]File not found: {schema_path}[/{self.formatter.colors['error']}]"
            )
            return
        except Exception as e:
            console.print(
                f"[{self.formatter.colors['error']}]Could not load schema: {e}[/{self.formatter.colors['error']}]"
            )
            return

        result = DataValidator(schema).validate(self.current_df)
        result.display()
        summary = result.to_dict()
        console.print(
            f"\n[{self.formatter.colors['dim']}]"
            f"Errors: {summary['error_count']}  |  Warnings: {summary['warning_count']}"
            f"[/{self.formatter.colors['dim']}]"
        )

    def _handle_save_schema(self) -> None:
        """Save the inferred schema to a JSON file."""
        default_stem = (
            Path(self.current_file).stem if self.current_file else "dataset"
        )
        default_name = f"{default_stem}_schema.json"

        out_path = Prompt.ask(
            f"[{self.formatter.colors['primary']}]Save schema to[/{self.formatter.colors['primary']}]",
            default=default_name,
            console=console,
        ).strip()

        if not out_path:
            return

        try:
            schema = infer_schema(self.current_df)
            schema.to_json(out_path)
            console.print(
                f"[{self.formatter.colors['success']}]✅  Schema saved to: {out_path}[/{self.formatter.colors['success']}]"
            )
            console.print(
                f"[{self.formatter.colors['dim']}]"
                "Edit the JSON file to tighten rules (add min_value, max_value, allowed_values, etc.), "
                "then use 'Validate from JSON schema file' to re-run."
                f"[/{self.formatter.colors['dim']}]"
            )
        except Exception as e:
            self._handle_operation_error("Save Schema", e)

    def handle_settings(self) -> None:
        """Handle settings menu"""
        self.formatter.display_section_header(
            "Settings",
            "Configure wrang preferences and behavior",
            "⚙️"
        )
        
        try:
            while True:
                choices = [
                    "View Current Settings",
                    "Memory Settings",
                    "Visualization Settings", 
                    "File Format Settings",
                    "Performance Settings",
                    "Reset to Defaults",
                    "Back to Main Menu"
                ]
                
                choice = self.formatter.prompt_user_choice(
                    "Settings category:",
                    choices
                )
                
                if not choice or choice == "Back to Main Menu":
                    break
                
                if choice == "View Current Settings":
                    self._display_current_settings()
                
                elif choice == "Memory Settings":
                    self._handle_memory_settings()
                
                elif choice == "Visualization Settings":
                    self._handle_visualization_settings()
                
                elif choice == "File Format Settings":
                    self._handle_file_format_settings()
                
                elif choice == "Performance Settings":
                    self._handle_performance_settings()
                
                elif choice == "Reset to Defaults":
                    if self.formatter.prompt_confirmation("Reset all settings to defaults?"):
                        from wrang.config import reset_config
                        self.config = reset_config()
                        console.print(f"[{self.formatter.colors['success']}]Settings reset to defaults[/{self.formatter.colors['success']}]")
                
                wait_for_input()
        
        except Exception as e:
            self._handle_operation_error("Settings", e)
            wait_for_input()
    
    def handle_quick_export(self) -> None:
        """Handle quick export functionality"""
        if not self._check_dataset_loaded():
            return
        
        try:
            # Quick export to CSV with default name
            default_name = f"{Path(self.current_file).stem}_processed.csv" if self.current_file else "processed_data.csv"
            
            start_time = time.time()
            self.saver.save(self.current_df, default_name)
            duration = time.time() - start_time
            
            self.formatter.display_operation_result(
                "Quick Export",
                True,
                f"Saved to {default_name}",
                duration
            )
            
        except Exception as e:
            self._handle_operation_error("Quick Export", e)
        
        wait_for_input()
    
    # Helper methods
    
    def _check_dataset_loaded(self) -> bool:
        """Check if a dataset is loaded"""
        if self.current_df is None:
            console.print(f"[{self.formatter.colors['warning']}]⚠️ No dataset loaded. Please load a dataset first.[/{self.formatter.colors['warning']}]")
            wait_for_input()
            return False
        return True
    
    def _check_no_dataset_warning(self) -> bool:
        """Warn user if dataset is already loaded"""
        if self.current_df is not None:
            if not self.formatter.prompt_confirmation(
                f"Dataset '{self.current_file}' is already loaded. Load a new dataset?",
                default=False
            ):
                return False
        return True
    
    def _handle_exit(self) -> None:
        """Handle application exit"""
        if self.current_df is not None:
            if self.formatter.prompt_confirmation("You have unsaved data. Export before exiting?"):
                self.handle_quick_export()
        
        console.print(f"\n[{self.formatter.colors['success']}]Thank you for using wrang! 🚀[/{self.formatter.colors['success']}]")
        console.print(f"[{self.formatter.colors['dim']}]Visit https://github.com/sudhanshumukherjeexx/wrang for updates[/{self.formatter.colors['dim']}]")
    
    def _handle_operation_error(self, operation: str, error: Exception) -> None:
        """Handle errors from operations"""
        if isinstance(error, RideError):
            error_message = create_user_friendly_message(error)
            console.print(f"[{self.formatter.colors['error']}]{error_message}[/{self.formatter.colors['error']}]")
        else:
            self.formatter.display_error(
                f"{operation} Error",
                str(error),
                suggestions=[
                    "Check your data format and try again",
                    "Verify all required columns exist",
                    "Try with a smaller dataset first"
                ]
            )
    
    def _handle_unexpected_error(self, error: Exception) -> None:
        """Handle unexpected errors"""
        console.print(f"[{self.formatter.colors['error']}]Unexpected error: {error}[/{self.formatter.colors['error']}]")
        console.print(f"[{self.formatter.colors['dim']}]Please report this issue at: https://github.com/sudhanshumukherjeexx/wrang/issues[/{self.formatter.colors['dim']}]")
        wait_for_input()
    
    def _log_operation(self, operation: str, details: Dict[str, Any]) -> None:
        """Log an operation for history"""
        self.operation_history.append({
            'operation': operation,
            'timestamp': time.time(),
            'details': details
        })
    
    # ------------------------------------------------------------------
    # Rollback / undo helpers
    # ------------------------------------------------------------------

    def _handle_undo_cleaning(self) -> None:
        """Undo the last cleaning operation via DataCleaner.undo()."""
        try:
            # DataCleaner.undo() works on its own _undo_stack.  Since we
            # don't keep a persistent cleaner between calls we instead
            # track rollback via MenuHandler._cleaner_snapshots.
            if not hasattr(self, '_cleaner_snapshots') or not self._cleaner_snapshots:
                console.print(f"[{self.formatter.colors['warning']}]No cleaning steps to undo.[/{self.formatter.colors['warning']}]")
                wait_for_input()
                return

            prev_df, label = self._cleaner_snapshots.pop()
            self.current_df = prev_df
            console.print(
                f"[{self.formatter.colors['success']}]✅ Undone: '{label}'. "
                f"DataFrame restored to {self.current_df.shape}[/{self.formatter.colors['success']}]"
            )
        except Exception as e:
            self._handle_operation_error("Undo Cleaning", e)
        wait_for_input()

    def _handle_undo_all_cleaning(self) -> None:
        """Restore the DataFrame to its state before any cleaning."""
        try:
            if not hasattr(self, '_cleaner_snapshots') or not self._cleaner_snapshots:
                console.print(f"[{self.formatter.colors['warning']}]No cleaning history found.[/{self.formatter.colors['warning']}]")
                wait_for_input()
                return

            if self.formatter.prompt_confirmation("Restore dataset to original (pre-cleaning) state?", default=False):
                original_df, _ = self._cleaner_snapshots[0]
                self._cleaner_snapshots.clear()
                self.current_df = original_df
                console.print(
                    f"[{self.formatter.colors['success']}]✅ Restored to original: "
                    f"{self.current_df.shape}[/{self.formatter.colors['success']}]"
                )
        except Exception as e:
            self._handle_operation_error("Undo All Cleaning", e)
        wait_for_input()

    def _handle_view_cleaning_history(self) -> None:
        """Display the snapshot history of cleaning operations."""
        try:
            if not hasattr(self, '_cleaner_snapshots') or not self._cleaner_snapshots:
                console.print(f"[{self.formatter.colors['warning']}]No cleaning history recorded yet.[/{self.formatter.colors['warning']}]")
                wait_for_input()
                return

            from rich.table import Table
            from rich import box

            tbl = Table(title="Cleaning History (oldest → newest)", box=box.SIMPLE_HEAD)
            tbl.add_column("#", style="dim", justify="right")
            tbl.add_column("Operation", style="cyan")
            tbl.add_column("Shape After", style="green")

            for i, (snap_df, label) in enumerate(self._cleaner_snapshots, start=1):
                tbl.add_row(str(i), label, str(snap_df.shape))

            console.print(tbl)
        except Exception as e:
            self._handle_operation_error("View Cleaning History", e)
        wait_for_input()

    def _snapshot_before_clean(self, label: str) -> None:
        """Save a snapshot of current_df before a destructive clean step."""
        if not hasattr(self, '_cleaner_snapshots'):
            self._cleaner_snapshots: List[Tuple] = []
        self._cleaner_snapshots.append((self.current_df.clone(), label))

    # Data cleaning helper methods

    def _handle_custom_cleaning(self) -> None:
        """Handle step-by-step custom cleaning"""
        cleaner = DataCleaner(self.current_df)
        
        while True:
            choices = [
                "Handle Missing Values",
                "Remove Duplicates",
                "Handle Outliers",
                "Validate Data Types",
                "Clean Text Data",
                "View Cleaning Summary",
                "Apply Changes and Finish",
                "Cancel and Return"
            ]
            
            choice = self.formatter.prompt_user_choice(
                "Choose cleaning operation:",
                choices
            )
            
            if not choice or choice == "Cancel and Return":
                break
            elif choice == "Apply Changes and Finish":
                self.current_df = cleaner.get_cleaned_data()
                cleaner.display_cleaning_report()
                console.print(f"[{self.formatter.colors['success']}]✅ Cleaning operations applied[/{self.formatter.colors['success']}]")
                break
            elif choice == "View Cleaning Summary":
                cleaner.display_cleaning_report()
                continue
            
            # Apply the chosen cleaning operation
            if choice == "Handle Missing Values":
                strategy_choices = [s.value.title() for s in ImputationStrategy]
                strategy = self.formatter.prompt_user_choice("Choose imputation strategy:", strategy_choices)
                if strategy:
                    strategy_enum = ImputationStrategy(strategy.lower())
                    cleaner.handle_missing_values(strategy=strategy_enum)
            
            elif choice == "Remove Duplicates":
                keep_choices = ["First", "Last", "None"]
                keep = self.formatter.prompt_user_choice("Which duplicates to keep:", keep_choices)
                if keep:
                    cleaner.remove_duplicates(keep=keep.lower())
            
            elif choice == "Handle Outliers":
                method = self._prompt_outlier_method()
                action_choices = ["Remove", "Cap", "Transform"]
                action = self.formatter.prompt_user_choice("How to handle outliers:", action_choices)
                if method and action:
                    cleaner.handle_outliers(method=method, action=action.lower())
            
            elif choice == "Validate Data Types":
                cleaner.validate_data_types(auto_convert=True)
            
            elif choice == "Clean Text Data":
                operation_choices = ["Strip whitespace", "Lowercase", "Remove special chars", "Normalize whitespace"]
                operations = []
                for op_choice in operation_choices:
                    if self.formatter.prompt_confirmation(f"Apply {op_choice.lower()}?"):
                        op_map = {
                            "Strip whitespace": "strip",
                            "Lowercase": "lower", 
                            "Remove special chars": "remove_special",
                            "Normalize whitespace": "normalize_whitespace"
                        }
                        operations.append(op_map[op_choice])
                
                if operations:
                    cleaner.clean_text_data(operations=operations)
    
    def _show_cleaning_results(self, original_shape: Tuple[int, int], 
                             final_shape: Tuple[int, int], duration: float) -> None:
        """Show before/after cleaning results"""
        before_data = {
            'Rows': original_shape[0],
            'Columns': original_shape[1],
            'Total Cells': original_shape[0] * original_shape[1]
        }
        
        after_data = {
            'Rows': final_shape[0],
            'Columns': final_shape[1], 
            'Total Cells': final_shape[0] * final_shape[1]
        }
        
        comparison_table = self.formatter.create_comparison_table(
            before_data, after_data, "Cleaning Results"
        )
        
        console.print()
        console.print(comparison_table)
        console.print(f"\n[{self.formatter.colors['dim']}]Cleaning completed in {duration:.2f} seconds[/{self.formatter.colors['dim']}]")
    
    # Data transformation helper methods
    
    def _handle_categorical_encoding(self) -> None:
        """Handle categorical encoding"""
        transformer = DataTransformer(self.current_df)
        
        method_choices = [m.value.title() for m in EncodingMethod]
        method = self.formatter.prompt_user_choice("Choose encoding method:", method_choices)
        
        if method:
            method_enum = EncodingMethod(method.lower())
            columns = self._prompt_column_selection("categorical")
            drop_original = self.formatter.prompt_confirmation("Drop original columns?", default=True)
            
            transformer.encode_categorical_features(
                method=method_enum,
                columns=columns,
                drop_original=drop_original
            )
            
            self.current_df = transformer.get_transformed_data()
    
    def _handle_feature_scaling(self) -> None:
        """Handle feature scaling"""
        transformer = DataTransformer(self.current_df)
        
        method_choices = [m.value.title().replace('_', ' ') for m in ScalingMethod]
        method = self.formatter.prompt_user_choice("Choose scaling method:", method_choices)
        
        if method:
            # Convert back to enum
            method_enum_value = method.lower().replace(' ', '_')
            method_enum = ScalingMethod(method_enum_value)
            
            columns = self._prompt_column_selection("numeric")
            
            feature_range = (0, 1)
            if method_enum == ScalingMethod.MINMAX:
                min_val = FloatPrompt.ask("Minimum value for scaling", default=0.0, console=console)
                max_val = FloatPrompt.ask("Maximum value for scaling", default=1.0, console=console)
                feature_range = (min_val, max_val)
            
            transformer.scale_features(
                method=method_enum,
                columns=columns,
                feature_range=feature_range
            )
            
            self.current_df = transformer.get_transformed_data()
    
    def _handle_mathematical_transforms(self) -> None:
        """Handle mathematical transformations"""
        transformer = DataTransformer(self.current_df)
        
        available_transforms = ["log", "log1p", "sqrt", "square", "cube", "reciprocal", "exp", "abs"]
        
        # Get numeric columns
        numeric_cols = [col for col in self.current_df.columns 
                       if str(self.current_df[col].dtype) in ['Int64', 'Int32', 'Float64', 'Float32']]
        
        if not numeric_cols:
            console.print(f"[{self.formatter.colors['warning']}]No numeric columns found for transformation[/{self.formatter.colors['warning']}]")
            return
        
        transforms = {}
        
        for col in numeric_cols:
            console.print(f"\n[{self.formatter.colors['info']}]Column: {col}[/{self.formatter.colors['info']}]")
            
            transform = self.formatter.prompt_user_choice(
                f"Choose transformation for '{col}' (or skip):",
                available_transforms + ["Skip"]
            )
            
            if transform and transform != "Skip":
                transforms[col] = transform
        
        if transforms:
            create_new = self.formatter.prompt_confirmation(
                "Create new columns (vs modify existing)?", 
                default=True
            )
            
            transformer.apply_mathematical_transforms(
                transforms=transforms,
                create_new_columns=create_new
            )
            
            self.current_df = transformer.get_transformed_data()
    
    def _handle_polynomial_features(self) -> None:
        """Handle polynomial feature creation"""
        transformer = DataTransformer(self.current_df)
        
        degree = IntPrompt.ask("Polynomial degree", default=2, console=console)
        interaction_only = self.formatter.prompt_confirmation(
            "Create interaction features only?", 
            default=False
        )
        include_bias = self.formatter.prompt_confirmation(
            "Include bias column?", 
            default=False
        )
        
        columns = self._prompt_column_selection("numeric")
        
        transformer.create_polynomial_features(
            columns=columns,
            degree=degree,
            interaction_only=interaction_only,
            include_bias=include_bias
        )
        
        self.current_df = transformer.get_transformed_data()
    
    def _handle_feature_binning(self) -> None:
        """Handle feature binning"""
        transformer = DataTransformer(self.current_df)
        
        n_bins = IntPrompt.ask("Number of bins", default=5, console=console)
        
        strategy_choices = ["Quantile", "Uniform", "KMeans"]
        strategy = self.formatter.prompt_user_choice("Choose binning strategy:", strategy_choices)
        
        if strategy:
            columns = self._prompt_column_selection("numeric")
            
            transformer.create_binned_features(
                columns=columns,
                n_bins=n_bins,
                strategy=strategy.lower()
            )
            
            self.current_df = transformer.get_transformed_data()
    
    def _handle_feature_selection(self) -> None:
        """Handle feature selection"""
        transformer = DataTransformer(self.current_df)
        
        # Get target column
        target_col = self._prompt_single_column_selection("any", "Select target column for feature selection")
        if not target_col:
            return
        
        method_choices = ["Mutual Info", "Chi2", "F Classif"]
        method = self.formatter.prompt_user_choice("Choose selection method:", method_choices)
        
        if method:
            k = IntPrompt.ask("Number of features to select", default=10, console=console)
            
            method_map = {
                "Mutual Info": "mutual_info",
                "Chi2": "chi2", 
                "F Classif": "f_classif"
            }
            
            transformer.select_features(
                target_column=target_col,
                method=method_map[method],
                k=k
            )
            
            self.current_df = transformer.get_transformed_data()
    
    def _handle_quick_transform(self) -> None:
        """Handle quick transformation"""
        encode = self.formatter.prompt_confirmation("Encode categorical features?", default=True)
        scale = self.formatter.prompt_confirmation("Scale numeric features?", default=True)
        
        if not encode and not scale:
            console.print(f"[{self.formatter.colors['warning']}]No transformations selected[/{self.formatter.colors['warning']}]")
            return
        
        encoding_method = "label"
        scaling_method = "standard"
        
        if encode:
            enc_choices = ["Label", "Onehot"]
            enc_choice = self.formatter.prompt_user_choice("Encoding method:", enc_choices)
            if enc_choice:
                encoding_method = enc_choice.lower()
        
        if scale:
            scale_choices = ["Standard", "Minmax", "Robust"]
            scale_choice = self.formatter.prompt_user_choice("Scaling method:", scale_choices)
            if scale_choice:
                scaling_method = scale_choice.lower()
        
        transformed_df = quick_transform(
            self.current_df,
            encode_categorical=encode,
            scale_features=scale,
            encoding_method=encoding_method,
            scaling_method=scaling_method
        )
        
        self.current_df = transformed_df
    
    def _handle_transformation_pipeline(self) -> None:
        """Handle transformation pipeline creation"""
        pipeline = create_pipeline()
        
        console.print(f"[{self.formatter.colors['info']}]Building transformation pipeline...[/{self.formatter.colors['info']}]")
        
        while True:
            choices = [
                "Add Encoding Step",
                "Add Scaling Step",
                "Add Mathematical Transform Step",
                "Add Polynomial Step",
                "Add Binning Step",
                "Add Feature Selection Step",
                "View Pipeline",
                "Execute Pipeline",
                "Cancel"
            ]
            
            choice = self.formatter.prompt_user_choice("Pipeline builder:", choices)
            
            if not choice or choice == "Cancel":
                break
            elif choice == "Execute Pipeline":
                self.current_df = pipeline.fit_transform(self.current_df)
                break
            elif choice == "View Pipeline":
                pipeline.display_pipeline()
                continue
            
            # Add steps to pipeline
            step_name = Prompt.ask("Step name", console=console)
            
            if choice == "Add Encoding Step":
                method = self.formatter.prompt_user_choice("Encoding method:", ["label", "onehot", "ordinal"])
                if method:
                    pipeline.add_step(step_name, "encode", method=method)
            
            elif choice == "Add Scaling Step":
                method = self.formatter.prompt_user_choice("Scaling method:", ["standard", "minmax", "robust"])
                if method:
                    pipeline.add_step(step_name, "scale", method=method)
            
            # Add other step types similarly...
            
            pipeline.display_pipeline()
    
    # Column selection helper methods
    
    def _prompt_column_selection(self, column_type: str = "any", 
                               message: Optional[str] = None) -> Optional[List[str]]:
        """Prompt user to select columns of a specific type"""
        if column_type == "numeric":
            available_cols = [col for col in self.current_df.columns 
                            if str(self.current_df[col].dtype) in ['Int64', 'Int32', 'Float64', 'Float32']]
            type_name = "numeric"
        elif column_type == "categorical":
            available_cols = [col for col in self.current_df.columns 
                            if str(self.current_df[col].dtype) == 'Utf8']
            type_name = "categorical"
        else:
            available_cols = list(self.current_df.columns)
            type_name = "any"
        
        if not available_cols:
            console.print(f"[{self.formatter.colors['warning']}]No {type_name} columns found[/{self.formatter.colors['warning']}]")
            return None
        
        if not message:
            message = f"Select {type_name} columns (comma-separated numbers, or 'all')"
        
        # Display available columns
        console.print(f"\n[{self.formatter.colors['info']}]Available {type_name} columns:[/{self.formatter.colors['info']}]")
        for i, col in enumerate(available_cols, 1):
            console.print(f"  {i}. {col}")
        
        selection = Prompt.ask(message, console=console).strip()
        
        if selection.lower() == 'all':
            return available_cols
        
        try:
            indices = [int(x.strip()) - 1 for x in selection.split(',')]
            selected_cols = [available_cols[i] for i in indices if 0 <= i < len(available_cols)]
            return selected_cols if selected_cols else None
        except (ValueError, IndexError):
            console.print(f"[{self.formatter.colors['error']}]Invalid selection[/{self.formatter.colors['error']}]")
            return None
    
    def _prompt_single_column_selection(self, column_type: str = "any", 
                                      message: Optional[str] = None) -> Optional[str]:
        """Prompt user to select a single column"""
        columns = self._prompt_column_selection(column_type, message)
        return columns[0] if columns else None
    
    def _prompt_two_column_selection(self, column_type: str = "any") -> Optional[List[str]]:
        """Prompt user to select exactly two columns"""
        columns = self._prompt_column_selection(column_type, "Select exactly 2 columns (comma-separated numbers)")
        
        if columns and len(columns) == 2:
            return columns
        elif columns:
            console.print(f"[{self.formatter.colors['warning']}]Please select exactly 2 columns[/{self.formatter.colors['warning']}]")
        
        return None
    
    def _prompt_correlation_method(self) -> str:
        """Prompt user for correlation method"""
        methods = ["Pearson", "Spearman", "Kendall"]
        method = self.formatter.prompt_user_choice("Choose correlation method:", methods)
        return method.lower() if method else "pearson"
    
    def _prompt_outlier_method(self) -> str:
        """Prompt user for outlier detection method"""
        methods = ["IQR", "Z-score", "Modified Z-score"]
        method = self.formatter.prompt_user_choice("Choose outlier detection method:", methods)
        method_map = {"IQR": "iqr", "Z-score": "zscore", "Modified Z-score": "modified_zscore"}
        return method_map.get(method, "iqr")
    
    # Display helper methods
    
    def _display_memory_analysis(self, memory_info: Dict[str, Any]) -> None:
        """Display memory usage analysis"""
        content = f"""[bold]Memory Usage Analysis[/bold]

[bold]Total Memory:[/bold] {memory_info['total_memory_human']}
[bold]Memory per MB:[/bold] {memory_info['total_memory_mb']:.2f} MB

[bold]Largest Columns:[/bold]"""
        
        for col, size_mb in memory_info['largest_columns']:
            content += f"\n  • {col}: {size_mb:.2f} MB"
        
        panel = Panel(content, title="💾 Memory Analysis", border_style=self.formatter.colors['info'])
        console.print(panel)
    
    def _display_data_issues(self, issues: List[Dict[str, Any]]) -> None:
        """Display detected data issues"""
        if not issues:
            console.print(f"[{self.formatter.colors['success']}]✅ No data quality issues detected[/{self.formatter.colors['success']}]")
            return
        
        # Group issues by severity
        high_issues = [i for i in issues if i['severity'] == 'high']
        medium_issues = [i for i in issues if i['severity'] == 'medium'] 
        low_issues = [i for i in issues if i['severity'] == 'low']
        
        content = f"[bold]Data Quality Issues Found[/bold]\n"
        
        if high_issues:
            content += f"\n[bold red]🚨 High Priority ({len(high_issues)} issues):[/bold red]"
            for issue in high_issues:
                content += f"\n  • {issue['column']}: {issue['details']}"
                content += f"\n    💡 {issue['recommendation']}"
        
        if medium_issues:
            content += f"\n\n[bold yellow]⚠️ Medium Priority ({len(medium_issues)} issues):[/bold yellow]"
            for issue in medium_issues:
                content += f"\n  • {issue['column']}: {issue['details']}"
        
        if low_issues:
            content += f"\n\n[bold blue]ℹ️ Low Priority ({len(low_issues)} issues):[/bold blue]"
            for issue in low_issues:
                content += f"\n  • {issue['column']}: {issue['details']}"
        
        panel = Panel(content, title="🔍 Data Quality Report", 
                     border_style=self.formatter.colors['warning'])
        console.print(panel)
    
    # Settings helper methods
    
    def _display_current_settings(self) -> None:
        """Display current configuration settings"""
        settings_content = f"""[bold]Current wrang Settings[/bold]

[bold]Performance:[/bold]
• Memory Limit: {self.config.max_memory_usage_mb} MB
• Chunk Size: {self.config.chunk_size:,}
• Sample Size: {self.config.sample_size:,}

[bold]Visualization:[/bold] 
• Plot Width: {self.config.plot_width}
• Plot Height: {self.config.plot_height}
• Max Categories: {self.config.max_categories_plot}

[bold]Data Processing:[/bold]
• Missing Threshold: {self.config.missing_value_threshold*100:.1f}%
• Correlation Threshold: {self.config.correlation_threshold}
• Outlier Method: {self.config.outlier_method}

[bold]Features:[/bold]
• Max One-Hot Features: {self.config.max_features_for_onehot}
• Max Label Features: {self.config.max_features_for_label}
• Random State: {self.config.random_state}"""
        
        panel = Panel(settings_content, title="⚙️ Configuration", 
                     border_style=self.formatter.colors['info'])
        console.print(panel)
    
    def _handle_memory_settings(self) -> None:
        """Handle memory-related settings"""
        console.print(f"[{self.formatter.colors['info']}]Current memory limit: {self.config.max_memory_usage_mb} MB[/{self.formatter.colors['info']}]")
        
        new_limit = IntPrompt.ask("New memory limit (MB)", 
                                 default=self.config.max_memory_usage_mb, 
                                 console=console)
        
        new_chunk = IntPrompt.ask("Chunk size for large files",
                                 default=self.config.chunk_size,
                                 console=console)
        
        update_config(max_memory_usage_mb=new_limit, chunk_size=new_chunk)
        self.config = get_config()
        
        console.print(f"[{self.formatter.colors['success']}]Memory settings updated[/{self.formatter.colors['success']}]")
    
    def _handle_visualization_settings(self) -> None:
        """Handle visualization settings"""
        new_width = IntPrompt.ask("Plot width", 
                                 default=self.config.plot_width, 
                                 console=console)
        
        new_height = IntPrompt.ask("Plot height",
                                  default=self.config.plot_height,
                                  console=console)
        
        max_cats = IntPrompt.ask("Maximum categories in plots",
                                default=self.config.max_categories_plot,
                                console=console)
        
        update_config(plot_width=new_width, plot_height=new_height, max_categories_plot=max_cats)
        self.config = get_config()
        
        console.print(f"[{self.formatter.colors['success']}]Visualization settings updated[/{self.formatter.colors['success']}]")
    
    def _handle_file_format_settings(self) -> None:
        """Handle file format settings"""
        console.print(f"[{self.formatter.colors['info']}]Current CSV delimiter: '{self.config.csv_delimiter}'[/{self.formatter.colors['info']}]")
        
        new_delimiter = Prompt.ask("CSV delimiter", 
                                  default=self.config.csv_delimiter, 
                                  console=console)
        
        new_encoding = Prompt.ask("Default file encoding",
                                 default=self.config.default_encoding,
                                 console=console)
        
        update_config(csv_delimiter=new_delimiter, default_encoding=new_encoding)
        self.config = get_config()
        
        console.print(f"[{self.formatter.colors['success']}]File format settings updated[/{self.formatter.colors['success']}]")
    
    def _handle_performance_settings(self) -> None:
        """Handle performance settings"""
        new_sample = IntPrompt.ask("Default sample size for previews",
                                  default=self.config.sample_size,
                                  console=console)
        
        new_random_state = IntPrompt.ask("Random state for reproducibility",
                                        default=self.config.random_state,
                                        console=console)
        
        parallel = self.formatter.prompt_confirmation("Enable parallel processing?", 
                                                     default=self.config.parallel_processing)
        
        update_config(sample_size=new_sample, random_state=new_random_state, 
                     parallel_processing=parallel)
        self.config = get_config()
        
        console.print(f"[{self.formatter.colors['success']}]Performance settings updated[/{self.formatter.colors['success']}]")