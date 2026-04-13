#!/usr/bin/env python3
"""
wrang Formatters
Beautiful terminal output formatting with Rich components
"""

import polars as pl
import shutil
from typing import Dict, Any, List, Optional, Union, Tuple
from datetime import datetime
from rich.console import Console
from rich.table import Table
from rich.panel import Panel
from rich.columns import Columns
from rich.progress import Progress, SpinnerColumn, TextColumn, BarColumn, TimeElapsedColumn
from rich.text import Text
from rich.align import Align
from rich.layout import Layout
from rich.live import Live
from rich.prompt import Prompt, Confirm, IntPrompt, FloatPrompt
from rich.syntax import Syntax
from rich.tree import Tree
from rich.rule import Rule
from rich.box import ROUNDED, DOUBLE, SIMPLE, HEAVY
from pyfiglet import Figlet

from wrang.config import get_config
from wrang import __version__ as _version

console = Console()


class RideFormatter:
    """
    Beautiful formatting for wrang output
    
    Provides consistent, attractive formatting for all CLI interactions
    with automatic terminal size adaptation and rich visual elements.
    """
    
    def __init__(self):
        self.config = get_config()
        self.terminal_width = shutil.get_terminal_size().columns
        self.console = Console()
        
        # Color themes
        self.colors = {
            'primary': 'cyan',
            'secondary': 'blue', 
            'success': 'green',
            'warning': 'yellow',
            'error': 'red',
            'info': 'white',
            'accent': 'magenta',
            'dim': 'bright_black'
        }
        
        # ASCII art fonts
        self.fonts = {
            'large': Figlet(font='big'),
            'medium': Figlet(font='standard'),
            'small': Figlet(font='small'),
            'banner': Figlet(font='banner3-D'),
            'digital': Figlet(font='digital')
        }
    
    def display_welcome_banner(self) -> None:
        """Display the main RIDE welcome banner"""
        self._clear_screen()
        
        # Create dynamic banner based on terminal width
        if self.terminal_width >= 100:
            banner_text = self.fonts['banner'].renderText('wrang')
            subtitle = "data wrangling toolkit"
            description = "fast · clean · simple"
        elif self.terminal_width >= 80:
            banner_text = self.fonts['large'].renderText('wrang')
            subtitle = "data wrangling toolkit"
            description = "fast · clean · simple"
        else:
            banner_text = self.fonts['medium'].renderText('wrang')
            subtitle = "wrangling toolkit"
            description = "fast · simple"
        
        # Create banner panel
        banner_content = f"""[bold {self.colors['primary']}]{banner_text}[/bold {self.colors['primary']}]

[bold {self.colors['secondary']}]{subtitle}[/bold {self.colors['secondary']}]
[{self.colors['dim']}]{description}[/{self.colors['dim']}]

[{self.colors['info']}]v{_version} | Built with Polars[/{self.colors['info']}]"""
        
        banner_panel = Panel(
            Align.center(banner_content),
            box=DOUBLE,
            border_style=self.colors['primary'],
            padding=(1, 2)
        )
        
        self.console.print(banner_panel)
        
        # Add decorative separator
        self.console.print(Rule(style=self.colors['primary']))
    
    def display_main_menu(self, current_dataset: Optional[str] = None) -> None:
        """Display the main menu with current dataset info"""
        
        # Dataset status panel
        if current_dataset:
            dataset_info = f"[{self.colors['success']}]✅ Dataset Loaded[/{self.colors['success']}]\n"
            dataset_info += f"[{self.colors['info']}]📁 {current_dataset}[/{self.colors['info']}]"
        else:
            dataset_info = f"[{self.colors['warning']}]⚠️ No Dataset Loaded[/{self.colors['warning']}]\n"
            dataset_info += f"[{self.colors['dim']}]Please load a dataset first[/{self.colors['dim']}]"
        
        status_panel = Panel(
            dataset_info,
            title="📊 Current Status",
            border_style=self.colors['success'] if current_dataset else self.colors['warning'],
            box=ROUNDED,
            width=40
        )
        
        # Menu options
        menu_content = self._create_menu_table()
        
        menu_panel = Panel(
            menu_content,
            title="🚀 Main Menu",
            border_style=self.colors['primary'],
            box=ROUNDED,
            expand=True
        )
        
        # Display in columns for better layout
        if self.terminal_width >= 120:
            layout = Columns([status_panel, menu_panel], equal=False, expand=True)
            self.console.print(layout)
        else:
            self.console.print(status_panel)
            self.console.print()
            self.console.print(menu_panel)
    
    def _create_menu_table(self) -> Table:
        """Create the main menu table"""
        menu_table = Table(
            show_header=False,
            box=None,
            pad_edge=False,
            collapse_padding=True
        )
        
        menu_table.add_column("Number", style=f"bold {self.colors['accent']}", width=4)
        menu_table.add_column("Icon", style=self.colors['primary'], width=4)
        menu_table.add_column("Option", style=f"bold {self.colors['info']}", min_width=20)
        menu_table.add_column("Description", style=self.colors['dim'], min_width=30)
        
        menu_items = [
            ("1", "📂", "Load Dataset", "Import CSV, Excel, Parquet, or JSON files"),
            ("2", "🔍", "Inspect Data", "Explore dataset structure and quality"),
            ("3", "📊", "Explore Data", "Statistical analysis and correlations"),
            ("4", "🧹", "Clean Data", "Handle missing values and duplicates"),
            ("5", "🔄", "Transform Data", "Encode and scale features"),
            ("6", "📈", "Visualize Data", "Create plots and charts"),
            ("7", "💾", "Export Data", "Save processed data"),
            ("8", "⚙️", "Settings", "Configure wrang preferences"),
            ("9", "🗄️", "SQL Query", "Query dataset with DuckDB SQL"),
            ("10", "📄", "HTML Profile", "Generate full HTML report"),
            ("11", "✅", "Validate Data", "Schema validation and quality checks"),
            ("", "", "", ""),
            ("$", "💾", "Quick Export", "Export current dataset"),
            ("q", "🚪", "Exit", "Quit wrang")
        ]
        
        for number, icon, option, description in menu_items:
            if not number:  # Empty row for spacing
                menu_table.add_row("", "", "", "")
                continue
                
            menu_table.add_row(number, icon, option, description)
        
        return menu_table
    
    def display_section_header(self, title: str, subtitle: Optional[str] = None, 
                             icon: str = "🔧") -> None:
        """Display a section header with icon and optional subtitle"""
        
        # Create header content
        if subtitle:
            header_content = f"[bold {self.colors['primary']}]{icon} {title}[/bold {self.colors['primary']}]\n"
            header_content += f"[{self.colors['dim']}]{subtitle}[/{self.colors['dim']}]"
        else:
            header_content = f"[bold {self.colors['primary']}]{icon} {title}[/bold {self.colors['primary']}]"
        
        # Create panel
        header_panel = Panel(
            Align.center(header_content),
            box=HEAVY,
            border_style=self.colors['primary'],
            padding=(0, 2)
        )
        
        self.console.print()
        self.console.print(header_panel)
        self.console.print()
    
    def display_data_summary(self, df: pl.DataFrame, filename: Optional[str] = None) -> None:
        """Display a beautiful data summary"""
        
        # Basic info
        n_rows, n_cols = df.shape
        memory_mb = df.estimated_size('mb')
        
        # Column type breakdown
        numeric_cols = len([col for col in df.columns if df[col].dtype in [pl.Int64, pl.Int32, pl.Float64, pl.Float32]])
        text_cols = len([col for col in df.columns if df[col].dtype == pl.Utf8])
        other_cols = n_cols - numeric_cols - text_cols
        
        # Missing values
        total_missing = df.null_count().sum_horizontal().item()
        missing_pct = (total_missing / (n_rows * n_cols)) * 100 if n_rows * n_cols > 0 else 0
        
        # Create summary panels
        overview_content = f"""[bold]Dataset Overview[/bold]

📏 [bold]Dimensions:[/bold] {n_rows:,} rows × {n_cols} columns
💾 [bold]Memory Usage:[/bold] {memory_mb:.1f} MB
📊 [bold]Column Types:[/bold]
   • Numeric: {numeric_cols}
   • Text: {text_cols}  
   • Other: {other_cols}

🚫 [bold]Missing Values:[/bold] {total_missing:,} ({missing_pct:.1f}%)"""
        
        if filename:
            overview_content = f"📁 [bold]File:[/bold] {filename}\n\n" + overview_content
        
        overview_panel = Panel(
            overview_content,
            title="📋 Data Summary",
            border_style=self.colors['info'],
            box=ROUNDED
        )
        
        # Sample data preview
        sample_data = df.head(5)
        preview_table = self.create_data_table(sample_data, title="Preview (First 5 Rows)")
        
        # Display layout
        if self.terminal_width >= 120:
            layout = Columns([overview_panel, preview_table], equal=False)
            self.console.print(layout)
        else:
            self.console.print(overview_panel)
            self.console.print()
            self.console.print(preview_table)
    
    def create_data_table(self, df: pl.DataFrame, title: str = "Data", 
                         max_rows: int = 20, max_cols: int = 10) -> Table:
        """Create a beautiful data table from DataFrame"""
        
        # Limit columns and rows for display
        display_df = df.head(max_rows)
        columns_to_show = df.columns[:max_cols]
        
        if len(df.columns) > max_cols:
            columns_to_show = df.columns[:max_cols-1] + ['...']
        
        # Create table
        table = Table(
            title=title,
            show_header=True,
            header_style=f"bold {self.colors['primary']}",
            box=ROUNDED,
            border_style=self.colors['secondary']
        )
        
        # Add columns
        for col in columns_to_show:
            if col == '...':
                table.add_column(col, style=self.colors['dim'], width=5)
            else:
                # Determine column style based on data type
                dtype = str(df[col].dtype)
                if 'int' in dtype.lower() or 'float' in dtype.lower():
                    style = self.colors['accent']
                elif 'utf8' in dtype.lower() or 'str' in dtype.lower():
                    style = self.colors['info']
                else:
                    style = self.colors['dim']
                
                table.add_column(
                    col,
                    style=style,
                    width=min(15, max(8, len(col) + 2))
                )
        
        # Add rows
        for i in range(min(max_rows, len(display_df))):
            row_data = []
            
            for col in columns_to_show:
                if col == '...':
                    row_data.append('...')
                else:
                    value = display_df[col][i]
                    if value is None:
                        row_data.append(Text("NULL", style=self.colors['dim']))
                    else:
                        # Format value based on type
                        str_value = str(value)
                        if len(str_value) > 15:
                            str_value = str_value[:12] + "..."
                        row_data.append(str_value)
            
            table.add_row(*row_data)
        
        # Add footer if data was truncated
        if len(df) > max_rows or len(df.columns) > max_cols:
            footer_text = f"... showing {min(max_rows, len(df))} of {len(df)} rows"
            if len(df.columns) > max_cols:
                footer_text += f", {max_cols} of {len(df.columns)} columns"
            
            table.caption = footer_text
            table.caption_style = self.colors['dim']
        
        return table
    
    def display_operation_result(self, operation: str, success: bool, 
                               details: Optional[str] = None,
                               duration: Optional[float] = None) -> None:
        """Display the result of an operation"""
        
        if success:
            icon = "✅"
            status = "Success"
            color = self.colors['success']
        else:
            icon = "❌"
            status = "Failed"
            color = self.colors['error']
        
        result_content = f"[bold {color}]{icon} {operation} {status}[/bold {color}]"
        
        if details:
            result_content += f"\n[{self.colors['info']}]{details}[/{self.colors['info']}]"
        
        if duration:
            result_content += f"\n[{self.colors['dim']}]Completed in {duration:.2f} seconds[/{self.colors['dim']}]"
        
        result_panel = Panel(
            result_content,
            border_style=color,
            box=ROUNDED,
            padding=(0, 1)
        )
        
        self.console.print()
        self.console.print(result_panel)
    
    def display_error(self, error_title: str, error_message: str, 
                     suggestions: Optional[List[str]] = None) -> None:
        """Display a formatted error message with suggestions"""
        
        error_content = f"[bold {self.colors['error']}]❌ {error_title}[/bold {self.colors['error']}]\n\n"
        error_content += f"[{self.colors['info']}]{error_message}[/{self.colors['info']}]"
        
        if suggestions:
            error_content += f"\n\n[bold {self.colors['warning']}]💡 Suggestions:[/bold {self.colors['warning']}]"
            for suggestion in suggestions:
                error_content += f"\n[{self.colors['dim']}]  • {suggestion}[/{self.colors['dim']}]"
        
        error_panel = Panel(
            error_content,
            title="Error",
            border_style=self.colors['error'],
            box=HEAVY,
            padding=(1, 2)
        )
        
        self.console.print()
        self.console.print(error_panel)
    
    def display_progress_operation(self, operation_name: str, items: List[Any], 
                                 operation_func: callable) -> Any:
        """Display a progress bar for long operations"""
        
        results = []
        
        with Progress(
            SpinnerColumn(),
            TextColumn("[progress.description]{task.description}"),
            BarColumn(),
            TextColumn("[progress.percentage]{task.percentage:>3.0f}%"),
            TimeElapsedColumn(),
            console=self.console
        ) as progress:
            
            task = progress.add_task(f"[{self.colors['primary']}]{operation_name}...", total=len(items))
            
            for item in items:
                result = operation_func(item)
                results.append(result)
                progress.advance(task)
        
        return results
    
    def create_comparison_table(self, before_data: Dict[str, Any], 
                              after_data: Dict[str, Any],
                              title: str = "Before vs After") -> Table:
        """Create a before/after comparison table"""
        
        table = Table(
            title=title,
            show_header=True,
            header_style=f"bold {self.colors['primary']}",
            box=ROUNDED,
            border_style=self.colors['secondary']
        )
        
        table.add_column("Metric", style=f"bold {self.colors['info']}")
        table.add_column("Before", style=self.colors['warning'])
        table.add_column("After", style=self.colors['success'])
        table.add_column("Change", style=self.colors['accent'])
        
        for key in before_data.keys():
            if key in after_data:
                before_val = before_data[key]
                after_val = after_data[key]
                
                # Calculate change
                if isinstance(before_val, (int, float)) and isinstance(after_val, (int, float)):
                    change = after_val - before_val
                    if before_val != 0:
                        change_pct = (change / before_val) * 100
                        change_str = f"{change:+.0f} ({change_pct:+.1f}%)"
                    else:
                        change_str = f"{change:+.0f}"
                else:
                    change_str = "Changed" if str(before_val) != str(after_val) else "No change"
                
                table.add_row(
                    key.replace('_', ' ').title(),
                    str(before_val),
                    str(after_val),
                    change_str
                )
        
        return table
    
    def display_column_info(self, df: pl.DataFrame, detailed: bool = False) -> None:
        """Display detailed column information"""
        
        table = Table(
            title="📋 Column Information",
            show_header=True,
            header_style=f"bold {self.colors['primary']}",
            box=ROUNDED,
            border_style=self.colors['secondary']
        )
        
        table.add_column("#", style=f"bold {self.colors['accent']}", width=3)
        table.add_column("Column", style=f"bold {self.colors['info']}", min_width=15)
        table.add_column("Type", style=self.colors['success'], width=12)
        table.add_column("Missing", style=self.colors['warning'], width=8)
        table.add_column("Unique", style=self.colors['primary'], width=8)
        
        if detailed:
            table.add_column("Min", style=self.colors['dim'], width=10)
            table.add_column("Max", style=self.colors['dim'], width=10)
            table.add_column("Sample", style=self.colors['info'], min_width=20)
        
        for i, col in enumerate(df.columns, 1):
            # Basic info
            dtype = str(df[col].dtype)
            missing_count = df[col].null_count()
            unique_count = df[col].n_unique()
            
            row_data = [
                str(i),
                col[:15] if len(col) > 15 else col,
                dtype.split('(')[0],  # Remove Polars type parameters
                str(missing_count),
                str(unique_count)
            ]
            
            if detailed:
                # Additional details for detailed view
                try:
                    if dtype in ['Int64', 'Int32', 'Float64', 'Float32']:
                        min_val = df[col].min()
                        max_val = df[col].max()
                        row_data.extend([
                            f"{min_val:.2f}" if min_val is not None else "N/A",
                            f"{max_val:.2f}" if max_val is not None else "N/A"
                        ])
                    else:
                        row_data.extend(["N/A", "N/A"])
                    
                    # Sample values
                    sample = df[col].drop_nulls().head(3).to_list()
                    sample_str = ", ".join(str(x)[:8] for x in sample)
                    if len(sample_str) > 20:
                        sample_str = sample_str[:17] + "..."
                    row_data.append(sample_str)
                    
                except Exception:
                    row_data.extend(["Error", "Error", "Error"])
            
            table.add_row(*row_data)
        
        self.console.print(table)
    
    def prompt_user_choice(self, question: str, choices: List[str], 
                          default: Optional[str] = None) -> str:
        """Prompt user for a choice with validation"""
        
        # Create choices display
        choices_text = "\n".join(f"  {i+1}. {choice}" for i, choice in enumerate(choices))
        
        prompt_panel = Panel(
            f"[bold {self.colors['primary']}]{question}[/bold {self.colors['primary']}]\n\n{choices_text}",
            border_style=self.colors['info'],
            box=ROUNDED
        )
        
        self.console.print(prompt_panel)
        
        while True:
            try:
                if default:
                    response = Prompt.ask(
                        f"Enter choice (1-{len(choices)})",
                        default=str(default),
                        console=self.console
                    )
                else:
                    response = Prompt.ask(
                        f"Enter choice (1-{len(choices)})",
                        console=self.console
                    )
                
                choice_num = int(response)
                if 1 <= choice_num <= len(choices):
                    return choices[choice_num - 1]
                else:
                    self.console.print(f"[{self.colors['error']}]Please enter a number between 1 and {len(choices)}[/{self.colors['error']}]")
            
            except ValueError:
                self.console.print(f"[{self.colors['error']}]Please enter a valid number[/{self.colors['error']}]")
            except KeyboardInterrupt:
                self.console.print(f"\n[{self.colors['warning']}]Operation cancelled[/{self.colors['warning']}]")
                return ""
    
    def prompt_confirmation(self, message: str, default: bool = False) -> bool:
        """Prompt user for yes/no confirmation"""
        return Confirm.ask(
            f"[{self.colors['warning']}]{message}[/{self.colors['warning']}]",
            default=default,
            console=self.console
        )
    
    def prompt_file_path(self, message: str = "Enter file path") -> str:
        """Prompt user for file path with validation"""
        return Prompt.ask(
            f"[{self.colors['info']}]{message}[/{self.colors['info']}]",
            console=self.console
        )
    
    def _clear_screen(self) -> None:
        """Clear the terminal screen"""
        import os
        os.system('cls' if os.name == 'nt' else 'clear')
    
    def display_help(self, topic: str) -> None:
        """Display help information for a specific topic"""
        
        help_content = {
            'main': """[bold]wrang Help[/bold]

[bold {self.colors['primary']}]Navigation:[/bold {self.colors['primary']}]
• Enter the number of your choice and press Enter
• Use 'q' to quit from any menu
• Use '$' for quick export from the main menu

[bold {self.colors['primary']}]File Formats Supported:[/bold {self.colors['primary']}]
• CSV (.csv) - Comma-separated values
• Excel (.xlsx, .xls) - Microsoft Excel files  
• Parquet (.parquet) - Columnar storage format
• JSON (.json) - JavaScript Object Notation

[bold {self.colors['primary']}]Tips:[/bold {self.colors['primary']}]
• Always inspect your data before processing
• Clean your data before transformation
• Export intermediate results to save progress""",
            
            'load': """[bold]Loading Data[/bold]

[bold {self.colors['primary']}]Supported Formats:[/bold {self.colors['primary']}]
• CSV files with automatic delimiter detection
• Excel files (both .xlsx and .xls)
• Parquet files for fast loading
• JSON files with automatic structure detection

[bold {self.colors['primary']}]Tips:[/bold {self.colors['primary']}]
• Large files (>100MB) will show progress bars
• Memory usage is optimized with Polars
• Invalid files will show helpful error messages""",
            
            'clean': """[bold]Data Cleaning[/bold]

[bold {self.colors['primary']}]Available Operations:[/bold {self.colors['primary']}]
• Handle missing values (multiple strategies)
• Remove duplicate rows
• Detect and handle outliers
• Validate data types
• Clean text data

[bold {self.colors['primary']}]Missing Value Strategies:[/bold {self.colors['primary']}]
• Drop: Remove rows/columns with missing data
• Mean/Median: Fill with statistical measures
• Forward/Backward Fill: Use adjacent values
• KNN: Use similar rows to predict values"""
        }
        
        content = help_content.get(topic, f"Help for '{topic}' not available.")
        
        help_panel = Panel(
            content,
            title=f"📚 Help: {topic.title()}",
            border_style=self.colors['info'],
            box=ROUNDED,
            padding=(1, 2)
        )
        
        self.console.print()
        self.console.print(help_panel)
        
        # Wait for user to read
        Prompt.ask("\nPress Enter to continue", default="", console=self.console)


# Global formatter instance
_formatter = None

def get_formatter() -> RideFormatter:
    """Get global formatter instance"""
    global _formatter
    if _formatter is None:
        _formatter = RideFormatter()
    return _formatter


def clear_screen() -> None:
    """Clear the terminal screen"""
    import os
    os.system('cls' if os.name == 'nt' else 'clear')


def wait_for_input(message: str = "Press Enter to continue...") -> None:
    """Wait for user input before continuing"""
    formatter = get_formatter()
    Prompt.ask(f"[{formatter.colors['dim']}]{message}[/{formatter.colors['dim']}]", 
              default="", console=formatter.console)