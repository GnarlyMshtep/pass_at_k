#!/usr/bin/env python3
"""
Interactive Parquet File Viewer for SFT Data

Navigate through parquet files with keyboard controls:
- Arrow Up/Down: Move between rows
- Arrow Left/Right: Switch between different views
- Page Up/Down: Jump by 10 rows
- Home/End: Jump to first/last row
- 1-9: Quick jump to specific sample
- 'm': Toggle between showing messages, metadata, or all data
- 'p': Show all plans for current sample
- 's': Show statistics
- 'q': Quit

Usage:
    python view_parquet.py <parquet_file_path>
"""

import argparse
import sys
import json
from typing import Any, Dict, List
import pandas as pd
from rich.console import Console
from rich.panel import Panel
from rich.table import Table
from rich.text import Text
from rich.layout import Layout
from rich import box
import os

# Try to import keyboard handling
try:
    import readchar
    READCHAR_AVAILABLE = True
except ImportError:
    READCHAR_AVAILABLE = False
    print("Note: Install 'readchar' for better keyboard navigation: pip install readchar")


class ParquetViewer:
    def __init__(self, parquet_path: str):
        self.parquet_path = parquet_path
        self.console = Console()
        self.df = None
        self.current_row = 0
        self.view_mode = 'messages'  # 'messages', 'metadata', 'all'
        self.load_data()
        
    def load_data(self):
        """Load the parquet file."""
        self.console.print(f"[cyan]Loading {self.parquet_path}...[/cyan]")
        try:
            self.df = pd.read_parquet(self.parquet_path)
            self.console.print(f"[green]✓ Loaded {len(self.df)} rows[/green]\n")
        except Exception as e:
            self.console.print(f"[red]Error loading file: {e}[/red]")
            sys.exit(1)
    
    def get_statistics(self) -> Table:
        """Generate statistics table."""
        table = Table(title="Dataset Statistics", box=box.ROUNDED)
        table.add_column("Metric", style="cyan")
        table.add_column("Value", style="green")
        
        table.add_row("Total Rows", str(len(self.df)))
        table.add_row("Columns", str(len(self.df.columns)))
        table.add_row("Column Names", ", ".join(self.df.columns))
        
        # Check for unique UIDs
        if 'extra_info' in self.df.columns:
            try:
                uids = self.df['extra_info'].apply(lambda x: x.get('uid', None) if isinstance(x, dict) else None)
                unique_uids = uids.nunique()
                table.add_row("Unique UIDs", str(unique_uids))
                
                # Count attempts per UID
                attempt_counts = self.df['extra_info'].apply(
                    lambda x: x.get('attempt_num', None) if isinstance(x, dict) else None
                )
                table.add_row("Attempts per UID", f"{len(self.df) / unique_uids:.1f}")
            except:
                pass
        
        # Memory usage
        memory_mb = self.df.memory_usage(deep=True).sum() / 1024 / 1024
        table.add_row("Memory Usage", f"{memory_mb:.2f} MB")
        
        return table
    
    def format_message(self, msg: Dict[str, str], width: int = 100) -> str:
        """Format a single message."""
        role = msg.get('role', 'unknown')
        content = msg.get('content', '')
        
        # No truncation - show full content
        return f"[bold {('cyan' if role == 'user' else 'yellow')}]{role.upper()}:[/bold {('cyan' if role == 'user' else 'yellow')}]\n{content}"
    
    def format_extra_info(self, extra_info: Dict[str, Any]) -> str:
        """Format extra_info dictionary."""
        # Fields to exclude from display
        excluded_fields = {
            'variable_labels', 'uid', 'split', 'solution', 'raw_sat', 
            'prompt_style', 'num_variables', 'num_clauses', 'num_attempts',
            'max_allowed_attempts', 'index'
        }
        
        lines = []
        for key, value in extra_info.items():
            if key in excluded_fields:
                continue
                
            # Convert numpy arrays to lists
            if hasattr(value, 'tolist'):
                value = value.tolist()
                
            if key == 'all_plans':
                lines.append(f"[cyan]{key}:[/cyan] <{len(value)} plans available>")
            elif key == 'plan_used':
                lines.append(f"[cyan]{key}:[/cyan] {value}")
            elif isinstance(value, (str, int, float, bool)):
                lines.append(f"[cyan]{key}:[/cyan] {value}")
            elif isinstance(value, list):
                lines.append(f"[cyan]{key}:[/cyan] <{len(value)} items>")
            else:
                lines.append(f"[cyan]{key}:[/cyan] <{type(value).__name__}>")
        return "\n".join(lines)
    
    def display_row(self, row_idx: int):
        """Display a single row."""
        self.console.clear()
        
        if row_idx < 0 or row_idx >= len(self.df):
            self.console.print("[red]Invalid row index[/red]")
            return
        
        row = self.df.iloc[row_idx]
        
        # Header
        header_text = Text()
        header_text.append(f"Row {row_idx + 1} of {len(self.df)}", style="bold cyan")
        header_text.append(f" | View: {self.view_mode}", style="bold yellow")
        header_text.append(f" | File: {os.path.basename(self.parquet_path)}", style="dim")
        self.console.print(Panel(header_text, box=box.DOUBLE))
        
        # Display based on view mode
        if self.view_mode == 'messages' or self.view_mode == 'all':
            # Show messages (handle both list and numpy array)
            if 'messages' in row:
                messages = row['messages']
                # Convert numpy array to list if needed
                if hasattr(messages, 'tolist'):
                    messages = messages.tolist()
                
                if messages and isinstance(messages, list):
                    for i, msg in enumerate(messages):
                        msg_text = self.format_message(msg)
                        self.console.print(Panel(
                            msg_text,
                            title=f"Message {i+1}/{len(messages)}",
                            border_style="blue"
                        ))
                        self.console.print()
        
        if self.view_mode == 'metadata' or self.view_mode == 'all':
            # Show metadata
            metadata_table = Table(title="Metadata", box=box.ROUNDED, show_header=False)
            metadata_table.add_column("Field", style="cyan")
            metadata_table.add_column("Value", style="white")
            
            # Show basic fields
            for col in self.df.columns:
                if col == 'messages':
                    continue
                elif col == 'extra_info' and isinstance(row[col], dict):
                    metadata_table.add_row("extra_info", self.format_extra_info(row[col]))
                else:
                    value_str = str(row[col])
                    if len(value_str) > 200:
                        value_str = value_str[:200] + "..."
                    metadata_table.add_row(col, value_str)
            
            self.console.print(metadata_table)
            self.console.print()
        
        # Controls footer
        controls = """
[bold cyan]Controls:[/bold cyan]
  ↑/↓: Previous/Next row  |  ←/→: Not used  |  PgUp/PgDn: Jump ±10 rows
  Home/End: First/Last    |  1-9: Jump to specific row
  [yellow]m[/yellow]: Toggle view mode  |  [yellow]p[/yellow]: Show all plans  |  [yellow]s[/yellow]: Statistics  |  [yellow]q[/yellow]: Quit
"""
        self.console.print(Panel(controls, border_style="dim"))
    
    def show_all_plans(self, row_idx: int):
        """Display all plans for the current sample."""
        if row_idx < 0 or row_idx >= len(self.df):
            return
        
        row = self.df.iloc[row_idx]
        
        if 'extra_info' in row and isinstance(row['extra_info'], dict):
            extra_info = row['extra_info']
            if 'all_plans' in extra_info:
                plans = extra_info['all_plans']
                
                # Convert numpy array to list if needed
                if hasattr(plans, 'tolist'):
                    plans = plans.tolist()
                
                self.console.clear()
                
                # Show attempt number instead of UID since we're excluding UID
                attempt_info = f"Row {row_idx + 1}"
                if 'attempt_num' in extra_info:
                    attempt_info += f" (Attempt {extra_info['attempt_num']})"
                
                self.console.print(Panel(
                    f"All Plans for {attempt_info}",
                    style="bold cyan"
                ))
                
                for i, plan in enumerate(plans, 1):
                    is_used = (extra_info.get('attempt_num', -1) == i)
                    title = f"Plan {i}" + (" [USED IN THIS SAMPLE]" if is_used else "")
                    style = "green" if is_used else "white"
                    
                    self.console.print(Panel(
                        plan,
                        title=title,
                        border_style=style
                    ))
                    self.console.print()
                
                self.console.print("\n[dim]Press any key to return...[/dim]")
                if READCHAR_AVAILABLE:
                    readchar.readkey()
                else:
                    input()
    
    def show_statistics(self):
        """Display dataset statistics."""
        self.console.clear()
        self.console.print(self.get_statistics())
        self.console.print("\n[dim]Press any key to return...[/dim]")
        if READCHAR_AVAILABLE:
            readchar.readkey()
        else:
            input()
    
    def run_interactive(self):
        """Run the interactive viewer."""
        if not READCHAR_AVAILABLE:
            self.console.print("[yellow]Warning: readchar not available. Using simple input mode.[/yellow]")
            self.console.print("[yellow]Install with: pip install readchar[/yellow]\n")
            return self.run_simple()
        
        while True:
            self.display_row(self.current_row)
            
            try:
                key = readchar.readkey()
                
                # Navigation
                if key == readchar.key.UP:
                    self.current_row = max(0, self.current_row - 1)
                elif key == readchar.key.DOWN:
                    self.current_row = min(len(self.df) - 1, self.current_row + 1)
                elif key == readchar.key.PAGE_UP:
                    self.current_row = max(0, self.current_row - 10)
                elif key == readchar.key.PAGE_DOWN:
                    self.current_row = min(len(self.df) - 1, self.current_row + 10)
                elif key == readchar.key.HOME:
                    self.current_row = 0
                elif key == readchar.key.END:
                    self.current_row = len(self.df) - 1
                
                # Mode toggles
                elif key.lower() == 'm':
                    modes = ['messages', 'metadata', 'all']
                    current_idx = modes.index(self.view_mode)
                    self.view_mode = modes[(current_idx + 1) % len(modes)]
                elif key.lower() == 'p':
                    self.show_all_plans(self.current_row)
                elif key.lower() == 's':
                    self.show_statistics()
                
                # Quick jump
                elif key.isdigit() and key != '0':
                    jump_to = int(key) - 1
                    if jump_to < len(self.df):
                        self.current_row = jump_to
                
                # Quit
                elif key.lower() == 'q':
                    self.console.clear()
                    self.console.print("[cyan]Goodbye![/cyan]")
                    break
                    
            except KeyboardInterrupt:
                self.console.clear()
                self.console.print("[cyan]Goodbye![/cyan]")
                break
    
    def run_simple(self):
        """Run a simple non-interactive viewer."""
        while True:
            self.display_row(self.current_row)
            
            command = input("\nEnter command (n=next, p=prev, q=quit, s=stats, #=jump, h=help): ").strip().lower()
            
            if command == 'n':
                self.current_row = min(len(self.df) - 1, self.current_row + 1)
            elif command == 'p':
                self.current_row = max(0, self.current_row - 1)
            elif command == 'q':
                break
            elif command == 's':
                self.show_statistics()
            elif command == 'm':
                modes = ['messages', 'metadata', 'all']
                current_idx = modes.index(self.view_mode)
                self.view_mode = modes[(current_idx + 1) % len(modes)]
            elif command == 'h':
                self.console.print("""
Commands:
  n - Next row
  p - Previous row
  q - Quit
  s - Show statistics
  m - Toggle view mode
  # - Jump to row number (e.g., '5' for row 5)
  h - Show this help
""")
                input("Press Enter to continue...")
            elif command.isdigit():
                jump_to = int(command) - 1
                if 0 <= jump_to < len(self.df):
                    self.current_row = jump_to
                else:
                    self.console.print(f"[red]Invalid row number. Must be between 1 and {len(self.df)}[/red]")
                    input("Press Enter to continue...")


def main():
    parser = argparse.ArgumentParser(
        description="Interactive viewer for parquet files generated by generate_multi_plan_sft_data.py"
    )
    parser.add_argument(
        "parquet_file",
        type=str,
        help="Path to the parquet file to view"
    )
    parser.add_argument(
        "--no-interactive",
        action="store_true",
        help="Use simple input mode instead of interactive mode"
    )
    
    args = parser.parse_args()
    
    if not os.path.exists(args.parquet_file):
        print(f"Error: File not found: {args.parquet_file}")
        sys.exit(1)
    
    viewer = ParquetViewer(args.parquet_file)
    
    if args.no_interactive or not READCHAR_AVAILABLE:
        viewer.run_simple()
    else:
        viewer.run_interactive()


if __name__ == "__main__":
    main()

