#!/usr/bin/env python3
"""
Simple Parquet Viewer - Shows prompts and answers from SFT data

Usage:
    python show_parquet.py <parquet_file> [--start N] [--limit M] [--stats]
"""

import argparse
import sys
import pandas as pd
from rich.console import Console
from rich.panel import Panel
from rich.table import Table
from rich import box


def show_statistics(df, console):
    """Display dataset statistics."""
    console.print("\n[bold cyan]Dataset Statistics[/bold cyan]")
    console.print("=" * 80)
    console.print(f"Total rows: [green]{len(df)}[/green]")
    console.print(f"Columns: {', '.join(df.columns)}")
    
    if 'extra_info' in df.columns:
        try:
            uids = df['extra_info'].apply(lambda x: x.get('uid', None) if isinstance(x, dict) else None)
            unique_uids = uids.nunique()
            console.print(f"Unique UIDs: [green]{unique_uids}[/green]")
            console.print(f"Attempts per UID: [green]{len(df) / unique_uids:.1f}[/green]")
        except:
            pass
    
    console.print("=" * 80 + "\n")


def show_sample(row, idx, total, console):
    """Display a single sample with prompt and answer."""
    console.print("\n" + "=" * 80)
    console.print(f"[bold cyan]Sample {idx + 1} of {total}[/bold cyan]")
    
    # Fields to exclude from extra_info display
    excluded_fields = {
        'variable_labels', 'uid', 'split', 'solution', 'raw_sat', 
        'prompt_style', 'num_variables', 'num_clauses', 'num_attempts',
        'max_allowed_attempts', 'index'
    }
    
    # Show filtered extra info if available
    if 'extra_info' in row and isinstance(row['extra_info'], dict):
        extra = row['extra_info']
        info_parts = []
        
        # Show only non-excluded fields
        for key, value in extra.items():
            if key not in excluded_fields:
                # Convert numpy arrays to lists
                if hasattr(value, 'tolist'):
                    value = value.tolist()
                
                if isinstance(value, (str, int, float, bool)):
                    info_parts.append(f"{key}: {value}")
                elif isinstance(value, list):
                    if key == 'all_plans':
                        info_parts.append(f"{key}: <{len(value)} plans available>")
                    else:
                        info_parts.append(f"{key}: <{len(value)} items>")
                else:
                    info_parts.append(f"{key}: <{type(value).__name__}>")
        
        if info_parts:
            console.print("[dim]" + " | ".join(info_parts) + "[/dim]")
    
    console.print("=" * 80)
    
    # Show messages (handle both list and numpy array)
    if 'messages' in row:
        messages = row['messages']
        # Convert numpy array to list if needed
        if hasattr(messages, 'tolist'):
            messages = messages.tolist()
        
        if messages and isinstance(messages, list):
            for msg in messages:
                role = msg.get('role', 'unknown')
                content = msg.get('content', '')
                
                if role == 'user':
                    console.print(Panel(
                        content,
                        title="[bold cyan]USER PROMPT[/bold cyan]",
                        border_style="cyan",
                        box=box.ROUNDED
                    ))
                elif role == 'assistant':
                    console.print(Panel(
                        content,
                        title="[bold yellow]ASSISTANT RESPONSE[/bold yellow]",
                        border_style="yellow",
                        box=box.ROUNDED
                    ))
                else:
                    console.print(Panel(
                        content,
                        title=f"[bold]{role.upper()}[/bold]",
                        border_style="white"
                    ))
                console.print()


def main():
    parser = argparse.ArgumentParser(
        description="Display prompts and answers from parquet file"
    )
    parser.add_argument(
        "parquet_file",
        type=str,
        help="Path to the parquet file to view"
    )
    parser.add_argument(
        "--start",
        type=int,
        default=0,
        help="Start row index (0-based, default: 0)"
    )
    parser.add_argument(
        "--limit",
        type=int,
        default=5,
        help="Number of samples to display (default: 5)"
    )
    parser.add_argument(
        "--stats",
        action="store_true",
        help="Show statistics only"
    )
    parser.add_argument(
        "--all",
        action="store_true",
        help="Show all samples (overrides --limit)"
    )
    
    args = parser.parse_args()
    
    console = Console()
    
    # Load data
    console.print(f"[cyan]Loading {args.parquet_file}...[/cyan]")
    try:
        df = pd.read_parquet(args.parquet_file)
        console.print(f"[green]✓ Loaded {len(df)} rows[/green]")
    except Exception as e:
        console.print(f"[red]Error loading file: {e}[/red]")
        sys.exit(1)
    
    # Show statistics
    show_statistics(df, console)
    
    if args.stats:
        return
    
    # Determine range
    start = args.start
    if args.all:
        limit = len(df)
    else:
        limit = args.limit
    
    end = min(start + limit, len(df))
    
    if start >= len(df):
        console.print(f"[red]Start index {start} is beyond dataset size {len(df)}[/red]")
        sys.exit(1)
    
    console.print(f"[cyan]Showing samples {start + 1} to {end}[/cyan]\n")
    
    # Display samples
    for idx in range(start, end):
        show_sample(df.iloc[idx], idx, len(df), console)
    
    # Show navigation hint
    if end < len(df):
        console.print("\n" + "=" * 80)
        console.print(f"[dim]To see more samples, use: --start {end} --limit {limit}[/dim]")
        console.print("=" * 80)


if __name__ == "__main__":
    main()

