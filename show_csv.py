#!/usr/bin/env python3
"""Display a CSV summary in the terminal with colors.

Uses DISPLAY_COLUMNS from ai_skills.config as single source of truth.
The column names in DISPLAY_COLUMNS are the actual DataFrame column names.
"""

import argparse
import sys
from pathlib import Path

import pandas as pd

# Add package to path if running as script
if __name__ == "__main__":
    sys.path.insert(0, str(Path(__file__).parent))

from ai_skills.config import DISPLAY_COLUMNS, DEFAULT_DISPLAY_COLS

# ANSI color codes
RESET = "\033[0m"
BOLD = "\033[1m"
GREEN = "\033[92m"
YELLOW = "\033[93m"
RED = "\033[91m"
GRAY = "\033[90m"
MAGENTA = "\033[95m"
BLUE = "\033[94m"
TEAL = "\033[36m"

# Tier colors
TIER_COLORS = {
    "core_ai": MAGENTA,
    "applied_ai": BLUE,
    "ai_integration": TEAL,
    "none": GRAY,
}


def colorize(text: str, color: str) -> str:
    """Apply ANSI color to text."""
    return f"{color}{text}{RESET}"


def format_value(col_name: str, value, width: int, fmt_type: str) -> str:
    """Format a cell value based on its display type."""
    # Handle None/NaN
    if pd.isna(value):
        return " " * width
    
    if fmt_type == "tier":
        if value == "none":
            return " " * width
        tier_str = f"{value:<{width}}"
        return colorize(tier_str, TIER_COLORS.get(str(value), ""))
    
    elif fmt_type == "confidence":
        try:
            val_float = float(value)
            conf_str = f"{val_float:.2f}"
            if val_float <= 0.75:
                return colorize(conf_str.rjust(width), RED)
            elif val_float < 0.85:
                return colorize(conf_str.rjust(width), YELLOW)
            return conf_str.rjust(width)
        except (ValueError, TypeError):
            return f"{str(value)[:width]:<{width}}"
    
    elif fmt_type == "boolean":
        try:
            val_str = str(value).lower().strip()
            if val_str in ("1", "true", "yes"):
                return colorize("Yes".center(width), GREEN)
            elif val_str in ("0", "false", "no"):
                return colorize("No".center(width), GRAY)
            return " " * width
        except (ValueError, TypeError):
            return " " * width
    
    elif fmt_type == "skills":
        s = str(value) if pd.notna(value) else ""
        s = "" if s.lower() == "nan" else s
        s = s[:width]
        return f"{s:<{width}}"
    
    else:  # text
        s = str(value)[:width] if pd.notna(value) else ""
        return f"{s:<{width}}"


def show_csv(filepath: str, sep: str, cols: list[str]) -> None:
    """Display CSV with distribution and all rows."""
    try:
        df = pd.read_csv(filepath, sep=sep, encoding="utf-8-sig", low_memory=False)
    except Exception as e:
        print(f"{RED}Error reading CSV: {e}{RESET}")
        sys.exit(1)

    # Show AI tier distribution if column exists
    tier_col = "desc_tier_llm"
    if tier_col in df.columns:
        print(f"{BOLD}AI Tier Distribution:{RESET}")
        counts = df[tier_col].value_counts()
        for tier, count in counts.items():
            color = TIER_COLORS.get(str(tier), "")
            print(f"  {colorize(f'{tier:<14}', color)} {count}")
    print()
    
    # Filter to only columns that exist in the DataFrame
    valid_cols = []
    for col_name in cols:
        if col_name not in df.columns:
            print(f"{YELLOW}Warning: '{col_name}' not in CSV, skipping.{RESET}")
            continue
        valid_cols.append(col_name)
    
    if not valid_cols:
        print(f"{RED}No valid columns to display.{RESET}")
        sys.exit(1)
    
    # Build header
    header_parts = [f"{'IDX':>4}"]
    for col_name in valid_cols:
        col_display = DISPLAY_COLUMNS.get(col_name)
        if col_display:
            header_parts.append(f"{col_display.label:<{col_display.width}}")
        else:
            # Column exists in CSV but not in DISPLAY_COLUMNS - show raw name
            header_parts.append(f"{col_name[:20]:<20}")
    print(f"{BOLD}{'  '.join(header_parts)}{RESET}")
    
    # Separator line
    total_width = 4
    for col_name in valid_cols:
        col_display = DISPLAY_COLUMNS.get(col_name)
        total_width += (col_display.width if col_display else 20) + 2
    print("-" * total_width)
    
    # Print rows
    for idx, row in df.iterrows():
        parts = [f"{idx+1:>4}"]
        for col_name in valid_cols:
            value = row.get(col_name)
            col_display = DISPLAY_COLUMNS.get(col_name)
            if col_display:
                parts.append(format_value(col_name, value, col_display.width, col_display.format_type))
            else:
                # Fallback for columns not in DISPLAY_COLUMNS
                s = str(value)[:20] if pd.notna(value) else ""
                parts.append(f"{s:<20}")
        print("  ".join(parts))


def main():
    """Main entry point."""
    parser = argparse.ArgumentParser(
        description="Display CSV with colored output",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=f"""
Available display columns (with formatting):
{chr(10).join(f'  {name:<20} ({col.label})' for name, col in DISPLAY_COLUMNS.items())}

Default: {', '.join(DEFAULT_DISPLAY_COLS[:4])}...
"""
    )
    parser.add_argument("file", help="Path to CSV file")
    parser.add_argument("-s", "--sep", default=";", help="CSV separator (default: ;)")
    parser.add_argument(
        "-c", "--cols", 
        nargs="+", 
        default=list(DEFAULT_DISPLAY_COLS),
        help="Column names to display"
    )
    parser.add_argument(
        "--all", 
        action="store_true",
        help="Show all columns from DISPLAY_COLUMNS"
    )
    
    args = parser.parse_args()
    
    cols_to_show = list(DISPLAY_COLUMNS.keys()) if args.all else args.cols
    show_csv(args.file, args.sep, cols_to_show)


if __name__ == "__main__":
    main()
