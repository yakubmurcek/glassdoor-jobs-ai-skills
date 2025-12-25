#!/Users/yakub/Python/ai-skills/.venv/bin/python
"""Display a CSV summary in the terminal with colors."""

import argparse
import sys
import pandas as pd

# ANSI color codes for tiers (distinct from status colors)
COLORS = {
    "core_ai": "\033[95m",       # Magenta - most AI focused
    "applied_ai": "\033[94m",    # Blue - applied AI
    "ai_integration": "\033[36m", # Teal - uses AI tools
    "none": "\033[90m",          # Gray - no AI
}
RESET = "\033[0m"
BOLD = "\033[1m"
GREEN = "\033[92m"
YELLOW = "\033[93m"
RED = "\033[91m"

# Available columns
COLUMNS = {
    "title": ("job_title_short", "JOB TITLE", 40),
    "tier": ("AI_tier_openai", "TIER", 14),
    "conf": ("AI_skill_openai_confidence", "CONF", 5),
    "hard": ("AI_skill_hard", "HARD", 4),
    "agree": ("AI_skill_agreement", "AGREE", 5),
    "skills_hard": ("AI_skills_found", "SKILLS (HARD)", 30),
    "skills_openai": ("AI_skills_openai_mentioned", "SKILLS (OPENAI)", 40),
}

DEFAULT_COLS = ["title", "tier", "conf", "hard", "agree", "skills_hard", "skills_openai"]

def colorize(text: str, color: str) -> str:
    return f"{color}{text}{RESET}"

def format_value(col_key: str, value, width: int) -> str:
    """Format a cell value with optional coloring."""
    if col_key == "tier":
        if value == "none":
            return " " * width
        tier_str = f"{value:<{width}}"
        return colorize(tier_str, COLORS.get(value, ""))
    elif col_key == "conf":
        if pd.notna(value):
            conf_str = f"{value:.2f}"
            if value <= 0.75:
                return colorize(conf_str, RED)
            elif value < 0.85:
                return colorize(conf_str, YELLOW)
            return conf_str
        return " N/A"
    elif col_key == "hard":
        return colorize(" Yes", GREEN) if value else "  No"
    elif col_key == "agree":
        return colorize(" Yes", GREEN) if value else colorize("  No", RED)
    elif col_key in ("skills_hard", "skills_openai"):
        s = str(value) if pd.notna(value) else ""
        s = "" if s == "nan" else s[:width]
        return f"{s:<{width}}"
    else:
        s = str(value)[:width] if pd.notna(value) else ""
        return f"{s:<{width}}"

def show_csv(filepath: str, sep: str, cols: list):
    """Display CSV with distribution and all rows."""
    df = pd.read_csv(filepath, sep=sep)
    df["job_title_short"] = df["job_title"].str[:40]
    
    # Show AI tier distribution
    print(f"{BOLD}AI_tier_openai distribution:{RESET}")
    counts = df["AI_tier_openai"].value_counts()
    for tier, count in counts.items():
        color = COLORS.get(tier, "")
        print(f"  {colorize(f'{tier:<14}', color)} {count}")
    print()
    
    # Build header
    header_parts = [f"{'IDX':>3}"]
    for col_key in cols:
        _, label, width = COLUMNS[col_key]
        header_parts.append(f"{label:<{width}}")
    print(f"{BOLD}{'  '.join(header_parts)}{RESET}")
    print("-" * (sum(COLUMNS[c][2] for c in cols) + len(cols) * 2 + 5))
    
    # Print rows
    for idx, row in df.iterrows():
        parts = [f"{idx+1:>3}"]
        for col_key in cols:
            col_name, _, width = COLUMNS[col_key]
            parts.append(format_value(col_key, row.get(col_name), width))
        print("  ".join(parts))

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Display CSV with colored output")
    parser.add_argument("file", help="Path to CSV file")
    parser.add_argument("-s", "--sep", default=";", help="CSV separator (default: ;)")
    parser.add_argument("-c", "--cols", nargs="+", choices=list(COLUMNS.keys()),
                        default=DEFAULT_COLS,
                        help=f"Columns to display. Available: {', '.join(COLUMNS.keys())}")
    
    args = parser.parse_args()
    show_csv(args.file, args.sep, args.cols)
