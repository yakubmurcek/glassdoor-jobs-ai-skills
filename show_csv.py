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
    "edu": ("EDUCATION2", "EDU", 10),
    "edu_req": ("EDUCATION2_REQUIRED", "EDU_REQ", 7),
    "hardskills": ("hardskills", "HARDSKILLS", 50),
    "softskills": ("softskills", "SOFTSKILLS", 30),
    "skills_hard": ("AI_skills_found", "SKILLS (HARD)", 30),
    "skills_openai": ("AI_skills_openai_mentioned", "SKILLS (OPENAI)", 40),
}

DEFAULT_COLS = ["title", "tier", "conf", "hard", "agree", "edu", "edu_req", "hardskills", "softskills", "skills_hard", "skills_openai"]

def colorize(text: str, color: str) -> str:
    return f"{color}{text}{RESET}"

def format_value(col_key: str, value, width: int) -> str:
    """Format a cell value with optional coloring."""
    if value is None:
        return " " * width

    if col_key == "tier":
        if value == "none":
            return " " * width
        tier_str = f"{value:<{width}}"
        return colorize(tier_str, COLORS.get(value, ""))
    elif col_key == "conf":
        if pd.notna(value):
            try:
                val_float = float(value)
                conf_str = f"{val_float:.2f}"
                if val_float <= 0.75:
                    return colorize(conf_str, RED)
                elif val_float < 0.85:
                    return colorize(conf_str, YELLOW)
                return conf_str
            except (ValueError, TypeError):
                return f"{str(value)[:width]:<{width}}"
        return " " * width
    elif col_key == "hard":
        # Check explicit True/False or 1/0, carefully handling None
        if pd.isna(value):
             return " " * width
        val_bool = bool(int(value)) if str(value).isdigit() else bool(value)
        return colorize(" Yes", GREEN) if val_bool else "  No"
    elif col_key == "agree":
        if pd.isna(value):
             return " " * width
        # agreement is usually 1 or 0
        try:
            val_bool = bool(int(value))
            return colorize(" Yes", GREEN) if val_bool else colorize("  No", RED)
        except (ValueError, TypeError):
             return " " * width
    elif col_key in ("skills_hard", "skills_openai"):
        s = str(value) if pd.notna(value) else ""
        s = "" if s.lower() == "nan" else s
        s = s[:width]
        return f"{s:<{width}}"
    else:
        s = str(value)[:width] if pd.notna(value) else ""
        return f"{s:<{width}}"

def show_csv(filepath: str, sep: str, cols: list):
    """Display CSV with distribution and all rows."""
    try:
        df = pd.read_csv(filepath, sep=sep, encoding="utf-8-sig", low_memory=False)
    except Exception as e:
        print(f"{RED}Error reading CSV: {e}{RESET}")
        sys.exit(1)

    # Ensure job_title exists
    if "job_title" in df.columns:
        df["job_title_short"] = df["job_title"].fillna("").astype(str).str[:40]
    else:
        df["job_title_short"] = ""

    # Show AI tier distribution only if column exists
    if "AI_tier_openai" in df.columns:
        print(f"{BOLD}AI_tier_openai distribution:{RESET}")
        counts = df["AI_tier_openai"].value_counts()
        for tier, count in counts.items():
            color = COLORS.get(tier, "")
            print(f"  {colorize(f'{tier:<14}', color)} {count}")
    else:
        print(f"{YELLOW}No AI tier data found.{RESET}")
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
