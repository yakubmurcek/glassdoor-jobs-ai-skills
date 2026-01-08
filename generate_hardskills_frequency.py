#!/usr/bin/env python3
"""Generate a frequency table of hard skills from the analyzed AI job data."""

import csv
from collections import Counter
from pathlib import Path


def main():
    input_file = Path("data/outputs/us_relevant_ai.csv")
    output_file = Path("data/outputs/hardskills_frequency.csv")
    
    # Read hard skills from input CSV
    skill_counter = Counter()
    total_rows = 0
    rows_with_skills = 0
    
    with open(input_file, "r", encoding="utf-8") as f:
        reader = csv.DictReader(f, delimiter=";")
        for row in reader:
            total_rows += 1
            hardskills = row.get("hardskills", "").strip()
            
            if hardskills:
                rows_with_skills += 1
                # Skills are comma-separated
                skills = [s.strip().lower() for s in hardskills.split(",") if s.strip()]
                skill_counter.update(skills)
    
    # Sort by frequency descending
    sorted_skills = skill_counter.most_common()
    
    # Write output CSV
    with open(output_file, "w", encoding="utf-8", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(["skill", "frequency", "percentage"])
        for skill, count in sorted_skills:
            percentage = round((count / rows_with_skills) * 100, 2)
            writer.writerow([skill, count, percentage])
    
    # Print summary
    print(f"Total rows processed: {total_rows}")
    print(f"Rows with hard skills: {rows_with_skills}")
    print(f"Unique hard skills found: {len(sorted_skills)}")
    print(f"\nTop 20 hard skills:")
    for skill, count in sorted_skills[:20]:
        pct = round((count / rows_with_skills) * 100, 2)
        print(f"  {skill}: {count} ({pct}%)")
    print(f"\nOutput saved to: {output_file}")


if __name__ == "__main__":
    main()
