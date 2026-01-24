#!/usr/bin/env python3
"""Check if duplicate IDs have identical job content."""
import pandas as pd

OUTPUT_FILE = "data/outputs/us_relevant_ai.csv"

print(f"Reading {OUTPUT_FILE}...")
# Read only the columns we need to speed up
df = pd.read_csv(OUTPUT_FILE, sep=";", usecols=['id', 'job_title', 'desc_tier_llm', 'company'], 
                 engine='python', on_bad_lines='skip')

print(f"Total rows: {len(df)}")

# Find IDs that appear more than once
dup_mask = df['id'].duplicated(keep=False)
dup_df = df[dup_mask]
dup_ids = dup_df['id'].unique()[:10]  # Check first 10 duplicated IDs

print(f"\nChecking {len(dup_ids)} sample duplicate IDs...\n")

for job_id in dup_ids:
    rows = df[df['id'] == job_id]
    titles = rows['job_title'].unique()
    companies = rows['company'].unique()
    tiers = rows['desc_tier_llm'].unique()
    
    is_identical = len(titles) == 1 and len(companies) == 1
    status = "IDENTICAL" if is_identical else "DIFFERENT!"
    
    print(f"ID {job_id}: {len(rows)} rows, Titles: {len(titles)}, Companies: {len(companies)} -> {status}")
    if not is_identical:
        print(f"  Titles: {list(titles)[:3]}")
        print(f"  Companies: {list(companies)[:3]}")
