#!/usr/bin/env python3
"""Diagnose output file for duplicates and data integrity."""
import pandas as pd

OUTPUT_FILE = "data/outputs/us_relevant_ai.csv"

print(f"Reading {OUTPUT_FILE}...")
df = pd.read_csv(OUTPUT_FILE, sep=";", engine='python', on_bad_lines='skip')

total_rows = len(df)
print(f"Total Rows: {total_rows}")

# Check for duplicate IDs
if 'id' in df.columns:
    unique_ids = df['id'].nunique()
    duplicate_count = total_rows - unique_ids
    print(f"Unique IDs: {unique_ids}")
    print(f"Duplicate IDs: {duplicate_count}")
    
    if duplicate_count > 0:
        # Show which IDs are duplicated
        dup_ids = df[df['id'].duplicated(keep=False)]['id'].value_counts().head(10)
        print(f"\nTop 10 duplicated IDs:\n{dup_ids}")
else:
    print("No 'id' column found!")

# Check ID range
try:
    numeric_ids = pd.to_numeric(df['id'], errors='coerce')
    print(f"\nID Range: {numeric_ids.min()} to {numeric_ids.max()}")
except:
    print("Could not analyze ID range")

# Check tier distribution
if 'desc_tier_llm' in df.columns:
    print(f"\nTier Distribution:")
    print(df['desc_tier_llm'].value_counts(dropna=False))

# Check for empty AI columns  
ai_cols = ['desc_tier_llm', 'desc_conf_llm', 'edulevel_llm', 'experience_min_llm']
print(f"\nEmpty/NaN counts for AI columns:")
for col in ai_cols:
    if col in df.columns:
        empty_count = df[col].isna().sum()
        print(f"  {col}: {empty_count} empty ({100*empty_count/total_rows:.1f}%)")
