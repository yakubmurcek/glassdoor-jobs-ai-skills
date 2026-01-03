
import pandas as pd
import ast
from collections import Counter

try:
    df = pd.read_csv("data/outputs/us_relevant_30_reprocessed.csv", sep=";")
    
    all_hardskills = []
    all_softskills = []

    for index, row in df.iterrows():
        try:
            # skills are stored as simple comma-separated strings
            hards_raw = row.get('hardskills', '')
            match_raw = row.get('desc_hard_det', '') # Also check deterministic if useful, but hardskills is the final one
            
            if isinstance(hards_raw, str) and hards_raw.strip():
                hards = [s.strip() for s in hards_raw.split(',') if s.strip()]
                all_hardskills.extend(hards)
            
            softs_raw = row.get('softskills', '')
            if isinstance(softs_raw, str) and softs_raw.strip():
                softs = [s.strip() for s in softs_raw.split(',') if s.strip()]
                all_softskills.extend(softs)

        except Exception as e:
            print(f"Error parsing row {index}: {e}")

    # Unique sorted lists
    unique_hard = sorted(list(set(all_hardskills)))
    unique_soft = sorted(list(set(all_softskills)))
    
    print("=== ALL UNIQUE HARD SKILLS (Alphabetical) ===")
    for skill in unique_hard:
        print(f"'{skill}'")

    print("\n=== ALL UNIQUE SOFT SKILLS (Alphabetical) ===")
    for skill in unique_soft:
        print(f"'{skill}'")

except Exception as e:
    print(f"Failed to read CSV: {e}")
