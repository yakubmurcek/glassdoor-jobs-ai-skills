
import pandas as pd
import numpy as np

OUTPUT_FILE = "data/outputs/us_relevant_ai_deduped.csv"
SPLIT_INDEX = 13055  # ID where we switched to Nano (approx)

def compare_models():
    print(f"Reading {OUTPUT_FILE}...")
    df = pd.read_csv(OUTPUT_FILE, sep=";", engine='python', on_bad_lines='warn')
    
    # Split into Mini (old) and Nano (new)
    # We use index because IDs might not be sequential or might have gaps, 
    # but the file is appended.
    
    df_mini = df.iloc[:SPLIT_INDEX]
    df_nano = df.iloc[SPLIT_INDEX:]
    
    print(f"\nBatch Sizes: Mini={len(df_mini)}, Nano={len(df_nano)}")
    
    if len(df_nano) == 0:
        print("No Nano rows found? Check split index.")
        return

    # 1. Tier Distribution
    print("\n--- Tier Distribution (%) ---")
    dist_mini = df_mini['desc_tier_llm'].value_counts(normalize=True) * 100
    dist_nano = df_nano['desc_tier_llm'].value_counts(normalize=True) * 100
    
    # Create comparison table
    print(f"{'TIER':<20} | {'MINI':<10} | {'NANO':<10} | {'DIFF':<10}")
    print("-" * 60)
    all_tiers = set(dist_mini.index) | set(dist_nano.index)
    for tier in sorted(list(all_tiers)):
        m = dist_mini.get(tier, 0)
        n = dist_nano.get(tier, 0)
        diff = n - m
        print(f"{tier:<20} | {m:6.1f}%    | {n:6.1f}%    | {diff:+6.1f}%")

    # 2. Confidence Scores
    print("\n--- Average Confidence (by Tier) ---")
    print(f"{'TIER':<20} | {'MINI':<10} | {'NANO':<10}")
    print("-" * 50)
    for tier in sorted(list(all_tiers)):
        conf_mini = df_mini[df_mini['desc_tier_llm'] == tier]['desc_conf_llm'].mean()
        conf_nano = df_nano[df_nano['desc_tier_llm'] == tier]['desc_conf_llm'].mean()
        print(f"{tier:<20} | {conf_mini:.3f}      | {conf_nano:.3f}")

    # 3. Check for False Negatives in Nano (None tiers with AI keywords)
    print("\n--- Potential False Negatives Check (Nano) ---")
    print("Checking 'none' rows that definitely contain 'machine learning' or 'artificial intelligence' in text...")
    
    # Simple keyword check on job text
    ai_keywords = ['machine learning', 'artificial intelligence', 'deep learning']
    
    # Filter Nano None rows
    nano_none = df_nano[df_nano['desc_tier_llm'] == 'none'].copy()
    nano_none['job_desc_text'] = nano_none['job_desc_text'].fillna("").astype(str).str.lower()
    
    suspicious = []
    for idx, row in nano_none.iterrows():
        text = row['job_desc_text']
        found = [kw for kw in ai_keywords if kw in text]
        if found:
            suspicious.append((row['id'], found))
            
    print(f"Found {len(suspicious)} suspicious 'none' rows in Nano batch.")
    if suspicious:
        print("First 5 suspicious IDs and keywords:")
        for sid, kws in suspicious[:5]:
            print(f"ID {sid}: Found {kws}")

if __name__ == "__main__":
    compare_models()
