import pandas as pd
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent))
from ai_skills.job_title_normalizer import analyze_job_titles

input_file = 'data/outputs/us_relevant_ai_stata.csv'
target_file = 'data/outputs/us_relevant_ai_stata_job_title_frequency.csv'

print(f"Loading {input_file}...")
try:
    df = pd.read_csv(input_file, sep=';')
    print(f"Loaded {len(df)} rows.")
    
    if 'job_title' not in df.columns:
        print("Error: 'job_title' column not found!")
        sys.exit(1)
        
    print("Analyzing job titles...")
    job_titles = df['job_title'].dropna().astype(str).tolist()
    counts = analyze_job_titles(job_titles)
    
    # Load target file to compare
    target_df = pd.read_csv(target_file, sep=';')
    print(f"\nTarget file has {len(target_df)} rows.")
    
    # Compare top 5
    print("\nTop 5 Comparison:")
    print(f"{'Rank':<5} {'My Count':<10} {'Target Count':<12} {'Job Title'}")
    print("-" * 50)
    
    keys = list(counts.keys())
    for i in range(min(5, len(keys))):
        title = keys[i]
        my_count = counts[title]
        # Find in target
        target_row = target_df[target_df['job_title_normalized'] == title]
        if not target_row.empty:
            target_count = target_row.iloc[0]['count']
            match = "OK" if my_count == target_count else "MISMATCH"
        else:
            target_count = "N/A"
            match = "MISSING"
            
        print(f"{i+1:<5} {my_count:<10} {target_count:<12} {title} ({match})")

    # Check 'Software Engineer' and 'Software Developer' explicitly
    print("\nSpecific Checks:")
    for title in ["Software Engineer", "Software Developer"]:
        my_count = counts.get(title, 0)
        target_row = target_df[target_df['job_title_normalized'] == title]
        target_count = target_row.iloc[0]['count'] if not target_row.empty else "N/A"
        print(f"{title}: My={my_count}, Target={target_count}")

except Exception as e:
    print(f"Error: {e}")
