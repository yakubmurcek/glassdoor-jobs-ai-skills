#!/usr/bin/env python3
"""
Analyze job title frequency from the Glassdoor jobs dataset.
Normalizes job titles using ai_skills.job_title_normalizer.
"""
import pandas as pd
import argparse
import sys
from pathlib import Path

# Add project root to path for imports
sys.path.insert(0, str(Path(__file__).parent))

from ai_skills.job_title_normalizer import analyze_job_titles

def analyze_titles_main(input_file: str):
    # Load the CSV file
    print(f"Loading {input_file}...")
    try:
        # Auto-detect separator if possible, but default to semicolon for project consistency
        # pipeline output is usually semicolon
        df = pd.read_csv(input_file, sep=';', dtype=str)
    except Exception:
        # Fallback to comma if semicolon fails
        df = pd.read_csv(input_file, sep=',', dtype=str)
        
    print(f"Loaded {len(df)} job postings\n")
    
    if 'job_title' not in df.columns:
        print("Error: 'job_title' column not found!")
        return

    # Extract job titles
    titles = df['job_title'].dropna().tolist()
    
    # Analyze (returns dict of Title -> Count)
    counts = analyze_job_titles(titles)
    
    # Convert to DataFrame
    data = []
    total_listings = len(df)
    
    for title, count in counts.items():
        pct = round((count / total_listings) * 100, 2)
        data.append({
            'job_title_normalized': title,
            'count': count,
            'percentage': pct
        })
        
    result_df = pd.DataFrame(data)
    
    # Save to CSV
    # Matches format: job_title_normalized;count;percentage
    output_file = input_file.replace('.csv', '_job_title_frequency.csv')
    result_df.to_csv(output_file, sep=';', index=False)
    
    print("-" * 60)
    print(f"saved to: {output_file}")
    print("-" * 60)
    print(result_df.head(20).to_string(index=False))
    print("-" * 60)

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Analyze job title frequency")
    parser.add_argument("--input", type=str, default="data/outputs/us_relevant_ai_stata.csv",
                        help="Input CSV file path")
    args = parser.parse_args()
    
    analyze_titles_main(args.input)
