import pandas as pd
from pathlib import Path

def display_samples():
    base_dir = Path("data/outputs")
    us_file = base_dir / "us_relevant_ai_stata.csv"
    de_file = base_dir / "de/de_relevant_ai_stata.csv"
    in_file = base_dir / "in_relevant_ai_stata_optimized.csv"

    files = {"US": us_file, "DE": de_file, "IN": in_file}

    # Columns of interest to verify data symmetry and quality
    sample_columns = [
        "job_title", 
        "company", 
        "location", 
        "region", 
        "job_family", 
        "sector_nace", 
        "core_ai", 
        "ai_tier",
        "cluster_data_science__ml", 
        "cluster_backend_development"
    ]

    for name, path in files.items():
        if not path.exists():
            print(f"Error: File for {name} missing at {path}")
            continue

        try:
            # Load the dataset
            df = pd.read_csv(path, sep=';', low_memory=False)
            
            # Ensure the columns exist
            available_cols = [c for c in sample_columns if c in df.columns]
            
            # Take a random sample of 3 rows
            sample = df[available_cols].sample(n=3, random_state=42)
            
            print(f"\n{'='*80}")
            print(f"{name} DATASET SAMPLE (3 Random Rows)")
            print(f"{'='*80}")
            
            # Print row by row for readability
            for idx, row in sample.iterrows():
                print(f"--- Row {idx} ---")
                for col in available_cols:
                    val = row[col]
                    print(f"{col:>25}: {val}")
                    
        except Exception as e:
            print(f"Error processing {name} dataset: {e}")

if __name__ == "__main__":
    display_samples()
