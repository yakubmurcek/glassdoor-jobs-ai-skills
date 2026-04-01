import pandas as pd
from pathlib import Path

def check_compatibility():
    base_dir = Path("data/outputs")
    us_file = base_dir / "us_relevant_ai_stata.csv"
    de_file = base_dir / "de/de_relevant_ai_stata.csv"
    in_file = base_dir / "in_relevant_ai_stata_optimized.csv"

    files = {"US": us_file, "DE": de_file, "IN": in_file}
    
    columns_map = {}
    
    for name, path in files.items():
        if not path.exists():
            print(f"Error: File for {name} missing at {path}")
            return
        
        # We only need headers, so we read just 0 rows (which gets columns)
        try:
            df = pd.read_csv(path, sep=';', nrows=0)
            columns_map[name] = set(df.columns)
            print(f"{name} dataset: {len(df.columns)} columns")
        except Exception as e:
            print(f"Error reading {name} dataset: {e}")
            return
            
    us_cols = columns_map["US"]
    de_cols = columns_map["DE"]
    in_cols = columns_map["IN"]

    print("\n--- Compatibility Check ---")
    
    # Compare with US (base)
    in_missing = us_cols - in_cols
    in_extra = in_cols - us_cols
    
    if not in_missing and not in_extra:
        print("✅ IN dataset is exactly compatible with US dataset (identical columns).")
    else:
        if in_missing:
            print(f"❌ IN dataset is missing columns found in US: {sorted(in_missing)}")
        if in_extra:
            print(f"❌ IN dataset has extra columns not in US: {sorted(in_extra)}")

    # Compare DE with US (base)
    de_missing = us_cols - de_cols
    de_extra = de_cols - us_cols

    if not de_missing and not de_extra:
        print("✅ DE dataset is exactly compatible with US dataset (identical columns).")
    else:
        if de_missing:
            print(f"❌ DE dataset is missing columns found in US: {sorted(de_missing)}")
        if de_extra:
            print(f"❌ DE dataset has extra columns not in US: {sorted(de_extra)}")


if __name__ == "__main__":
    check_compatibility()
