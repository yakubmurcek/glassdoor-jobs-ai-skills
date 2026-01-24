
import pandas as pd
import sys

OUTPUT_FILE = "data/outputs/us_relevant_ai_deduped.csv"
INPUT_FILE = "data/inputs/us_relevant.csv"  # Assuming this based on previous context, will check if exists

def count_rows():
    print(f"Analyzing {OUTPUT_FILE}...")
    try:
        # Read output
        df = pd.read_csv(OUTPUT_FILE, sep=";", engine='python', on_bad_lines='warn')
        
        total_rows = len(df)
        unique_ids = df['id'].nunique()
        
        # Check for valid processing
        # We assume 'desc_tier_llm' is populated if processed
        processed_df = df[df['desc_tier_llm'].notna()]
        valid_processed_count = len(processed_df)
        unique_processed_ids = processed_df['id'].nunique()
        
        print(f"{'='*40}")
        print(f"Raw Line/Row Count (Pandas): {total_rows}")
        print(f"Unique Job IDs:           {unique_ids}")
        print(f"Rows with LLM Output:      {valid_processed_count}")
        print(f"Unique IDs with Output:    {unique_processed_ids}")
        print(f"{'='*40}")
        
        if total_rows != unique_ids:
            print(f"WARNING: Found {total_rows - unique_ids} duplicate rows!")
            
        # Try to read input to give percentage
        try:
            print(f"\nChecking input file {INPUT_FILE} for context...")
            # Just count lines roughly for speed or read headers?
            # It might be large, let's just use a quick check or try to read columns='id'
            input_df = pd.read_csv(INPUT_FILE, sep=";", usecols=['id'], engine='python')
            total_input = len(input_df)
            print(f"Total Input Rows: {total_input}")
            print(f"Progress: {unique_processed_ids} / {total_input} ({unique_processed_ids/total_input:.1%})")
        except Exception as e:
            print(f"Could not read input file for comparison: {e}")

    except Exception as e:
        print(f"Error reading file: {e}")

if __name__ == "__main__":
    count_rows()
