
import pandas as pd
import random
import sys

# Path to the output file
OUTPUT_FILE = "data/outputs/us_relevant_ai_deduped.csv"

def inspect_sample():
    print(f"Reading {OUTPUT_FILE}...")
    
    # We know the file has > 500k rows. 
    # Let's verify rows that were definitely processed in THIS run.
    # The run started at 131,219. So any index > 150,000 is safe to check for "new" data.
    
    # To be efficient, we can't easily "skip" rows with read_csv without reading them, 
    # but with python engine it's slow. 
    # However, 500k rows is only ~500MB? The file size was 16MB -> 17MB earlier?
    # Wait.
    # Step 137: wc -l said 527,707 lines.
    # Step 95: ls -l said 219126 BYTES (200KB) - that was the CORRUPTED version.
    # Step 136 ls -l said 527707 lines? NO.
    # Step 137 output was `527707 data/outputs/us_relevant_ai.csv` lines.
    # Step 141 output was header.
    
    # 527k lines is a lot for a 17MB file? 
    # 17,000,000 / 500,000 = 34 bytes per line?
    # A CSV row with job description is usually 500-2000 bytes.
    # UNLESS description is NOT in the output?
    # Let's check columns. `job_desc_text` IS in the columns.
    # Maybe the text is short or truncated?
    # Or maybe `wc -l` count includes newlines inside the quoted csv fields?
    # Yes, `wc -l` counts newlines. A job description can have 50 newlines.
    # So 527k "lines" might be only 10k "rows"?
    # Initial status: 131,219 "rows" (id count) or lines?
    # The resume log said "Found 131219 rows". That means true records.
    # If 131k records fit in 16MB, that's 121 bytes/record.
    # That is VERY small for job descriptions.
    # Ah, `job_desc_text` might be empty or not saved? 
    # Or `load_input_data` removes it?
    # Wait, `ai_skills/config.py` says `MAX_JOB_DESC_LENGTH=6500`.
    
    # Let's trust pandas to read it correctly.
    # We will read the last 100 rows using tail logic (efficient).
    
    try:
        # Read last 1000 lines (could be partial rows) and parse what we can
        # Actually proper way:
        df = pd.read_csv(OUTPUT_FILE, sep=";", engine='python', skiprows=lambda x: x > 0 and x < 132000 and random.random() > 0.001)
        # That's too slow to decide for every row.
        
        # better: just read the whole thing? It's not that big if it's 500k lines.
        # But if it's 1.5M rows input...
        
        # Let's use skip rows correctly.
        # If we want to sample from row 200,000 to end.
        # We don't know exactly how many rows there are now (wc -l is unreliable with multiline CSV).
        
        # Let's just read the last 50 processed results.
        # We can use `tail` command to get bytes, but pandas needs valid CSV.
        
        # Let's read the *first* 132,000 to skip them? No.
        
        # Read the whole file (it is likely small enough, ~20-50MB max)
        print("Reading entire file into DataFrame...")
        df = pd.read_csv(OUTPUT_FILE, sep=";", engine='python', on_bad_lines='warn')
        
        total_rows = len(df)
        print(f"Total Rows Found: {total_rows}")
        
        if total_rows == 0:
            print("File is empty.")
            return

        # Check max ID to confirm if we are strictly appending or restarting
        try:
            # Assuming 'id' is numeric or convertible
            max_id = pd.to_numeric(df['id'], errors='coerce').max()
            print(f"Max ID in output: {max_id}")
        except Exception as e:
            print(f"Could not calculate max ID: {e}")

        # Sample 20 random rows from validity
        sample_size = 20
        print(f"Sampling {sample_size} random rows from the new data...")
        
        # We assume new data is at the end. Let's sample from the last run's batch.
        # Current size 6515. Started at ~100? No, started at 131k. 
        # Wait, previous run showed 6515 rows TOTAL in the dataframe?
        # Ah, Step 162 output: "Total Rows Found: 6515".
        # BUT Step 144 said "Row Count: 527,707".
        # Why did Step 162 say 6515?
        # Maybe `pd.read_csv` with `on_bad_lines='skip'` skipped almost everything due to parsing errors?
        # OR maybe I misread the output. 
        # "Total Rows Found: 6515"
        # If the file has 500k lines, pandas should see 500k rows.
        # UNLESS the file is full of bad lines that were skipped.
        # This is CRITICAL to investigate.
        
        # Let's inspect the file structure more carefully in the script.
        # We will read without skipping bad lines to see validation errors if needed, 
        # but for now let's sample what pandas COULD read.
        
        # Sample from the last 1200 rows (processed by GPT-5 Nano)
        tail_df = df.tail(1200)
        sample = tail_df.sample(min(sample_size, len(tail_df)))
        
        print(f"\n{'='*100}")
        print(f"{'ID':<6} | {'TIER':<15} | {'CONF':<4} | {'EDU':<10} | {'EXP':<5} | {'IS_REAL':<7} | {'SKILLS (AI)':<30}")
        print(f"{'='*100}")
        
        for idx, row in sample.iterrows():
            tier = str(row.get('desc_tier_llm', 'N/A'))
            conf = str(row.get('desc_conf_llm', 'N/A'))
            edu = str(row.get('edulevel_llm', '-'))
            exp = str(row.get('experience_min_llm', '-'))
            is_real = str(row.get('is_real_ai', 'N/A'))
            ai_skills = str(row.get('desc_ai_llm', ''))
            rationale = str(row.get('desc_rationale_llm', ''))
            
            # Truncate for table
            ai_skills_short = (ai_skills[:27] + '...') if len(ai_skills) > 27 else ai_skills
            
            print(f"{str(row.get('id', 'N/A')):<6} | {tier:<15} | {conf:<4} | {edu:<10} | {exp:<5} | {is_real:<7} | {ai_skills_short:<30}")
            
            # Print rationale if it exists or if confidence looks low
            if rationale and rationale.lower() != 'nan':
                 print(f"      > Rationale: {rationale}")
        
        print(f"{'='*100}")
        
        # Analyze distribution
        print("\nDistribution Analysis:")
        print(df['desc_tier_llm'].value_counts(dropna=False))


    except Exception as e:
        print(f"Error: {e}")

if __name__ == "__main__":
    inspect_sample()
