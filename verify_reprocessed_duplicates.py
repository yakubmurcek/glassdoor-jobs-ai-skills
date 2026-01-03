
import pandas as pd
import sys

try:
    df = pd.read_csv("data/outputs/us_relevant_30_reprocessed.csv", sep=";")
    print(f"Loaded {len(df)} rows.")

    issues_found = 0
    
    for i, row in df.iterrows():
        raw = str(row['hardskills'])
        if not raw or raw == "nan": continue
        skills = [s.strip().lower() for s in raw.split(",")]
        
        # Check specific known duplicates
        if "sql server" in skills and "sql" in skills:
            # Only an issue if the text DOESN'T contain standalone SQL
            text = str(row['job_desc_text']).lower()
            # Crude check: if "sql" appears only as part of "sql server"
            count_sql = text.count("sql")
            count_sqls = text.count("sql server")
            if count_sql == count_sqls:
                print(f"Row {i} FAIL: 'sql' and 'sql server' extracted, but text likely only has 'sql server'.")
                issues_found += 1
            else:
                print(f"Row {i} INFO: 'sql' and 'sql server' both present (likely valid).")

        # Check 1: Overlap (SQL vs SQL Server) - Existing check
        # This is technically allowed now if they are physically distinct, 
        # but we want to ensure we didn't break anything.
        # The original check above handles the "FAIL" condition.

        # Check 2: HTML vs HTML5 (Should be normalized to same, or at least one)
        if "html" in skills and "html5" in skills:
             print(f"Row {i} FAIL: Found both 'html' and 'html5': {skills}")
             issues_found += 1

        # Check 3: Scalability in Hard Skills
        if "scalability" in skills:
            print(f"Row {i} FAIL: Found 'scalability' in hard skills: {skills}")
            issues_found += 1

    if issues_found == 0:
        print("SUCCESS: No obvious invalid duplicates found.")
    else:
        print(f"FAIL: Found {issues_found} potential duplicates.")

except Exception as e:
    print(f"Error: {e}")
    sys.exit(1)
