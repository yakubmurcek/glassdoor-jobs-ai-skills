import pandas as pd
import csv

input_file = 'data/outputs/us_relevant_ai_stata.csv'

# Try to detect separator
with open(input_file, 'r') as f:
    line = f.readline()
    if ';' in line:
        sep = ';'
    else:
        sep = ','
    print(f"Detected separator: '{sep}'")

try:
    df = pd.read_csv(input_file, sep=sep)
    print(f"Total rows: {len(df)}")
    
    if 'hardskills' not in df.columns:
        print("Column 'hardskills' not found!")
        print(f"Columns: {df.columns.tolist()}")
    else:
        target_skills = ['software engineering', 'python']
        print(f"\nVerifying counts for: {target_skills}")
        
        for target in target_skills:
            count = 0
            for skills_str in df['hardskills'].dropna():
                skills = [s.strip().lower() for s in str(skills_str).split(',') if s.strip()]
                if target in skills:
                    count += 1
            
            percentage = (count / len(df)) * 100
            print(f"Skill: '{target}' | Count: {count} | Percentage: {percentage:.2f}%")

except Exception as e:
    print(f"Error reading CSV: {e}")
