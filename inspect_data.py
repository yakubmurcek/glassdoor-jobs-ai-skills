
import pandas as pd
import json

# Load the dataframe
df = pd.read_csv('data/outputs/us_relevant_ai_deduped_sample_20.csv', sep=';')

# Columns of interest
cols = [
    'skills', 'skills_ai_det', 'desc_tier_llm', 'desc_conf_llm', 
    'skill_cluster', 'edu_level_det', 'edulevel_llm', 
    'experience_min_llm', 'ai_det_llm_match', 'educations'
]

print("--- Column Inspection ---")
for col in cols:
    if col in df.columns:
        print(f"\nCol: {col}")
        print(df[col].head(5).to_string())
        if col == 'educations':
            print("\nEducations Value Sample:")
            val = df[col].dropna().iloc[0] if not df[col].dropna().empty else "N/A"
            print(val)
    else:
        print(f"\nCol: {col} NOT FOUND")

print("\n--- Column Types ---")
print(df.dtypes)
