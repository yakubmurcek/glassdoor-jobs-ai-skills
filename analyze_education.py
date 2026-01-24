
import pandas as pd

# Load sample
df = pd.read_csv('data/outputs/us_relevant_ai_deduped_sample_20.csv', sep=';')

print("--- 'educations' Column Analysis ---")
# Show all non-null values to see structure
educations_vals = df['educations'].dropna().tolist()
print(f"Total non-null 'educations' rows: {len(educations_vals)}")
for i, val in enumerate(educations_vals):
    print(f"[{i}] {val}")

print("\n--- 'edulevel_llm' Column Analysis ---")
print(df['edulevel_llm'].value_counts(dropna=False))
