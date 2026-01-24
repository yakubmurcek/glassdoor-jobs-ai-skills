
import pandas as pd

df = pd.read_csv('data/outputs/us_relevant_ai_deduped_sample_20.csv', sep=';')

print("Unique edulevel_llm:")
print(df['edulevel_llm'].unique())

print("\nUnique desc_tier_llm:")
print(df['desc_tier_llm'].unique())
