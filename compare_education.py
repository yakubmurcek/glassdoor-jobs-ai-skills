
import pandas as pd
import numpy as np

# Load sample
df = pd.read_csv('data/outputs/us_relevant_ai_deduped_sample_20.csv', sep=';')

# Select education columns
cols = ['educations', 'edu_level_det', 'edulevel_llm']
print(df[cols].fillna("MISSING").to_string())

# Simple agreement check
def normalize(val):
    if pd.isna(val) or val == 'MISSING' or val == '-': return None
    return str(val).lower().replace("'s","").replace("'","")

df['norm_det'] = df['edu_level_det'].apply(normalize)
df['norm_llm'] = df['edulevel_llm'].apply(normalize)

df['agreement'] = df.apply(lambda x: x['norm_det'] == x['norm_llm'] if (x['norm_det'] and x['norm_llm']) else "N/A", axis=1)

print("\n--- Agreement Analysis ---")
print(df[['norm_det', 'norm_llm', 'agreement']].to_string())
