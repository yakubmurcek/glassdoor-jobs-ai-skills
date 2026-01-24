
import pandas as pd

df = pd.read_csv('data/outputs/us_relevant_ai_stata.csv', sep=';')

print(f"Shape: {df.shape}")
print(f"Columns: {list(df.columns)}")

# Check for new clusters
cluster_cols = [c for c in df.columns if c.startswith('cluster_')]
print(f"\nCluster columns found: {len(cluster_cols)}")
print(cluster_cols[:5]) # sample

# Check education hybrid
print("\nEducation Hybrid Value Counts:")
print(df['education_hybrid'].value_counts(dropna=False))

# Check confidence
min_conf = df['desc_conf_llm'].min()
print(f"\nMin Confidence: {min_conf}")
