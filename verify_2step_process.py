
from ai_skills.pipeline import JobAnalysisPipeline
import pandas as pd
import logging
import os
import sys

# Setup
logging.basicConfig(level=logging.ERROR) # Quiet mode
input_sample = 'data/outputs/us_relevant_ai_deduped_sample_20.csv'
pipeline_out = 'data/outputs/verify_step1_full.csv'
clean_out = 'data/outputs/verify_step2_clean.csv'

print("--- Step 1: Running Pipeline (Hydration Mode) ---")
pipeline = JobAnalysisPipeline()
df_full = pipeline.run(
    input_csv=input_sample,
    output_csv=pipeline_out,
    skip_llm=True
)

print(f"Step 1 Output: {df_full.shape}")
# Check columns kept
cols_kept = ['job_desc_text', 'educations', 'desc_rationale_llm', 'skill_cluster']
print("Checking preserved columns (should exist except skill_cluster):")
for c in cols_kept:
    exists = c in df_full.columns
    print(f"  {c}: {exists}")

# Check new columns
print("Checking new Stata columns:")
has_clusters = any(c.startswith('cluster_') for c in df_full.columns)
has_hybrid = 'education_hybrid' in df_full.columns
print(f"  Clusters created: {has_clusters}")
print(f"  Education Hybrid: {has_hybrid}")

print("\n--- Step 2: Running CLI Clean Command ---")
# Simulate CLI call
from ai_skills.cli import _handle_clean_stata
import argparse
from pathlib import Path

args = argparse.Namespace(
    input_csv=Path(pipeline_out),
    output_csv=Path(clean_out)
)

ret = _handle_clean_stata(args)
if ret != 0:
    print("CLI command failed!")
    sys.exit(1)

print("\n--- Verification of Optimized File ---")
df_clean = pd.read_csv(clean_out, sep=';')
print(f"Step 2 Output: {df_clean.shape}")

print("Checking dropped columns (should be False):")
cols_dropped = ['job_desc_text', 'educations', 'desc_rationale_llm']
for c in cols_dropped:
    exists = c in df_clean.columns
    print(f"  {c}: {exists}")

orig_size = os.path.getsize(pipeline_out)
new_size = os.path.getsize(clean_out)
print(f"\nSize Reduction: {orig_size} -> {new_size} bytes ({(1-new_size/orig_size)*100:.1f}%)")
