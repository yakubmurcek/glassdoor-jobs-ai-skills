
import os
import sys
import pandas as pd
import subprocess
import logging

# Ensure output dir
os.makedirs('data/outputs/verify_cli', exist_ok=True)

INPUT_CSV = 'data/outputs/us_relevant_ai_deduped_sample_20.csv'
FULL_CSV = 'data/outputs/verify_cli/full_output.csv'
OPTIMIZED_CSV = 'data/outputs/verify_cli/optimized_output.csv'

def run_command(cmd):
    print(f"Running: {cmd}")
    try:
        # Use uv run to match user environment
        subprocess.check_call(f"uv run {cmd}", shell=True)
    except subprocess.CalledProcessError as e:
        print(f"Error running command: {e}")
        sys.exit(1)

print("=== STARTING FINAL VERIFICATION ===")

# 1. Run Pipeline (Analyze)
# Using python -m to be sure we hit the code
cmd_analyze = f"python -m ai_skills.cli analyze --input-csv {INPUT_CSV} --output-csv {FULL_CSV} --skip-llm --no-progress"
run_command(cmd_analyze)

if not os.path.exists(FULL_CSV):
    print("FAILED: Full CSV not created.")
    sys.exit(1)

print(">> Pipeline output created.")
df_full = pd.read_csv(FULL_CSV, sep=';')
print(f"Full CSV Shape: {df_full.shape}")

# Verify Full CSV has Stata cols AND Text cols
stata_cols = [c for c in df_full.columns if c.startswith('cluster_')]
has_edu_hybrid = 'education_hybrid' in df_full.columns
has_text = 'job_desc_text' in df_full.columns
has_edu_orig = 'educations' in df_full.columns

if not (len(stata_cols) == 24 and has_edu_hybrid and has_text and has_edu_orig):
    print(f"FAILED: Full CSV missing columns. Clusters: {len(stata_cols)}, Hybrid: {has_edu_hybrid}, Text: {has_text}")
    sys.exit(1)
print("SUCCESS: Full CSV contains all required data (Stata vars + Text vars).")


# 2. Run Clean Stata CLI
cmd_clean = f"python -m ai_skills.cli clean-stata --input-csv {FULL_CSV} --output-csv {OPTIMIZED_CSV}"
run_command(cmd_clean)

if not os.path.exists(OPTIMIZED_CSV):
    print("FAILED: Optimized CSV not created.")
    sys.exit(1)

print(">> Optimized CSV created.")
df_opt = pd.read_csv(OPTIMIZED_CSV, sep=';')
print(f"Optimized CSV Shape: {df_opt.shape}")

# Verify Optimized CSV dropped Text but kept Stata
has_text_opt = 'job_desc_text' in df_opt.columns
has_edu_orig_opt = 'educations' in df_opt.columns
has_stata_opt = 'education_hybrid' in df_opt.columns and 'cluster_generative_ai' in df_opt.columns

if has_text_opt or has_edu_orig_opt:
    print(f"FAILED: Optimized CSV still has text columns. Text: {has_text_opt}, Edu: {has_edu_orig_opt}")
    sys.exit(1)

if not has_stata_opt:
    print("FAILED: Optimized CSV lost Stata columns.")
    sys.exit(1)

print("SUCCESS: Optimized CSV is clean and Stata-ready.")
print("=== VERIFICATION COMPLETE ===")
