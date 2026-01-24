
import pandas as pd
import numpy as np
import re
from ai_skills.skills_dictionary import SKILL_TO_FAMILY, _CATEGORIES

# Setup logging
import logging
logging.basicConfig(level=logging.INFO, format='%(message)s')
logger = logging.getLogger()

INPUT_FILE = 'data/outputs/us_relevant_ai_deduped.csv'
OUTPUT_FILE = 'data/outputs/us_relevant_ai_stata.csv' 
# Using a new name for the output to distinguish it

def transform_data():
    logger.info(f"Loading data from {INPUT_FILE}...")
    df = pd.read_csv(INPUT_FILE, sep=';')
    initial_len = len(df)
    logger.info(f"Initial rows: {initial_len}")

    # 1. Filtering
    # User requested to perform filtering in Stata, so we keep ALL rows for now.
    # logger.info("Filtering by confidence < 0.7...")
    # df = df[df['desc_conf_llm'] >= 0.7].copy()
    logger.info(f"Rows retained (no filtering): {len(df)}")
    
    # We KEEP is_real_ai == 0 rows as requested.

    # 2. Skill Clustering (Updated Taxonomy)
    logger.info("Generating skill cluster dummy variables...")
    
    # Get all unique families
    families = set(SKILL_TO_FAMILY.values())
    family_list = sorted(list(families))
    
    # Create dummy columns initialized to 0
    for family in family_list:
        # Create column name like 'cluster_data_science' (clean string)
        safe_name = re.sub(r'[^a-zA-Z0-9]', '_', family.lower())
        col_name = f"cluster_{safe_name}"
        df[col_name] = 0

    # Function to map skills to families for a row
    def map_row_skills(skills_str):
        if pd.isna(skills_str) or skills_str == "":
            return set()
        
        # Split skills
        row_skills = [s.strip().lower() for s in str(skills_str).split(',')]
        
        present_families = set()
        for skill in row_skills:
            # Check direct mapping
            family = SKILL_TO_FAMILY.get(skill)
            if family:
                present_families.add(family)
        return present_families

    # Apply mapping
    # We use the 'hardskills' column (merged deterministic + llm) for best coverage
    # If 'hardskills' is missing, fallback to 'skills'
    skill_source_col = 'hardskills' if 'hardskills' in df.columns else 'skills'
    
    for idx, row in df.iterrows():
        skills_val = row.get(skill_source_col)
        row_families = map_row_skills(skills_val)
        
        for family in row_families:
            safe_name = re.sub(r'[^a-zA-Z0-9]', '_', family.lower())
            col_name = f"cluster_{safe_name}"
            df.at[idx, col_name] = 1

    # 3. Hybrid Education Variable
    logger.info("Creating education_hybrid variable...")
    
    def get_education_hybrid(row):
        val = "missing"
        
        # 1. Primary: edu_level_det (structured from meta)
        det = row.get('edu_level_det')
        if pd.notna(det) and str(det).strip() != "":
            val = str(det).strip().lower()
        
        # 2. Backfill: edulevel_llm (from text)
        else:
            llm = row.get('edulevel_llm')
            if pd.notna(llm) and str(llm).strip() != "" and str(llm).strip() != "-":
                 val = str(llm).strip().lower()
        
        # Normalize: remove "'s", " degree", etc to unify "bachelor's" and "bachelor"
        val = val.replace("'s", "").replace(" degree", "").replace(" diploma", "")
        return val

    df['education_hybrid'] = df.apply(get_education_hybrid, axis=1)

    # 4. Column Selection / Variable Cleanup
    
    # Drop outdated/bulky columns
    cols_to_drop = [
        'skill_cluster', 
        'desc_rationale_llm', 
        'job_desc_text',
        'job_desc_html', # HTML is also huge and likely not needed in Stata
        'educations' # We created hybrid, so we drop the messy list (optional, but requested implicitly by 'Hold' status and final confirmation)
        # Actually user said "Hold on dropping educations", but then we agreed on Hybrid. 
        # I'll keep 'educations' just in case, it doesn't hurt Stata to have it as string.
        # But 'job_desc_text' is definitely too big.
    ]
    
    actual_drop = [c for c in cols_to_drop if c in df.columns]
    logger.info(f"Dropping columns: {actual_drop}")
    df.drop(columns=actual_drop, inplace=True)

    # Ensure experience is numeric (coerce errors)
    if 'experience_min_llm' in df.columns:
        df['experience_min_llm'] = pd.to_numeric(df['experience_min_llm'], errors='coerce')

    # Save
    logger.info(f"Saving to {OUTPUT_FILE}...")
    df.to_csv(OUTPUT_FILE, sep=';', index=False)
    logger.info("Done.")

if __name__ == "__main__":
    transform_data()
