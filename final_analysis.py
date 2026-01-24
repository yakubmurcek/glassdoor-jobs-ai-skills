
import pandas as pd
import sys
import matplotlib.pyplot as plt
import seaborn as sns
from pathlib import Path

# Paths
INPUT_CSV = "data/outputs/us_relevant_ai_deduped.csv"
OUTPUT_REPORT = "data/outputs/final_analysis_report.md"
OUTPUT_IMG_TIERS = "data/outputs/tier_distribution.png"

def generate_analysis():
    print(f"Loading {INPUT_CSV}...")
    try:
        df = pd.read_csv(INPUT_CSV, sep=";", engine='python', on_bad_lines='warn')
    except Exception as e:
        print(f"Error reading CSV: {e}")
        return

    total_jobs = len(df)
    
    # 1. Tier Analysis
    tier_counts = df['desc_tier_llm'].value_counts()
    tier_pct = df['desc_tier_llm'].value_counts(normalize=True) * 100
    
    # 2. "Real AI" Analysis
    real_ai_count = df['is_real_ai'].sum()
    real_ai_pct = (real_ai_count / total_jobs) * 100
    
    # 3. Process Skills
    all_skills = []
    ai_skills_list = []
    
    for _, row in df.iterrows():
        # Hard skills
        h_skills = str(row.get('hardskills', '')).split(', ')
        all_skills.extend([s for s in h_skills if s and s.lower() != 'nan'])
        
        # AI skills (from desc_ai_llm)
        a_skills = str(row.get('desc_ai_llm', '')).split(', ')
        ai_skills_list.extend([s for s in a_skills if s and s.lower() != 'nan'])

    # Top Skills
    from collections import Counter
    top_skills = Counter(all_skills).most_common(20)
    top_ai_skills = Counter(ai_skills_list).most_common(20)
    
    # 4. Generate Plot
    plt.figure(figsize=(10, 6))
    sns.countplot(y=df['desc_tier_llm'], order=tier_counts.index)
    plt.title('Distribution of AI Job Tiers')
    plt.xlabel('Number of Jobs')
    plt.ylabel('Tier')
    plt.tight_layout()
    plt.savefig(OUTPUT_IMG_TIERS)
    print(f"Saved tier plot to {OUTPUT_IMG_TIERS}")

    # 5. Write Markdown Report
    print(f"Writing report to {OUTPUT_REPORT}...")
    with open(OUTPUT_REPORT, 'w') as f:
        f.write("# Final AI Skills Analysis Report\n\n")
        f.write(f"**Total Processed Jobs**: {total_jobs:,}\n")
        f.write(f"**Identified 'Real AI' Roles**: {real_ai_count:,} ({real_ai_pct:.1f}%)\n\n")
        
        f.write("## 1. AI Tier Distribution\n")
        f.write("| Tier | Count | Percentage |\n")
        f.write("|------|-------|------------|\n")
        for tier, count in tier_counts.items():
            pct = tier_pct[tier]
            f.write(f"| {tier} | {count:,} | {pct:.1f}% |\n")
        
        f.write("\n![Tier Distribution](tier_distribution.png)\n\n")
        
        f.write("## 2. Top 20 Hard Skills (Overall)\n")
        f.write("| Rank | Skill | Frequency |\n")
        f.write("|------|-------|-----------|\n")
        for i, (skill, count) in enumerate(top_skills, 1):
             f.write(f"| {i} | {skill} | {count:,} |\n")
             
        f.write("\n## 3. Top 20 AI Specific Skills\n")
        f.write("*Extracted specifically from AI context*\n")
        f.write("| Rank | Skill | Frequency |\n")
        f.write("|------|-------|-----------|\n")
        for i, (skill, count) in enumerate(top_ai_skills, 1):
             f.write(f"| {i} | {skill} | {count:,} |\n")

    print("Analysis Complete.")

if __name__ == "__main__":
    generate_analysis()
