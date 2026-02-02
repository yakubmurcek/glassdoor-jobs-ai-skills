#!/usr/bin/env python3
"""
Analyze hard skills frequency from the Glassdoor jobs dataset.
Uses existing cluster columns from the pipeline with official category names.
"""
import pandas as pd
from collections import Counter
import argparse
import re
import sys
from pathlib import Path

# Add project root to path for imports
sys.path.insert(0, str(Path(__file__).parent))

from ai_skills.skills_dictionary import SKILL_TO_FAMILY


def get_cluster_label_mapping() -> dict[str, str]:
    """
    Build mapping from cluster column names to official family labels.
    
    Returns:
        Dict mapping e.g. 'cluster_frontend_development' -> 'Frontend Development'
    """
    # Get unique family names from the dictionary
    families = set(SKILL_TO_FAMILY.values())
    
    # Build reverse mapping: column_name -> official label
    mapping = {}
    for family in families:
        # This matches the logic in pipeline.py line 329
        safe_name = re.sub(r'[^a-zA-Z0-9]', '_', family.lower())
        col_name = f"cluster_{safe_name}"
        mapping[col_name] = family
    
    return mapping


def analyze_hard_skills(input_file: str):
    """
    1) Vygeneruj tabulku četností pro jednotlivé hard skills
    2) Použij existující clustery z pipeline pro seskupení
    """
    # Load the CSV file
    df = pd.read_csv(input_file, sep=';')
    
    print(f"Loaded {len(df)} job postings\n")
    
    # Get official cluster label mapping
    cluster_label_map = get_cluster_label_mapping()
    
    # =========================================================================
    # ČÁST 1: TABULKA ČETNOSTÍ HARD SKILLS
    # =========================================================================
    
    # Extract hard skills from the 'hardskills' column
    all_hard_skills = []
    
    for skills_str in df['hardskills'].dropna():
        if pd.notna(skills_str) and skills_str.strip():
            skills = [s.strip().lower() for s in str(skills_str).split(',') if s.strip()]
            all_hard_skills.extend(skills)
    
    # Count frequencies
    skill_counts = Counter(all_hard_skills)
    
    # Convert to DataFrame
    skill_freq_df = pd.DataFrame(
        skill_counts.most_common(),
        columns=['Hard Skill', 'Frequency']
    )
    skill_freq_df['Percentage'] = (skill_freq_df['Frequency'] / len(df) * 100).round(2)
    
    # Print frequency table
    print("=" * 70)
    print("TABULKA ČETNOSTÍ HARD SKILLS")
    print("=" * 70)
    print(f"\nCelkem unikátních hard skills: {len(skill_counts)}")
    print(f"Celkem zmínek hard skills: {len(all_hard_skills)}")
    print()
    
    # Top 50 skills
    print("Top 50 nejčastějších hard skills:")
    print("-" * 70)
    print(f"{'Skill':<40} {'Počet':>10} {'%':>10}")
    print("-" * 70)
    for idx, row in skill_freq_df.head(50).iterrows():
        print(f"{row['Hard Skill']:<40} {row['Frequency']:>10} {row['Percentage']:>9.1f}%")
    
    # Save full frequency table to CSV
    output_file = input_file.replace('.csv', '_hardskills_frequency.csv')
    skill_freq_df.to_csv(output_file, index=False)
    print(f"\nPlná tabulka uložena do: {output_file}")
    
    # =========================================================================
    # ČÁST 2: SESKUPENÍ PODLE EXISTUJÍCÍCH CLUSTERŮ Z PIPELINE
    # =========================================================================
    
    print("\n" + "=" * 70)
    print("SESKUPENÍ PODLE CLUSTERŮ (z pipeline)")
    print("=" * 70)
    
    # Get all cluster columns
    cluster_cols = [col for col in df.columns if col.startswith('cluster_')]
    
    # Calculate frequency of each cluster (how many job postings)
    cluster_freq = df[cluster_cols].sum().sort_values(ascending=False)
    
    print(f"\nCelkem clusterů: {len(cluster_cols)}")
    print("\n📊 ČETNOST CLUSTERŮ (počet job postings v každém clusteru):")
    print("-" * 70)
    print(f"{'Cluster':<45} {'Počet':>10} {'%':>10}")
    print("-" * 70)
    
    for cluster_name, count in cluster_freq.items():
        # Use official label from mapping
        label = cluster_label_map.get(cluster_name, cluster_name)
        pct = count / len(df) * 100
        print(f"{label:<45} {int(count):>10} {pct:>9.1f}%")
    
    # Save cluster frequencies with official labels
    cluster_freq_df = pd.DataFrame({
        'Cluster': cluster_freq.index,
        'Cluster_Label': [cluster_label_map.get(c, c) for c in cluster_freq.index],
        'Frequency': cluster_freq.values,
        'Percentage': (cluster_freq.values / len(df) * 100).round(2)
    })
    cluster_output = input_file.replace('.csv', '_cluster_frequency.csv')
    cluster_freq_df.to_csv(cluster_output, index=False)
    print(f"\nČetnost clusterů uložena do: {cluster_output}")
    
    # =========================================================================
    # ČÁST 3: TOP SKILLS PRO KAŽDÝ CLUSTER
    # =========================================================================
    
    print("\n" + "=" * 70)
    print("TOP 10 HARD SKILLS PRO KAŽDÝ CLUSTER")
    print("=" * 70)
    
    cluster_skills_data = []
    
    for cluster_col in cluster_freq.index:
        # Filter jobs that belong to this cluster
        cluster_jobs = df[df[cluster_col] == 1]
        
        if len(cluster_jobs) == 0:
            continue
        
        # Extract hard skills from these jobs
        cluster_hard_skills = []
        for skills_str in cluster_jobs['hardskills'].dropna():
            if pd.notna(skills_str) and skills_str.strip():
                skills = [s.strip().lower() for s in str(skills_str).split(',') if s.strip()]
                cluster_hard_skills.extend(skills)
        
        # Count frequencies
        cluster_skill_counts = Counter(cluster_hard_skills)
        top_skills = cluster_skill_counts.most_common(10)
        
        # Use official label
        label = cluster_label_map.get(cluster_col, cluster_col)
        
        print(f"\n📁 {label.upper()} ({len(cluster_jobs)} job postings)")
        print("-" * 50)
        for skill, count in top_skills:
            pct = count / len(cluster_jobs) * 100
            print(f"   {skill:<35} {count:>6} ({pct:>5.1f}%)")
        
        # Store for export
        for rank, (skill, count) in enumerate(top_skills, 1):
            cluster_skills_data.append({
                'Cluster': cluster_col,
                'Cluster_Label': label,
                'Rank': rank,
                'Skill': skill,
                'Count': count,
                'Percentage': round(count / len(cluster_jobs) * 100, 2)
            })
    
    # Save detailed cluster-skill mapping
    cluster_skills_df = pd.DataFrame(cluster_skills_data)
    cluster_skills_output = input_file.replace('.csv', '_cluster_skills.csv')
    cluster_skills_df.to_csv(cluster_skills_output, index=False)
    print(f"\n\nDetaily skills pro každý cluster uloženy do: {cluster_skills_output}")
    
    return skill_freq_df, cluster_freq_df


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Analyze hard skills frequency")
    parser.add_argument("--input", type=str, default="data/outputs/us_relevant_ai_stata.csv",
                        help="Input CSV file path")
    args = parser.parse_args()
    
    analyze_hard_skills(args.input)
