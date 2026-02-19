"""
Analýza spolehlivosti skills_cleaned:
- Kolik jobů má tier != none, ale skills_cleaned je prázdný? (filtrováno pryč has_ai_flag)
- Kolik jobů má tier == none, ale skills_cleaned je neprázdný? (potenciální false positives)
- Co je v skills_cleaned u none-tier jobů? Jsou to skutečné AI skills nebo šum?
- Jaký dopad má skills_cleaned filtr na has_ai_flag vs. samotný tier?
"""
import pandas as pd
import re

# Load data
df = pd.read_csv("data/outputs/us_relevant_ai_stata.csv", sep=";")
print(f"Total rows: {len(df)}")
print(f"Columns used: desc_tier_llm, desc_conf_llm, skills_ai_det, desc_ai_llm")
print()

# Apply same confidence filter as Stata
df = df[df["desc_conf_llm"] >= 0.7].copy()
print(f"After confidence >= 0.7 filter: {len(df)}")
print()

# Replicate Stata logic for skills_cleaned
df["skills_ai_det"] = df["skills_ai_det"].fillna("")
df["desc_ai_llm"] = df["desc_ai_llm"].fillna("")
df["desc_tier_llm"] = df["desc_tier_llm"].fillna("missing")

df["skills_combined"] = (df["skills_ai_det"] + " " + df["desc_ai_llm"]).str.lower()

# Remove buzzwords (same regex as Stata)
df["skills_no_buzz"] = df["skills_combined"].apply(
    lambda x: re.sub(r"(?i)\b(ai|ml|artificial intelligence|machine learning|genai)\b", "", x)
)

# Remove punctuation (commas, spaces, semicolons)
df["skills_cleaned"] = (
    df["skills_no_buzz"]
    .str.replace(",", "", regex=False)
    .str.replace(" ", "", regex=False)
    .str.replace(";", "", regex=False)
)

# has_ai_flag: tier != none AND skills_cleaned non-empty (length > 1)
df["tier_is_ai"] = (df["desc_tier_llm"] != "none") & (df["desc_tier_llm"] != "missing")
df["skills_nonempty"] = df["skills_cleaned"].str.len() > 1
df["has_ai_flag"] = df["tier_is_ai"] & df["skills_nonempty"]

print("=" * 70)
print("1. CROSS-TABULATION: tier_is_ai × skills_nonempty")
print("=" * 70)
ct = pd.crosstab(df["tier_is_ai"], df["skills_nonempty"], margins=True,
                 rownames=["Tier!=none"], colnames=["Skills non-empty"])
print(ct)
print()

# Key groups
group_A = df[df["tier_is_ai"] & df["skills_nonempty"]]      # Both say AI → has_ai=1
group_B = df[df["tier_is_ai"] & ~df["skills_nonempty"]]     # Tier says AI, but NO skills → FILTERED OUT
group_C = df[~df["tier_is_ai"] & df["skills_nonempty"]]     # Tier says none, but skills found → ignored
group_D = df[~df["tier_is_ai"] & ~df["skills_nonempty"]]    # Both say no AI

print("=" * 70)
print("2. IMPACT OF skills_cleaned FILTER")
print("=" * 70)
print(f"  A) Tier=AI  & Skills=YES → has_ai=1:  {len(group_A):>5}  (agreement)")
print(f"  B) Tier=AI  & Skills=NO  → has_ai=0:  {len(group_B):>5}  ← skills filter REMOVES these")
print(f"  C) Tier=none & Skills=YES → has_ai=0: {len(group_C):>5}  ← tier already filters, skills irrelevant")
print(f"  D) Tier=none & Skills=NO  → has_ai=0: {len(group_D):>5}  (agreement)")
print()
print(f"  Without skills filter (tier only): {len(group_A) + len(group_B)} AI jobs")
print(f"  With skills filter (has_ai_flag):  {len(group_A)} AI jobs")
print(f"  Difference:                        {len(group_B)} jobs removed by skills filter")
print(f"  Removal rate:                      {len(group_B) / (len(group_A) + len(group_B)) * 100:.1f}%")
print()

print("=" * 70)
print("3. GROUP B DETAIL: Tier says AI, but skills_cleaned is EMPTY")
print("   (Are these legitimate AI jobs being wrongly excluded?)")
print("=" * 70)
if len(group_B) > 0:
    print(f"\n  Tier breakdown of group B:")
    print(group_B["desc_tier_llm"].value_counts().to_string())
    print(f"\n  Their raw skills_combined (before buzz removal):")
    raw_skills = group_B["skills_combined"].value_counts().head(20)
    for skill, count in raw_skills.items():
        print(f"    [{count:>3}x] '{skill.strip()}'")
    print(f"\n  Sample job titles from group B (first 15):")
    for _, row in group_B.head(15).iterrows():
        print(f"    Tier={row['desc_tier_llm']:<16} | '{row['job_title']}'")
        print(f"      skills_ai_det='{row['skills_ai_det']}' | desc_ai_llm='{row['desc_ai_llm']}'")
else:
    print("  (No jobs in this group)")
print()

print("=" * 70)
print("4. GROUP C DETAIL: Tier says NONE, but skills_cleaned is NON-EMPTY")
print("   (Would these become false positives without the tier filter?)")
print("=" * 70)
if len(group_C) > 0:
    print(f"\n  What's in their skills_cleaned? Top 20:")
    # Show actual skills before cleaning
    raw_skills_c = group_C["skills_no_buzz"].str.strip().value_counts().head(20)
    for skill, count in raw_skills_c.items():
        print(f"    [{count:>3}x] '{skill.strip()}'")
    print(f"\n  Their original skills sources:")
    print(f"    skills_ai_det non-empty: {(group_C['skills_ai_det'].str.len() > 0).sum()}")
    print(f"    desc_ai_llm non-empty:   {(group_C['desc_ai_llm'].str.len() > 0).sum()}")
    print(f"    both non-empty:          {((group_C['skills_ai_det'].str.len() > 0) & (group_C['desc_ai_llm'].str.len() > 0)).sum()}")
    print(f"\n  Sample job titles from group C (first 15):")
    for _, row in group_C.head(15).iterrows():
        print(f"    Tier={row['desc_tier_llm']:<16} | '{row['job_title']}'")
        print(f"      skills_ai_det='{row['skills_ai_det']}' | desc_ai_llm='{row['desc_ai_llm']}'")
else:
    print("  (No jobs in this group)")
print()

print("=" * 70)
print("5. CONFIDENCE COMPARISON")
print("   (Is the LLM less confident when tier and skills disagree?)")
print("=" * 70)
for name, grp in [("A: Tier=AI, Skills=YES", group_A),
                   ("B: Tier=AI, Skills=NO", group_B),
                   ("C: Tier=none, Skills=YES", group_C),
                   ("D: Tier=none, Skills=NO", group_D)]:
    if len(grp) > 0:
        conf = grp["desc_conf_llm"]
        print(f"  {name:<30} n={len(grp):>5}  conf: mean={conf.mean():.3f}  median={conf.median():.3f}  min={conf.min():.3f}")
print()

print("=" * 70)
print("6. SUMMARY: Does skills_cleaned help or hurt?")
print("=" * 70)
print(f"""
  skills_cleaned filter removes {len(group_B)} jobs that tier classifies as AI.
  
  If group B contains jobs where tier correctly identified AI involvement
  but the job just doesn't mention specific AI tool names → skills_cleaned
  is creating FALSE NEGATIVES (wrongly excluding real AI jobs).
  
  If group B contains jobs where tier over-classified (hallucinated AI) and 
  the job truly has no specific AI skills → skills_cleaned is correctly
  filtering noise.
  
  Check group B samples above to judge which case applies.
""")
