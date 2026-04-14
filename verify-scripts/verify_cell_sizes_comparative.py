"""
Verify minimum cell sizes (50-observation rule) for all categorical predictors
used in logit and mlogit models in the comparative analysis.

For binary logit: cross-tabulate each factor × has_ai (0/1)
For multinomial logit: cross-tabulate each factor × ai_level (0/1/2)

Rule: Every cell should have >= 50 observations.
"""
from __future__ import annotations

import pandas as pd
from pathlib import Path

# --- Load all three datasets ---
data_dir = Path("data/inputs")

us = pd.read_csv(data_dir / "us_relevant_ai_stata.csv", sep=";", encoding="utf-8-sig")
us["country"] = "US"

de = pd.read_csv(data_dir / "de_relevant_ai_stata.csv", sep=";", encoding="utf-8-sig")
de["country"] = "DE"

india = pd.read_csv(data_dir / "in_relevant_ai_stata.csv", sep=";", encoding="utf-8-sig")
india["country"] = "IN"

df = pd.concat([us, de, india], ignore_index=True)
print(f"Total observations loaded: {len(df)}")

# --- Apply same filters as Stata DO-file ---
df = df[df["desc_conf_llm"] >= 0.7].copy()

# Date filter (drop <= 2023)
df["discover_date_parsed"] = pd.to_datetime(df["discover_date"].str[:10], errors="coerce")
df["age_in_days"] = pd.to_numeric(df["age_in_days"], errors="coerce")
df["real_post_date"] = df["discover_date_parsed"] - pd.to_timedelta(df["age_in_days"], unit="D")
df["post_year"] = df["real_post_date"].dt.year
df = df[df["post_year"] > 2023].copy()

# Drop non-standard currencies for DE/IN
df = df[~((df["country"] == "DE") & (~df["pay_currency"].isin(["EUR", ""]) & df["pay_currency"].notna()))].copy()
df = df[~((df["country"] == "IN") & (df["pay_currency"].isna() | (df["pay_currency"] == "")) & df["salary_mid"].notna())].copy()

print(f"After filters: {len(df)}")

# --- Derive variables exactly as in DO-file ---

# AI tier
df["desc_tier_llm"] = df["desc_tier_llm"].fillna("missing")
df.loc[df["desc_tier_llm"] == "core_ai", "desc_tier_llm"] = "applied_ai"

# has_ai (binary)
df["has_ai"] = ((df["desc_tier_llm"] != "none") & (df["desc_tier_llm"] != "missing")).astype(int)

# ai_level (multinomial)
df["ai_level"] = 0
df.loc[df["desc_tier_llm"] == "ai_integration", "ai_level"] = 1
df.loc[df["desc_tier_llm"] == "applied_ai", "ai_level"] = 2

# Education hybrid
df["edulevel_llm"] = df["edulevel_llm"].fillna("")
df["education_hybrid"] = df["edulevel_llm"].str.lower().str.replace("'s", "", regex=False)
df.loc[df["education_hybrid"] == "high school", "education_hybrid"] = "highschool"
df.loc[df["education_hybrid"].isin(["-", ""]), "education_hybrid"] = "missing"
# fallback
mask_missing_edu = (df["education_hybrid"] == "missing") & (df["edu_level_det"].fillna("") != "")
df.loc[mask_missing_edu, "education_hybrid"] = df.loc[mask_missing_edu, "edu_level_det"]
df.loc[df["education_hybrid"] == "", "education_hybrid"] = "missing"
df.loc[df["education_hybrid"] == "phd", "education_hybrid"] = "master"
df.loc[df["education_hybrid"] == "diploma", "education_hybrid"] = "associate"
# safety net
valid_edu = {"highschool", "associate", "bachelor", "master", "missing"}
df.loc[~df["education_hybrid"].isin(valid_edu), "education_hybrid"] = "missing"

# edu_logit (3-level for binary logit)
df["edu_logit"] = "missing"
df.loc[df["education_hybrid"].isin(["missing", ""]), "edu_logit"] = "Missing"
df.loc[df["education_hybrid"].isin(["highschool", "associate"]), "edu_logit"] = "HS / Associate"
df.loc[df["education_hybrid"].isin(["bachelor", "master"]), "edu_logit"] = "Bachelor+"

# Experience category
df["experience_min_llm"] = pd.to_numeric(df["experience_min_llm"], errors="coerce")
df["exp_category"] = "Missing"
df.loc[(df["experience_min_llm"] >= 0) & (df["experience_min_llm"] <= 2), "exp_category"] = "Junior (0-2)"
df.loc[(df["experience_min_llm"] > 2) & (df["experience_min_llm"] <= 5), "exp_category"] = "Mid (3-5)"
df.loc[(df["experience_min_llm"] > 5), "exp_category"] = "Senior+ (6+)"

# Size category
df["size"] = df["size"].fillna("Unknown")
df.loc[df["size"] == "", "size"] = "Unknown"

size_map = {
    "Unknown": "Unknown", "Unbekannt": "Unknown",
    "1 to 50 Employees": "1-50", "1 bis 50 Mitarbeiter": "1-50",
    "51 to 200 Employees": "51-200", "51 bis 200 Mitarbeiter": "51-200",
    "201 to 500 Employees": "201-500", "201 bis 500 Mitarbeiter": "201-500",
    "501 to 1000 Employees": "501-1000", "501 bis 1.000 Mitarbeiter": "501-1000",
    "1001 to 5000 Employees": "1001-5000", "1.001 bis 5.000 Mitarbeiter": "1001-5000",
    "5001 to 10000 Employees": "5001-10000", "5.001 bis 10.000 Mitarbeiter": "5001-10000",
    "10000+ Employees": "10000+", "Mehr als 10.000 Mitarbeiter": "10000+",
}
df["size_cat"] = df["size"].map(size_map)

# Type category
type_unknown = {"", "Unknown", "Contract", "Self-employed", "Private Practice / Firm", "Franchise",
                "Unbekannt", "Auftragsunternehmen", "Selbstständig", "Privatpraxis/Kanzlei"}
type_private = {"Company - Private", "Subsidiary or Business Segment",
                "Privatunternehmen", "Tochtergesellschaft oder Geschäftsbereich"}
type_public = {"Company - Public", "Aktiengesellschaft"}
type_ngo = {"Nonprofit Organization", "Non-profit Organisation", "Government",
            "College / University", "School / School District", "Hospital",
            "Gemeinnützige Organisation", "Öffentlicher Dienst",
            "Hochschule/Universität", "Schule/Schulbezirk", "Krankenhaus"}

df["type"] = df["type"].fillna("")
df["type_cat"] = "Unknown/Other"
df.loc[df["type"].isin(type_unknown), "type_cat"] = "Unknown/Other"
df.loc[df["type"].isin(type_private), "type_cat"] = "Private/Subsidiary"
df.loc[df["type"].isin(type_public), "type_cat"] = "Public"
df.loc[df["type"].isin(type_ngo), "type_cat"] = "Nonprofit/Gov/Edu"

# NACE sector
df["sector_nace"] = df["sector_nace"].fillna("Unknown")
df.loc[~df["sector_nace"].isin(["J", "C", "K", "M", "Q", "Unknown"]), "sector_nace"] = "Other"

# Job family
df["job_family"] = df["job_family"].fillna("Unknown")
df.loc[df["job_family"].isin(["Frontend & Design", "QA & Testing", "Security", "Systems & Embedded"]), "job_family"] = "Other"

# Country
df["country_cat"] = df["country"]

# ============================================================
# CHECK CELL SIZES
# ============================================================

# Categorical predictors used in logit/mlogit
factors_logit = {
    "country_cat": "Country",
    "sector_nace": "NACE sector",
    "type_cat": "Type",
    "size_cat": "Size",
    "edu_logit": "Education (logit)",  # used in binary logit only
    "exp_category": "Experience",
    "job_family": "Job family",
}

factors_mlogit = {
    "country_cat": "Country",
    "sector_nace": "NACE sector",
    "type_cat": "Type",
    "size_cat": "Size",
    # edu_logit NOT used in mlogit (by design — documented in DO-file)
    "exp_category": "Experience",
    "job_family": "Job family",
}

THRESHOLD = 50

print("\n" + "=" * 80)
print("BINARY LOGIT CELL SIZES: factor × has_ai")
print("=" * 80)

violations_logit = []
for col, label in factors_logit.items():
    ct = pd.crosstab(df[col], df["has_ai"], margins=False)
    ct.columns = ["has_ai=0", "has_ai=1"]
    print(f"\n--- {label} ({col}) ---")
    print(ct.to_string())
    for idx in ct.index:
        for c in ct.columns:
            val = ct.loc[idx, c]
            if val < THRESHOLD:
                violations_logit.append((label, str(idx), c, val))
                print(f"  ⚠️  VIOLATION: {label}={idx} × {c} = {val} (< {THRESHOLD})")

print("\n" + "=" * 80)
print("MULTINOMIAL LOGIT CELL SIZES: factor × ai_level")
print("=" * 80)

violations_mlogit = []
for col, label in factors_mlogit.items():
    ct = pd.crosstab(df[col], df["ai_level"], margins=False)
    ct.columns = ["None", "AI_Integration", "Applied_Core_AI"]
    print(f"\n--- {label} ({col}) ---")
    print(ct.to_string())
    for idx in ct.index:
        for c in ct.columns:
            val = ct.loc[idx, c]
            if val < THRESHOLD:
                violations_mlogit.append((label, str(idx), c, val))
                print(f"  ⚠️  VIOLATION: {label}={idx} × {c} = {val} (< {THRESHOLD})")

# Per-country breakdown for mlogit (since models include country FE, 
# effective cell sizes within each country matter)
print("\n" + "=" * 80)
print("PER-COUNTRY MLOGIT CELL SIZES: factor × ai_level (within country)")
print("=" * 80)

violations_percountry = []
for c_name in ["US", "DE", "IN"]:
    print(f"\n{'=' * 40}")
    print(f"  COUNTRY = {c_name}")
    print(f"{'=' * 40}")
    sub = df[df["country"] == c_name]
    for col, label in factors_mlogit.items():
        if col == "country_cat":
            continue
        ct = pd.crosstab(sub[col], sub["ai_level"], margins=False)
        ct.columns = ["None", "AI_Integration", "Applied_Core_AI"]
        has_violation = False
        for idx in ct.index:
            for cc in ct.columns:
                val = ct.loc[idx, cc]
                if val < THRESHOLD:
                    has_violation = True
                    violations_percountry.append((c_name, label, str(idx), cc, val))
        if has_violation:
            print(f"\n--- {label} ({col}) --- ⚠️ HAS VIOLATIONS")
            print(ct.to_string())
            for idx in ct.index:
                for cc in ct.columns:
                    val = ct.loc[idx, cc]
                    if val < THRESHOLD:
                        print(f"  ⚠️  {c_name}/{label}={idx} × {cc} = {val}")

# ============================================================
# SUMMARY
# ============================================================
print("\n" + "=" * 80)
print("SUMMARY")
print("=" * 80)

print(f"\nBinary logit violations (pooled, < {THRESHOLD}):")
if violations_logit:
    for label, cat, outcome, n in violations_logit:
        print(f"  ⚠️  {label}={cat} × {outcome}: N={n}")
else:
    print("  ✅ None — all cells >= 50")

print(f"\nMultinomial logit violations (pooled, < {THRESHOLD}):")
if violations_mlogit:
    for label, cat, outcome, n in violations_mlogit:
        print(f"  ⚠️  {label}={cat} × {outcome}: N={n}")
else:
    print("  ✅ None — all cells >= 50")

print(f"\nPer-country mlogit violations (within country, < {THRESHOLD}):")
if violations_percountry:
    for c_name, label, cat, outcome, n in violations_percountry:
        print(f"  ⚠️  {c_name}: {label}={cat} × {outcome}: N={n}")
    print(f"\n  Total per-country violations: {len(violations_percountry)}")
else:
    print("  ✅ None — all cells >= 50")

print("\nDone.")
