"""
Verify that all string values in DE/IN/US datasets are properly mapped
to their categorical counterparts in the comparative Stata DO-file.

This catches the bug where German strings (e.g. "Privatunternehmen")
were not matched by English-only patterns, causing listwise deletion.
"""
from __future__ import annotations

import pandas as pd
from pathlib import Path

DATA_DIR = Path("data/outputs")

# --- Mapping definitions (must mirror the DO-file exactly) ---

SIZE_MAP: dict[str, int] = {
    "Unknown": 0, "Unbekannt": 0,
    "1 to 50 Employees": 1, "1 bis 50 Mitarbeiter": 1,
    "51 to 200 Employees": 2, "51 bis 200 Mitarbeiter": 2,
    "201 to 500 Employees": 3, "201 bis 500 Mitarbeiter": 3,
    "501 to 1000 Employees": 4, "501 bis 1.000 Mitarbeiter": 4,
    "1001 to 5000 Employees": 5, "1.001 bis 5.000 Mitarbeiter": 5,
    "5001 to 10000 Employees": 6, "5.001 bis 10.000 Mitarbeiter": 6,
    "10000+ Employees": 7, "Mehr als 10.000 Mitarbeiter": 7,
}

TYPE_MAP: dict[str, int] = {
    "": 0, "Unknown": 0, "Contract": 0, "Self-employed": 0,
    "Private Practice / Firm": 0, "Franchise": 0,
    "Unbekannt": 0, "Auftragsunternehmen": 0,
    "Selbstständig": 0, "Privatpraxis/Kanzlei": 0,
    "Company - Private": 1, "Subsidiary or Business Segment": 1,
    "Privatunternehmen": 1, "Tochtergesellschaft oder Geschäftsbereich": 1,
    "Company - Public": 2, "Aktiengesellschaft": 2,
    "Nonprofit Organization": 4, "Non-profit Organisation": 4,
    "Government": 4, "College / University": 4,
    "School / School District": 4, "Hospital": 4,
    "Gemeinnützige Organisation": 4, "Öffentlicher Dienst": 4,
    "Hochschule/Universität": 4, "Schule/Schulbezirk": 4,
    "Krankenhaus": 4,
}

EDU_VALID = {"highschool", "associate", "bachelor", "master", "missing"}

datasets = {
    "US": DATA_DIR / "us_relevant_ai_stata.csv",
    "DE": DATA_DIR / "de" / "de_relevant_ai_stata.csv",
    "IN": DATA_DIR / "in_relevant_ai_stata.csv",
}

all_ok = True

for country, path in datasets.items():
    print(f"\n{'='*60}")
    print(f"  COUNTRY: {country}  ({path})")
    print(f"{'='*60}")
    df = pd.read_csv(path, sep=";", low_memory=False)

    # --- size ---
    size_vals = set(df["size"].dropna().unique())
    unmapped_size = size_vals - set(SIZE_MAP.keys())
    if unmapped_size:
        print(f"  ❌ UNMAPPED size values: {unmapped_size}")
        for v in unmapped_size:
            cnt = (df["size"] == v).sum()
            print(f"     '{v}' -> {cnt} rows")
        all_ok = False
    else:
        print(f"  ✅ size: all {len(size_vals)} values mapped")

    # --- type ---
    type_vals = set(df["type"].dropna().unique())
    unmapped_type = type_vals - set(TYPE_MAP.keys())
    if unmapped_type:
        print(f"  ❌ UNMAPPED type values: {unmapped_type}")
        for v in unmapped_type:
            cnt = (df["type"] == v).sum()
            print(f"     '{v}' -> {cnt} rows")
        all_ok = False
    else:
        print(f"  ✅ type: all {len(type_vals)} values mapped")

    # --- education (simulate the hybrid logic) ---
    edu = df["edulevel_llm"].fillna("").str.lower()
    edu = edu.str.replace("'s", "", regex=False)
    edu = edu.replace({"high school": "highschool", "-": "missing", "": "missing"})
    edu = edu.replace({"phd": "master", "diploma": "associate"})
    # Apply edu_level_det fallback
    det = df["edu_level_det"].fillna("")
    mask_missing = edu == "missing"
    mask_det = det != ""
    edu.loc[mask_missing & mask_det] = det.loc[mask_missing & mask_det]
    edu = edu.replace({"phd": "master", "": "missing"})
    
    unmapped_edu = set(edu.unique()) - EDU_VALID
    if unmapped_edu:
        print(f"  ❌ UNMAPPED education values: {unmapped_edu}")
        for v in unmapped_edu:
            cnt = (edu == v).sum()
            print(f"     '{v}' -> {cnt} rows")
        all_ok = False
    else:
        print(f"  ✅ education: all values resolve to valid categories")

    # --- pay_currency ---
    currencies = set(df["pay_currency"].dropna().unique())
    expected = {"USD", "EUR", "INR", "COP"}
    unexpected = currencies - expected
    if unexpected:
        print(f"  ⚠️  Unexpected currencies: {unexpected}")
        all_ok = False
    else:
        print(f"  ✅ pay_currency: {currencies}")

    # --- sector_nace (should all be single letters or 'Unknown') ---
    sectors = set(df["sector_nace"].dropna().unique())
    print(f"  ℹ️  sector_nace values: {sorted(sectors)}")

    # --- NaN counts for key columns ---
    for col in ["size", "type", "salary_mid", "pay_currency", "pay_period"]:
        n_na = df[col].isna().sum()
        if n_na > 0:
            print(f"  ℹ️  {col}: {n_na} NaN rows (Stata imports as empty string)")


print(f"\n{'='*60}")
if all_ok:
    print("✅ ALL CHECKS PASSED — no unmapped categories found.")
else:
    print("❌ SOME CHECKS FAILED — see above for details.")
print(f"{'='*60}")
