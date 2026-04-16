"""
Verify the GenAI -> AI Integration tier override applied in the comparative
Stata do-file (§3.2b).

Rule: if cluster_generative_ai == 1 AND desc_tier_llm == "none" (after filters),
      the posting should be re-assigned to ai_level = 1 (AI Integration).
The rule must never DOWNGRADE a posting (i.e., never change ai_level from 2 -> 1).

Also produces diagnostic crosstabs for:
  - cluster_generative_ai x ai_level (pre-override): sanity
  - cluster_data_science__ml x ai_level (pre-override): to decide whether to
    additionally override ML cluster as well.

Run with:
    uv run python verify-scripts/verify_genai_override.py
"""
from __future__ import annotations

from pathlib import Path

import pandas as pd

DATA_DIR = Path("data/outputs")

FILES = {
    "US": DATA_DIR / "us_relevant_ai_stata.csv",
    "DE": DATA_DIR / "de" / "de_relevant_ai_stata.csv",
    "IN": DATA_DIR / "in_relevant_ai_stata.csv",
}


def load() -> pd.DataFrame:
    frames = []
    for country, path in FILES.items():
        if not path.exists():
            raise FileNotFoundError(f"Missing input file: {path}")
        frame = pd.read_csv(path, sep=";", encoding="utf-8-sig", low_memory=False)
        frame["country"] = country
        frames.append(frame)
    return pd.concat(frames, ignore_index=True)


def apply_stata_filters(df: pd.DataFrame) -> pd.DataFrame:
    # Confidence filter
    df = df[pd.to_numeric(df["desc_conf_llm"], errors="coerce") >= 0.7].copy()
    # Date filter (drop post_year <= 2023)
    df["discover_parsed"] = pd.to_datetime(df["discover_date"].str[:10], errors="coerce")
    df["age_in_days"] = pd.to_numeric(df["age_in_days"], errors="coerce")
    df["real_post_date"] = df["discover_parsed"] - pd.to_timedelta(df["age_in_days"], unit="D")
    df["post_year"] = df["real_post_date"].dt.year
    df = df[df["post_year"] > 2023].copy()
    return df


def derive_ai_level(df: pd.DataFrame) -> pd.DataFrame:
    tier = df["desc_tier_llm"].fillna("missing").replace({"": "missing"})
    tier = tier.replace({"core_ai": "applied_ai"})
    df["_tier"] = tier
    df["ai_level_pre"] = 0
    df.loc[tier == "ai_integration", "ai_level_pre"] = 1
    df.loc[tier == "applied_ai", "ai_level_pre"] = 2
    return df


def apply_override(df: pd.DataFrame) -> pd.DataFrame:
    df["ai_level_post"] = df["ai_level_pre"].copy()
    mask = (df["cluster_generative_ai"] == 1) & (df["ai_level_pre"] == 0)
    df.loc[mask, "ai_level_post"] = 1
    return df


def crosstab(df: pd.DataFrame, col: str, row_label: str) -> None:
    print(f"\n=== Crosstab {col} x ai_level (pre-override) ===")
    ct = pd.crosstab(df[col], df["ai_level_pre"], margins=True)
    print(ct)
    print(f"\nRow % (within {row_label} = value):")
    print(pd.crosstab(df[col], df["ai_level_pre"], normalize="index").mul(100).round(1))


def main() -> None:
    print("Loading datasets...")
    df = load()
    print(f"Raw observations: {len(df):,}")

    df = apply_stata_filters(df)
    print(f"After Stata filters (conf >= 0.7, post_year > 2023): {len(df):,}")

    df = derive_ai_level(df)
    df["cluster_generative_ai"] = pd.to_numeric(
        df["cluster_generative_ai"], errors="coerce"
    ).fillna(0).astype(int)
    df["cluster_data_science__ml"] = pd.to_numeric(
        df["cluster_data_science__ml"], errors="coerce"
    ).fillna(0).astype(int)

    # --- Pre-override diagnostics ---
    crosstab(df, "cluster_generative_ai", "cluster_generative_ai")
    crosstab(df, "cluster_data_science__ml", "cluster_data_science__ml")

    # --- Apply GenAI override and validate ---
    df = apply_override(df)

    n_moved = int(((df["ai_level_pre"] == 0) & (df["ai_level_post"] == 1)).sum())
    n_genai_none_pre = int(
        ((df["cluster_generative_ai"] == 1) & (df["ai_level_pre"] == 0)).sum()
    )
    n_genai_none_post = int(
        ((df["cluster_generative_ai"] == 1) & (df["ai_level_post"] == 0)).sum()
    )
    n_demoted = int((df["ai_level_post"] < df["ai_level_pre"]).sum())

    print("\n=== GenAI -> AI Integration override results ===")
    print(f"Postings moved None -> Integration: {n_moved:,}")
    print(f"Expected (GenAI == 1 & ai_level_pre == 0): {n_genai_none_pre:,}")
    print(f"Residual GenAI == 1 & ai_level_post == 0: {n_genai_none_post:,} (must be 0)")
    print(f"Postings demoted (ai_level_post < ai_level_pre): {n_demoted:,} (must be 0)")

    assert n_moved == n_genai_none_pre, "Override count mismatch"
    assert n_genai_none_post == 0, "Residual GenAI-None after override"
    assert n_demoted == 0, "Override caused a demotion — must not happen"

    # --- Post-override distribution by country ---
    print("\n=== AI tier distribution by country (post-override) ===")
    ct = pd.crosstab(df["ai_level_post"], df["country"], margins=True)
    print(ct)
    ct_pct = pd.crosstab(df["ai_level_post"], df["country"], normalize="columns").mul(100).round(1)
    print("\nSloupcova % (per country):")
    print(ct_pct)

    print("\nOK — GenAI override is internally consistent.")


if __name__ == "__main__":
    main()
