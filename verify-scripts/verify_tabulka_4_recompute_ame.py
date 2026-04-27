"""
Reproduce Tabulka 4 AMEs in Python via statsmodels MNLogit,
compare to values in Tabulka_4_Mlogit_AI_Tier.rtf.

Does NOT replicate cluster-robust SEs (that is a 2nd-order concern);
focuses on POINT ESTIMATES of AME (which is what the numbers verify).

Pipeline follows ai_skills_thesis_final.do sections 2, 3, 6:
 - Import 3 CSVs (US, DE, IN)
 - Filter desc_conf_llm >= 0.7
 - Construct real_post_date from discover_date - age_in_days, drop post_year <= 2023
 - Drop rows with missing desc_tier_llm
 - Merge core_ai into applied_ai
 - ai_level: 0 None, 1 AI Integration, 2 Applied/Core AI
 - exp_category: 0 Missing, 2 Junior, 3 Mid(ref), 4 Senior
 - sector_nace_num: encoded, baseline = "J" (Information & Communication)
 - type_cat: baseline = 1 (Private)
 - size_cat: baseline = 5 (1001-5000)
 - mlogit ai_level ~ cluster_* + ib3.exp + ib<J>.nace + ib1.type + ib5.size + is_remote
 - baseoutcome = 0 (None)
 - Compute AME for each outcome.
"""
from __future__ import annotations
import sys
import warnings
from pathlib import Path

import numpy as np
import pandas as pd

warnings.filterwarnings("ignore")

ROOT = Path("/Users/yakub/Projects/glassdoor-jobs-ai-skills")
DATADIR = ROOT / "data/outputs"


def load_country(csv: Path, country: str) -> pd.DataFrame:
    df = pd.read_csv(csv, sep=";", encoding="utf-8-sig", low_memory=False)
    df["country"] = country
    return df


def prep(df: pd.DataFrame) -> pd.DataFrame:
    print(f"\nprep(): input size {len(df):,}")
    # confidence
    df = df[df["desc_conf_llm"] >= 0.7].copy()
    print(f"  after conf>=0.7: {len(df):,}")
    # date filter
    df["discover_dt"] = pd.to_datetime(df["discover_date"].astype(str).str[:10], errors="coerce")
    df["age_in_days_num"] = pd.to_numeric(df["age_in_days"], errors="coerce")
    df["real_post_date"] = df["discover_dt"] - pd.to_timedelta(df["age_in_days_num"], unit="D")
    df["post_year"] = df["real_post_date"].dt.year
    df = df[df["post_year"] > 2023].copy()
    print(f"  after year>2023: {len(df):,}")
    # drop missing LLM
    df = df[df["desc_tier_llm"].notna() & (df["desc_tier_llm"].astype(str).str.strip() != "")].copy()
    print(f"  after drop empty tier: {len(df):,}")
    # merge core_ai -> applied_ai
    df["desc_tier_llm"] = df["desc_tier_llm"].replace({"core_ai": "applied_ai"})
    # ai_level
    df["ai_level"] = 0
    df.loc[df["desc_tier_llm"] == "ai_integration", "ai_level"] = 1
    df.loc[df["desc_tier_llm"] == "applied_ai", "ai_level"] = 2
    # exp_category from experience_min_llm
    df["exp_min"] = pd.to_numeric(df["experience_min_llm"], errors="coerce")
    df["exp_category"] = np.nan
    df.loc[df["exp_min"].isna(), "exp_category"] = 0
    df.loc[df["exp_min"].between(0, 2, inclusive="both"), "exp_category"] = 2
    df.loc[(df["exp_min"] > 2) & (df["exp_min"] <= 5), "exp_category"] = 3
    df.loc[df["exp_min"] > 5, "exp_category"] = 4
    return df


def prep_controls(df: pd.DataFrame) -> pd.DataFrame:
    """Construct size_cat, type_cat, sector_nace_num, is_remote as in do file."""
    # size_cat
    size = df["size"].fillna("").astype(str)
    size_cat = pd.Series(np.nan, index=df.index)
    unknown = size.isin(["", "Unknown", "Unbekannt"])
    size_cat[unknown] = 0
    size_cat[size.isin(["1 to 50 Employees", "1 bis 50 Mitarbeiter"])] = 1
    size_cat[size.isin(["51 to 200 Employees", "51 bis 200 Mitarbeiter"])] = 2
    size_cat[size.isin(["201 to 500 Employees", "201 bis 500 Mitarbeiter", "501 to 1000 Employees", "501 bis 1.000 Mitarbeiter"])] = 3
    size_cat[size.isin(["1001 to 5000 Employees", "1.001 bis 5.000 Mitarbeiter"])] = 5
    size_cat[size.isin(["5001 to 10000 Employees", "5.001 bis 10.000 Mitarbeiter"])] = 6
    size_cat[size.isin(["10000+ Employees", "Mehr als 10.000 Mitarbeiter"])] = 7
    df["size_cat"] = size_cat

    # type_cat
    t = df["type"].fillna("").astype(str)
    type_cat = pd.Series(0, index=df.index)
    type_cat[t.isin(["Company - Private", "Subsidiary or Business Segment", "Privatunternehmen", "Tochtergesellschaft oder Geschäftsbereich"])] = 1
    type_cat[t.isin(["Company - Public", "Aktiengesellschaft"])] = 2
    df["type_cat"] = type_cat

    # sector_nace aggregated to J / C / Other / Unknown
    nace = df["sector_nace"].fillna("").astype(str)
    nace = nace.replace({"": "Unknown"})
    nace = nace.where(nace.isin(["J", "C", "Unknown"]), "Other")
    df["sector_nace_agg"] = nace

    # is_remote
    rwt = df.get("remote_work_types", pd.Series("", index=df.index)).fillna("").astype(str).str.lower()
    df["is_remote"] = ((rwt.str.contains("home", na=False)) | (rwt.str.contains("remote", na=False))).astype(int)

    return df


def fit_and_ame(df: pd.DataFrame, country: str) -> dict[str, dict[str, tuple[float, float, float]]]:
    """Fit MNLogit for a given country and compute AMEs (no cluster robust SE).

    Returns dict {variable: {outcome_label: (AME, SE, p)}}
    """
    from statsmodels.discrete.discrete_model import MNLogit
    import statsmodels.formula.api as smf

    d = df[df["country"] == country].copy()
    print(f"\n=== {country}: N={len(d):,} ===")

    # cluster variables (exclude the 3 dropped in line 372-374)
    cluster_vars = [c for c in d.columns if c.startswith("cluster_")]
    drop_clusters = ["cluster_legacy__mainframe", "cluster_data_analysis__stats", "cluster_tools__editors"]
    cluster_vars = [c for c in cluster_vars if c not in drop_clusters]
    print(f"  Clusters: {len(cluster_vars)}")

    # Build RHS matrix with reference categories:
    #   exp_category: ref=3
    #   sector_nace_agg: ref=J
    #   type_cat: ref=1
    #   size_cat: ref=5
    exp_dummies = pd.get_dummies(d["exp_category"], prefix="exp").reindex(
        columns=["exp_0.0", "exp_2.0", "exp_3.0", "exp_4.0"], fill_value=0
    ).drop(columns=["exp_3.0"], errors="ignore")
    nace_dummies = pd.get_dummies(d["sector_nace_agg"], prefix="nace").reindex(
        columns=["nace_C", "nace_J", "nace_Other", "nace_Unknown"], fill_value=0
    ).drop(columns=["nace_J"], errors="ignore")
    type_dummies = pd.get_dummies(d["type_cat"], prefix="type").reindex(
        columns=["type_0", "type_1", "type_2"], fill_value=0
    ).drop(columns=["type_1"], errors="ignore")
    size_dummies = pd.get_dummies(d["size_cat"], prefix="size").reindex(
        columns=["size_0.0", "size_1.0", "size_2.0", "size_3.0", "size_5.0", "size_6.0", "size_7.0"], fill_value=0
    ).drop(columns=["size_5.0"], errors="ignore")

    X = pd.concat(
        [d[cluster_vars].astype(float), exp_dummies.astype(float), nace_dummies.astype(float),
         type_dummies.astype(float), size_dummies.astype(float), d[["is_remote"]].astype(float)],
        axis=1,
    )
    X["_cons"] = 1.0

    y = d["ai_level"].astype(int)

    # Drop rows with NaN in X or y
    mask = X.notna().all(axis=1) & y.notna()
    X = X[mask]
    y = y[mask]
    print(f"  After drop NaN: {len(y):,}")

    # Fit
    model = MNLogit(y, X)
    res = model.fit(method="newton", maxiter=100, disp=False)
    print(f"  Pseudo R² (McFadden): {res.prsquared:.4f}")

    # AMEs for all regressors, for each outcome
    # statsmodels MNLogit margeff returns dy/dx for a GIVEN outcome via get_margeff
    # For multi-outcome, we need to compute per outcome manually.
    # Use numerical AME: for each binary var, compute average of predicted P(outcome=k | x) - P(outcome=k | x where var=0 or baseline).
    # Since clusters are binary (0/1), AME = mean[P(k|x=1, rest) - P(k|x=0, rest)].
    # This is the "discrete change" interpretation which matches Stata's dy/dx for dummy vars.

    n_outcomes = 3  # 0, 1, 2
    ame_out = {}

    # Use numerical derivative (continuous) AME like Stata's default
    # d P(y=k) / d x_j, averaged over sample
    # Use symmetric difference quotient with small h
    h = 1e-4
    for var in cluster_vars:
        X_plus = X.copy()
        X_minus = X.copy()
        X_plus[var] = X[var] + h
        X_minus[var] = X[var] - h
        p_plus = res.predict(X_plus)  # (N, 3)
        p_minus = res.predict(X_minus)
        deriv = (p_plus - p_minus) / (2 * h)  # (N, 3)
        ame = deriv.mean(axis=0)
        ame_out[var] = ame

    # Store res for further diagnostics
    ame_out["__prsquared__"] = res.prsquared
    ame_out["__llf__"] = res.llf
    ame_out["__N__"] = len(y)
    return ame_out


def main():
    us = load_country(DATADIR / "us_relevant_ai_stata.csv", "US")
    de = load_country(DATADIR / "de/de_relevant_ai_stata.csv", "DE")
    inn = load_country(DATADIR / "in_relevant_ai_stata.csv", "IN")
    df = pd.concat([us, de, inn], ignore_index=True, sort=False)
    print(f"Raw concat: {len(df):,}")

    df = prep(df)
    df = prep_controls(df)

    print("\nPer-country N after prep:")
    print(df["country"].value_counts())

    # Expected Stata AMEs (decimal) from RTF, col 1 (USA-Integ), col 2 (USA-Applied) etc.
    # Let's grab them from the RTF parser. Quick inline dict for spot check:
    # From RTF, USA columns (Integ=col 1, Applied=col 2):
    EXPECTED = {
        "US": {
            "cluster_generative_ai": (-0.332, 0.233, 0.099),
            "cluster_data_science__ml": (-0.268, 0.142, 0.126),
            "cluster_cloud_computing": (-0.042, 0.028, 0.014),
            "cluster_dynamic__web": (-0.056, 0.026, 0.030),
            "cluster_frontend_development": (-0.032, 0.043, -0.011),
            "cluster_enterprise__managed": (0.048, -0.032, -0.016),
            "cluster_scripting__shell": (0.032, -0.016, -0.016),
            "cluster_systems_programming": (-0.016, -0.020, 0.036),
        },
        "DE": {
            "cluster_generative_ai": (-0.365, 0.225, 0.140),
            "cluster_data_science__ml": (-0.147, 0.024, 0.123),
            "cluster_cloud_computing": (-0.030, 0.025, 0.006),
            "cluster_dynamic__web": (-0.069, 0.008, 0.061),
            "cluster_frontend_development": (-0.012, 0.044, -0.032),
            "cluster_backend_development": (-0.025, 0.033, -0.008),
        },
        "IN": {
            "cluster_generative_ai": (-0.122, 0.070, 0.052),
            "cluster_data_science__ml": (-0.064, 0.006, 0.058),
            "cluster_cloud_computing": (-0.011, 0.005, 0.006),
            "cluster_dynamic__web": (-0.027, 0.005, 0.022),
            "cluster_frontend_development": (-0.007, 0.015, -0.007),
            "cluster_scripting__shell": (0.017, 0.009, -0.026),
        },
    }

    # Stata log values for sanity check
    STATA_REF = {
        "US": {"N": 17848, "prsquared": 0.3226, "llf": -7761.1528},
        "DE": {"N": 6402, "prsquared": 0.3608, "llf": -2462.7457},
        "IN": {"N": 14186, "prsquared": 0.5182, "llf": -1899.8059},
    }

    for country in ["US", "DE", "IN"]:
        ame = fit_and_ame(df, country)
        print(f"\n--- {country} AMEs (computed vs RTF) ---")
        sref = STATA_REF[country]
        print(f"  Stata: N={sref['N']:>6,}  prsquared={sref['prsquared']:.4f}  llf={sref['llf']:.2f}")
        print(f"  Python: N={ame['__N__']:>6,}  prsquared={ame['__prsquared__']:.4f}  llf={ame['__llf__']:.2f}")
        print(f"  {'Cluster':<35s} {'Outcome':<8s} {'Python':>8s} {'RTF':>8s} {'Δ(p.p.)':>8s}")
        for var, expected in EXPECTED[country].items():
            if var not in ame:
                print(f"  {var}: NOT FOUND")
                continue
            for k in (0, 1, 2):
                p_val = ame[var][k] * 100
                r_val = expected[k] * 100
                delta = p_val - r_val
                print(f"  {var:<35s} out={k}    {p_val:>7.2f}  {r_val:>7.2f}  {delta:>+7.2f}")


if __name__ == "__main__":
    main()
