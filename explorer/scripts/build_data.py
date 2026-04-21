"""Build JSON data assets for the Thesis Explorer web app.

Reads Stata-generated charts_data CSVs and raw ai_stata CSVs, transforms them
into typed JSON assets consumed by the Next.js frontend.

Usage:
    uv run python explorer/scripts/build_data.py [--run RUN_DIR]

Outputs to: explorer/public/data/
"""
from __future__ import annotations

import argparse
import json
import logging
import random
import sys
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import pandas as pd

# ---------------------------------------------------------------------------
# Paths
# ---------------------------------------------------------------------------
SCRIPT_DIR = Path(__file__).resolve().parent
EXPLORER_ROOT = SCRIPT_DIR.parent
PROJECT_ROOT = EXPLORER_ROOT.parent
STATA_OUT = PROJECT_ROOT / "analysis" / "stata" / "output"
DATA_OUT = PROJECT_ROOT / "data" / "outputs"
PUBLIC_DATA = EXPLORER_ROOT / "public" / "data"
CHARTS_DATA = PUBLIC_DATA / "charts"

# ---------------------------------------------------------------------------
# Logging
# ---------------------------------------------------------------------------
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s %(levelname)-7s %(message)s",
    datefmt="%H:%M:%S",
)
log = logging.getLogger("build_data")

# ---------------------------------------------------------------------------
# Config — labels mirror analysis/charts/build_charts.py (English only)
# ---------------------------------------------------------------------------
CLUSTER_LABELS: dict[str, str] = {
    "cluster_generative_ai": "Generative AI",
    "cluster_data_science__ml": "Data Science / ML",
    "cluster_dynamic_web": "Dynamic Web",
    "cluster_cloud_computing": "Cloud Computing",
    "cluster_data_engineering": "Data Engineering",
    "cluster_bi__analytics": "BI & Analytics",
    "cluster_frontend_development": "Frontend Dev",
    "cluster_enterprise_platforms": "Enterprise Platforms",
    "cluster_devops__containers": "DevOps & Containers",
    "cluster_backend_development": "Backend Dev",
    "cluster_systems_programming": "Systems Programming",
    "cluster_architecture__methods": "Architecture & Methods",
    "cluster_security__identity": "Security & Identity",
    "cluster_mobile__desktop": "Mobile & Desktop",
    "cluster_testing__qa__debugging": "Testing / QA",
    "cluster_databases__storage": "Databases & Storage",
    "cluster_networking": "Networking",
    "cluster_os__embedded": "OS & Embedded",
    "cluster_certifications": "Certifications",
    "cluster_scripting__shell": "Scripting / Shell",
    "cluster_enterprise__managed": "Enterprise / Managed",
}

# Frequency CSVs use a slightly different naming convention (triple underscore).
CLUSTER_LABELS_ALT: dict[str, str] = {
    "cluster_architecture___methods": "Architecture & Methods",
    "cluster_bi___analytics": "BI & Analytics",
    "cluster_data_science___ml": "Data Science / ML",
    "cluster_devops___containers": "DevOps & Containers",
    "cluster_enterprise___managed": "Enterprise / Managed",
    "cluster_os___embedded": "OS & Embedded",
    "cluster_scripting___shell": "Scripting / Shell",
    "cluster_security___identity": "Security & Identity",
    "cluster_mobile___desktop": "Mobile & Desktop",
    "cluster_testing___qa___debugging": "Testing / QA",
    "cluster_databases___storage": "Databases & Storage",
    "cluster_data_analysis___stats": "Data Analysis / Stats",
    "cluster_legacy___mainframe": "Legacy / Mainframe",
    "cluster_tools___editors": "Tools / Editors",
    "cluster_dynamic___web": "Dynamic Web",
}


def _nice_cluster(key: str) -> str:
    return (
        CLUSTER_LABELS.get(key)
        or CLUSTER_LABELS_ALT.get(key)
        or key.replace("cluster_", "").replace("___", " / ").replace("__", " & ").replace("_", " ").title()
    )


COUNTRY_LABELS: dict[str, str] = {"US": "United States", "DE": "Germany", "IN": "India"}
COUNTRIES = ["US", "DE", "IN"]
AI_LEVEL_LABELS = {0: "None", 1: "AI Integration", 2: "Applied/Core AI"}
AI_LEVEL_ORDER = ["None", "AI Integration", "Applied/Core AI"]


def _sig_stars(p: float) -> str:
    if p < 0.001:
        return "***"
    if p < 0.01:
        return "**"
    if p < 0.05:
        return "*"
    return "ns"


def _to_float(x: Any) -> float | None:
    """Stata writes 9.07906569791e-06 correctly; keep precision as float."""
    try:
        val = float(x)
    except (TypeError, ValueError):
        return None
    if pd.isna(val):
        return None
    return val


# ---------------------------------------------------------------------------
# Path resolution
# ---------------------------------------------------------------------------
def latest_run_dir() -> Path:
    candidates = sorted(STATA_OUT.glob("thesis_final_run_*"))
    if not candidates:
        raise FileNotFoundError(f"No thesis_final_run_* in {STATA_OUT}")
    return candidates[-1]


# ---------------------------------------------------------------------------
# Builders
# ---------------------------------------------------------------------------
def build_g1(run_dir: Path) -> list[dict[str, Any]]:
    """AI tier composition per country."""
    df = pd.read_csv(run_dir / "charts_data" / "g1_ai_tier_by_country.csv", keep_default_na=False)
    df["pct"] = df["pct"].astype(float)
    df["_freq"] = df["_freq"].astype(int)
    rows: list[dict[str, Any]] = []
    for country in COUNTRIES:
        sub = df[df["country"] == country].copy()
        if sub.empty:
            continue
        row: dict[str, Any] = {
            "country": country,
            "country_label": COUNTRY_LABELS[country],
            "total": int(sub["_freq"].sum()),
        }
        for lvl in AI_LEVEL_ORDER:
            lvl_rows = sub[sub["ai_level"] == lvl]
            if lvl_rows.empty:
                row[lvl] = {"pct": 0.0, "count": 0}
            else:
                row[lvl] = {
                    "pct": round(float(lvl_rows.iloc[0]["pct"]), 2),
                    "count": int(lvl_rows.iloc[0]["_freq"]),
                }
        row["ai_share"] = round(row["AI Integration"]["pct"] + row["Applied/Core AI"]["pct"], 2)
        rows.append(row)
    log.info("g1: %d country rows", len(rows))
    return rows


def build_g2(run_dir: Path) -> list[dict[str, Any]]:
    """AI share by job family for each country."""
    df = pd.read_csv(run_dir / "charts_data" / "g2_ai_share_by_jobfamily.csv")
    out: list[dict[str, Any]] = []
    for _, row in df.iterrows():
        out.append({
            "country": str(row["country"]),
            "job_family": str(row["job_family"]),
            "ai_share": round(_to_float(row["ai_share"]) or 0.0, 2),
            "n": int(row["n"]),
        })
    log.info("g2: %d rows (country × job_family)", len(out))
    return out


def _coef_rows(df: pd.DataFrame) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    for _, r in df.iterrows():
        b = _to_float(r["b"])
        se = _to_float(r["se"])
        p = _to_float(r["p"])
        rows.append({
            "coef": str(r["coef"]),
            "label": _nice_cluster(str(r["coef"])),
            "b": b,
            "se": se,
            "z": _to_float(r["z"]),
            "p": p,
            "ci_low": _to_float(r["ci_low"]),
            "ci_high": _to_float(r["ci_high"]),
            "sig": _sig_stars(p) if p is not None else "ns",
        })
    return rows


def build_g3(run_dir: Path) -> list[dict[str, Any]]:
    """Forest plot — logit AME US."""
    df = pd.read_csv(run_dir / "charts_data" / "g3_logit_ame_us.csv")
    rows = _coef_rows(df)
    log.info("g3: %d coefficients (US logit AME)", len(rows))
    return rows


def build_g4(run_dir: Path) -> dict[str, list[dict[str, Any]]]:
    """Multinomial logit: integration vs applied (US)."""
    integ = _coef_rows(pd.read_csv(run_dir / "charts_data" / "g4_mlogit_us_integration.csv"))
    appl = _coef_rows(pd.read_csv(run_dir / "charts_data" / "g4_mlogit_us_applied.csv"))
    log.info("g4: integration=%d, applied=%d", len(integ), len(appl))
    return {"integration": integ, "applied": appl}


def build_g5(run_dir: Path) -> dict[str, list[dict[str, Any]]]:
    """Cross-country cluster AME — heatmap."""
    result: dict[str, list[dict[str, Any]]] = {}
    for country in COUNTRIES:
        path = run_dir / "charts_data" / f"g5_logit_ame_{country}.csv"
        result[country] = _coef_rows(pd.read_csv(path))
        log.info("g5[%s]: %d coefficients", country, len(result[country]))
    return result


def _premium_rows(df: pd.DataFrame) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    for _, r in df.iterrows():
        coef = str(r["coef"])
        # ai_level comes in as "1.ai_level" / "2.ai_level"
        tier_label = None
        if coef.startswith("1."):
            tier_label = "AI Integration"
        elif coef.startswith("2."):
            tier_label = "Applied/Core AI"
        p = _to_float(r["p"])
        rows.append({
            "coef": coef,
            "tier": tier_label,
            "b": _to_float(r["b"]),
            "se": _to_float(r["se"]),
            "p": p,
            "ci_low": _to_float(r["ci_low"]),
            "ci_high": _to_float(r["ci_high"]),
            "sig": _sig_stars(p) if p is not None else "ns",
        })
    return rows


def build_g6(run_dir: Path) -> dict[str, list[dict[str, Any]]]:
    """Salary premium decomposition: Model A → B → C (US)."""
    result: dict[str, list[dict[str, Any]]] = {}
    for model in ["A", "B", "C"]:
        df = pd.read_csv(run_dir / "charts_data" / f"g6_ols_{model}.csv")
        # Keep only ai_level rows for the premium chart
        df = df[df["coef"].str.contains(r"\.ai_level", regex=True, na=False)]
        result[model] = _premium_rows(df)
        log.info("g6[Model %s]: %d tier coefficients", model, len(result[model]))
    return result


def build_g7(run_dir: Path) -> dict[str, list[dict[str, Any]]]:
    """Cross-country salary premium."""
    result: dict[str, list[dict[str, Any]]] = {}
    for country in COUNTRIES:
        df = pd.read_csv(run_dir / "charts_data" / f"g7_ols_{country}.csv")
        df = df[df["coef"].str.contains(r"\.ai_level", regex=True, na=False)]
        result[country] = _premium_rows(df)
        log.info("g7[%s]: %d tier coefficients", country, len(result[country]))
    return result


# ---------------------------------------------------------------------------
# Aggregate builders
# ---------------------------------------------------------------------------
def build_kpi(g1: list[dict[str, Any]], g7: dict[str, list[dict[str, Any]]]) -> dict[str, Any]:
    """KPI dashboard cards."""
    kpi: dict[str, Any] = {"countries": []}
    for country_row in g1:
        code = country_row["country"]
        premium_rows = g7.get(code, [])
        integration = next((r for r in premium_rows if r["tier"] == "AI Integration"), None)
        applied = next((r for r in premium_rows if r["tier"] == "Applied/Core AI"), None)

        def _pct(b: float | None) -> float | None:
            if b is None:
                return None
            return round(b * 100, 2)

        kpi["countries"].append({
            "country": code,
            "country_label": COUNTRY_LABELS[code],
            "total_jobs": country_row["total"],
            "ai_share": country_row["ai_share"],
            "tier_none_pct": country_row["None"]["pct"],
            "tier_integration_pct": country_row["AI Integration"]["pct"],
            "tier_applied_pct": country_row["Applied/Core AI"]["pct"],
            "premium_integration_pct": _pct(integration["b"]) if integration else None,
            "premium_applied_pct": _pct(applied["b"]) if applied else None,
            "premium_integration_sig": integration["sig"] if integration else "ns",
            "premium_applied_sig": applied["sig"] if applied else "ns",
        })
    # Top-line headlines
    us = next(c for c in kpi["countries"] if c["country"] == "US")
    de = next(c for c in kpi["countries"] if c["country"] == "DE")
    in_ = next(c for c in kpi["countries"] if c["country"] == "IN")
    kpi["headlines"] = {
        "ai_share_diff_us_in": round(us["ai_share"] - in_["ai_share"], 2),
        "premium_us_applied": us["premium_applied_pct"],
        "premium_de_applied_ns": de["premium_applied_pct"],
        "most_common_ai_country": max(kpi["countries"], key=lambda c: c["ai_share"])["country"],
    }
    log.info("kpi: headlines computed")
    return kpi


def build_clusters() -> list[dict[str, Any]]:
    """Cluster → hardskills mapping with counts."""
    freq_path = DATA_OUT / "us_relevant_ai_stata_cluster_frequency.csv"
    skills_path = DATA_OUT / "us_relevant_ai_stata_cluster_skills.csv"
    freq = pd.read_csv(freq_path)
    skills = pd.read_csv(skills_path)

    out: list[dict[str, Any]] = []
    for _, row in freq.iterrows():
        cluster_key = str(row["Cluster"])
        label = _nice_cluster(cluster_key)
        # Collect top skills for this cluster (already sorted by Rank)
        cluster_skills = skills[skills["Cluster"] == cluster_key].sort_values("Rank")
        top_skills = [
            {
                "skill": str(s["Skill"]),
                "count": int(s["Count"]),
                "pct": round(float(s["Percentage"]), 2),
            }
            for _, s in cluster_skills.head(25).iterrows()
        ]
        out.append({
            "key": cluster_key,
            "label": label,
            "frequency": int(row["Frequency"]),
            "pct": round(float(row["Percentage"]), 2),
            "top_skills": top_skills,
        })
    out.sort(key=lambda x: x["frequency"], reverse=True)
    log.info("clusters: %d clusters mapped to skills", len(out))
    return out


def build_jobfamilies(g2: list[dict[str, Any]]) -> list[dict[str, Any]]:
    """Job families summary (aggregate across countries where desired)."""
    # Keep per-country entries as-is; frontend can pivot
    log.info("jobfamilies: %d rows across countries", len(g2))
    return g2


# ---------------------------------------------------------------------------
# Raw sample builder
# ---------------------------------------------------------------------------
SAMPLE_COLUMNS = [
    "id", "job_title", "company", "country", "city", "state",
    "job_family", "skill_cluster",
    "desc_tier_llm", "is_real_ai",
    "salary_min", "salary_mid", "salary_max", "pay_currency",
    "edu_level_det", "experience_min_llm",
    "industry", "sector", "size", "type",
    "hardskills", "softskills",
    "remote_work_types",
]


def _read_csv_subset(path: Path, country_override: str | None = None) -> pd.DataFrame:
    df = pd.read_csv(path, sep=";", encoding="utf-8-sig", low_memory=False)
    # Some columns might be missing in some regional files — fill with NaN
    for col in SAMPLE_COLUMNS:
        if col not in df.columns:
            df[col] = None
    if country_override:
        df["country"] = country_override
    return df[SAMPLE_COLUMNS].copy()


def build_jobs_sample(sample_size: int = 500, seed: int = 42) -> list[dict[str, Any]]:
    """Stratified sample across country × ai_level."""
    random.seed(seed)

    frames: list[pd.DataFrame] = []
    src_us = DATA_OUT / "us_relevant_ai_stata.csv"
    src_de = DATA_OUT / "de" / "de_relevant_ai_stata.csv"
    src_in = DATA_OUT / "in_relevant_ai_stata.csv"
    for src, override in [(src_us, "US"), (src_de, "DE"), (src_in, "IN")]:
        if not src.exists():
            log.warning("Missing %s — skipping", src)
            continue
        df = _read_csv_subset(src, country_override=override)
        frames.append(df)

    if not frames:
        raise FileNotFoundError("No stata CSVs found for sample")
    all_df = pd.concat(frames, ignore_index=True)

    # Map ai_level strings to ordinal 0/1/2 for stratification
    def tier_ord(val: Any) -> int:
        s = str(val).strip().lower()
        if s in ("none", "nan", "", "0"):
            return 0
        if "integration" in s or s == "1":
            return 1
        if "applied" in s or "core" in s or s == "2":
            return 2
        return 0

    all_df["_tier_ord"] = all_df["desc_tier_llm"].apply(tier_ord)

    # Strata counts: proportional to country counts, but with per-tier floor
    strata = all_df.groupby(["country", "_tier_ord"]).size()
    total = strata.sum()
    per_cell: dict[tuple[str, int], int] = {}
    for (c, t), n in strata.items():
        per_cell[(c, t)] = max(5, round(sample_size * n / total))
    # Rescale to hit sample_size exactly
    scale = sample_size / sum(per_cell.values())
    per_cell = {k: max(1, int(round(v * scale))) for k, v in per_cell.items()}

    samples: list[pd.DataFrame] = []
    for (c, t), n in per_cell.items():
        pool = all_df[(all_df["country"] == c) & (all_df["_tier_ord"] == t)]
        if pool.empty:
            continue
        take = min(n, len(pool))
        samples.append(pool.sample(n=take, random_state=seed))
    sampled = pd.concat(samples, ignore_index=True).head(sample_size)

    # Clean up for JSON: replace NaN with None, cast numerics
    import math
    sampled = sampled.drop(columns=["_tier_ord"])
    records = sampled.to_dict(orient="records")

    def _clean(v: Any) -> Any:
        if v is None:
            return None
        if isinstance(v, float) and math.isnan(v):
            return None
        if isinstance(v, str) and v.lower() in ("nan", ""):
            return None
        return v

    cleaned: list[dict[str, Any]] = []
    for rec in records:
        rec = {k: _clean(v) for k, v in rec.items()}
        for k in ("salary_min", "salary_mid", "salary_max", "experience_min_llm"):
            v = rec.get(k)
            if v is not None:
                try:
                    f = float(v)
                    rec[k] = None if math.isnan(f) else f
                except (TypeError, ValueError):
                    rec[k] = None
        for k in ("id",):
            v = rec.get(k)
            if v is not None:
                try:
                    rec[k] = int(float(v))
                except (TypeError, ValueError):
                    pass
        cleaned.append(rec)
    records = cleaned

    log.info("jobs_sample: %d rows (stratified)", len(records))
    return records


# ---------------------------------------------------------------------------
# Row-level dataset (full ~45k postings for client-side analytics)
# ---------------------------------------------------------------------------
ROWS_CLUSTER_KEYS = [
    "cluster_architecture__methods",
    "cluster_bi__analytics",
    "cluster_backend_development",
    "cluster_certifications",
    "cluster_cloud_computing",
    "cluster_data_analysis__stats",
    "cluster_data_engineering",
    "cluster_data_science__ml",
    "cluster_databases__storage",
    "cluster_devops__containers",
    "cluster_dynamic__web",
    "cluster_enterprise__managed",
    "cluster_enterprise_platforms",
    "cluster_frontend_development",
    "cluster_generative_ai",
    "cluster_legacy__mainframe",
    "cluster_mobile__desktop",
    "cluster_networking",
    "cluster_os__embedded",
    "cluster_scripting__shell",
    "cluster_security__identity",
    "cluster_systems_programming",
    "cluster_testing_qa__debugging",
    "cluster_tools__editors",
]

ROWS_CORE_COLUMNS = [
    "id", "country", "job_title", "company", "city", "state",
    "job_family", "skill_cluster",
    "desc_tier_llm", "is_real_ai",
    "salary_min", "salary_mid", "salary_max", "pay_currency",
    "edu_level_det", "experience_min_llm",
    "industry", "sector", "size",
    "hardskills",
]


def _tier_ord(val: Any) -> int | None:
    """Map desc_tier_llm strings to ordinal. Returns None for unknown."""
    if val is None:
        return None
    s = str(val).strip().lower()
    if s in ("", "nan"):
        return None
    if s == "none":
        return 0
    if s == "ai_integration":
        return 1
    if s in ("applied_ai", "core_ai"):
        return 2
    return None


def _seniority_band(exp: Any) -> str | None:
    """Bucket experience_min_llm into Junior / Mid / Senior / Lead+."""
    try:
        y = float(exp)
    except (TypeError, ValueError):
        return None
    import math
    if math.isnan(y):
        return None
    if y < 2:
        return "Junior"
    if y < 5:
        return "Mid"
    if y < 8:
        return "Senior"
    return "Lead+"


def _size_band(size: Any) -> str | None:
    if size is None:
        return None
    s = str(size).strip()
    if s in ("", "nan", "Unknown"):
        return None
    if "1 to 50" in s or "51 to 200" in s:
        return "1-200"
    if "201 to 500" in s or "501 to 1000" in s:
        return "201-1000"
    if "1001 to 5000" in s or "5001 to 10000" in s:
        return "1001-10000"
    if "10000+" in s:
        return "10000+"
    # German/Indian variants
    if "bis 50" in s or "51 bis 200" in s or "bis 200" in s:
        return "1-200"
    if "201 bis" in s or "501 bis" in s:
        return "201-1000"
    if "1.001" in s or "1001" in s or "5.001" in s or "5001" in s:
        return "1001-10000"
    if "10.000" in s.lower() or "10000" in s:
        return "10000+"
    return None


def build_rows() -> list[dict[str, Any]]:
    """Full row-level dataset for client-side analytics (~45k rows)."""
    import math

    frames: list[pd.DataFrame] = []
    sources = [
        (DATA_OUT / "us_relevant_ai_stata.csv", "US"),
        (DATA_OUT / "de" / "de_relevant_ai_stata.csv", "DE"),
        (DATA_OUT / "in_relevant_ai_stata.csv", "IN"),
    ]
    wanted_cols = ROWS_CORE_COLUMNS + ROWS_CLUSTER_KEYS

    for src, country in sources:
        if not src.exists():
            log.warning("rows: missing %s — skipping", src)
            continue
        df = pd.read_csv(src, sep=";", encoding="utf-8-sig", low_memory=False)
        for col in wanted_cols:
            if col not in df.columns:
                df[col] = None
        df = df[wanted_cols].copy()
        df["country"] = country
        frames.append(df)

    if not frames:
        raise FileNotFoundError("rows: no stata CSVs found")

    all_df = pd.concat(frames, ignore_index=True)
    log.info("rows: loaded %d rows from %d files", len(all_df), len(frames))

    def _clean_scalar(v: Any) -> Any:
        if v is None:
            return None
        if isinstance(v, float) and math.isnan(v):
            return None
        if isinstance(v, str):
            s = v.strip()
            if s == "" or s.lower() == "nan":
                return None
            return s
        return v

    def _num(v: Any) -> float | None:
        v = _clean_scalar(v)
        if v is None:
            return None
        try:
            f = float(v)
            if math.isnan(f):
                return None
            return f
        except (TypeError, ValueError):
            return None

    def _int(v: Any) -> int | None:
        f = _num(v)
        return int(f) if f is not None else None

    records: list[dict[str, Any]] = []
    for _, r in all_df.iterrows():
        rec: dict[str, Any] = {
            "id": _int(r["id"]),
            "co": str(r["country"]),  # compact keys reduce JSON size
            "jt": _clean_scalar(r["job_title"]),
            "cp": _clean_scalar(r["company"]),
            "ct": _clean_scalar(r["city"]),
            "st": _clean_scalar(r["state"]),
            "jf": _clean_scalar(r["job_family"]),
            "sc": _clean_scalar(r["skill_cluster"]),
            "t": _tier_ord(r["desc_tier_llm"]),
            "ai": _int(r["is_real_ai"]),
            "sn": _num(r["salary_min"]),
            "sm": _num(r["salary_mid"]),
            "sx": _num(r["salary_max"]),
            "cur": _clean_scalar(r["pay_currency"]),
            "ed": _clean_scalar(r["edu_level_det"]),
            "ex": _num(r["experience_min_llm"]),
            "sen": _seniority_band(r["experience_min_llm"]),
            "in": _clean_scalar(r["industry"]),
            "se": _clean_scalar(r["sector"]),
            "sz": _size_band(r["size"]),
            "hs": _clean_scalar(r["hardskills"]),
        }
        # Pack cluster flags as a bit mask (24 bits fits in a regular number)
        mask = 0
        for i, key in enumerate(ROWS_CLUSTER_KEYS):
            val = r.get(key)
            try:
                if val is not None and not (isinstance(val, float) and math.isnan(val)) and int(float(val)) == 1:
                    mask |= 1 << i
            except (TypeError, ValueError):
                pass
        rec["cl"] = mask
        records.append(rec)

    log.info("rows: %d cleaned records", len(records))
    return records


def build_row_meta() -> dict[str, Any]:
    """Metadata describing the compact row-level schema."""
    return {
        "cluster_keys": ROWS_CLUSTER_KEYS,
        "cluster_labels": [_nice_cluster(k) for k in ROWS_CLUSTER_KEYS],
        "tier_order": AI_LEVEL_ORDER,
        "key_map": {
            "id": "id", "co": "country", "jt": "job_title", "cp": "company",
            "ct": "city", "st": "state", "jf": "job_family", "sc": "skill_cluster",
            "t": "tier_ord", "ai": "is_real_ai",
            "sn": "salary_min", "sm": "salary_mid", "sx": "salary_max",
            "cur": "pay_currency", "ed": "edu_level", "ex": "experience_min",
            "sen": "seniority_band", "in": "industry", "se": "sector", "sz": "size_band",
            "hs": "hardskills", "cl": "cluster_mask",
        },
        "seniority_order": ["Junior", "Mid", "Senior", "Lead+"],
        "size_order": ["1-200", "201-1000", "1001-10000", "10000+"],
    }


# ---------------------------------------------------------------------------
# Orchestrator
# ---------------------------------------------------------------------------
def write_json(path: Path, data: Any, *, minify: bool = False) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as f:
        if minify:
            json.dump(data, f, ensure_ascii=False, separators=(",", ":"), allow_nan=False)
        else:
            json.dump(data, f, ensure_ascii=False, indent=2, allow_nan=False)
    size_kb = path.stat().st_size / 1024
    log.info("wrote %s (%.1f KB)", path.relative_to(EXPLORER_ROOT), size_kb)


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--run", type=Path, default=None, help="Stata run dir")
    parser.add_argument("--sample-size", type=int, default=500)
    args = parser.parse_args()

    run_dir = args.run or latest_run_dir()
    log.info("Using Stata run: %s", run_dir.name)

    CHARTS_DATA.mkdir(parents=True, exist_ok=True)

    g1 = build_g1(run_dir)
    g2 = build_g2(run_dir)
    g3 = build_g3(run_dir)
    g4 = build_g4(run_dir)
    g5 = build_g5(run_dir)
    g6 = build_g6(run_dir)
    g7 = build_g7(run_dir)

    write_json(CHARTS_DATA / "g1_ai_tier_by_country.json", g1)
    write_json(CHARTS_DATA / "g2_ai_share_by_jobfamily.json", g2)
    write_json(CHARTS_DATA / "g3_logit_ame_us.json", g3)
    write_json(CHARTS_DATA / "g4_mlogit_us.json", g4)
    write_json(CHARTS_DATA / "g5_logit_ame_by_country.json", g5)
    write_json(CHARTS_DATA / "g6_premium_decomposition.json", g6)
    write_json(CHARTS_DATA / "g7_premium_by_country.json", g7)

    kpi = build_kpi(g1, g7)
    clusters = build_clusters()
    jobfamilies = build_jobfamilies(g2)
    jobs_sample = build_jobs_sample(sample_size=args.sample_size)

    write_json(PUBLIC_DATA / "kpi.json", kpi)
    write_json(PUBLIC_DATA / "clusters.json", clusters)
    write_json(PUBLIC_DATA / "jobfamilies.json", jobfamilies)
    write_json(PUBLIC_DATA / "jobs_sample.json", jobs_sample)

    # V2: full row-level dataset for client-side analytics
    rows = build_rows()
    row_meta = build_row_meta()
    write_json(PUBLIC_DATA / "rows.json", rows, minify=True)
    write_json(PUBLIC_DATA / "rows_meta.json", row_meta)

    metadata = {
        "run_dir": run_dir.name,
        "generated_at": datetime.now(timezone.utc).isoformat(),
        "counts": {
            "clusters": len(clusters),
            "job_families": len(set(r["job_family"] for r in jobfamilies)),
            "jobs_sample": len(jobs_sample),
            "countries": len(g1),
            "rows_total": len(rows),
        },
        "cluster_labels": CLUSTER_LABELS,
        "country_labels": COUNTRY_LABELS,
        "ai_level_order": AI_LEVEL_ORDER,
    }
    write_json(PUBLIC_DATA / "metadata.json", metadata)

    log.info("Done.")
    return 0


if __name__ == "__main__":
    sys.exit(main())
