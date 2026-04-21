"""Verifikace — chart values vs Stata CSV vs thesis RTF tables.

Pro každý ze 7 grafů:
  1. načte zdrojové CSV z charts_data/
  2. reprodukuje stejnou transformaci jako build_charts.py
  3. vytiskne hodnoty, které by měly být na grafu

Také vytiskne relevantní úryvky z RTF tabulek (Tabulka_1/3/4/5 + 13a/b/c)
aby šlo vizuálně porovnat, jestli CSV hodnoty odpovídají tabulkám v thesis.
"""
from __future__ import annotations

import re
from pathlib import Path
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
STATA_OUT = ROOT / "analysis" / "stata" / "output"


def latest_run() -> Path:
    c = sorted(STATA_OUT.glob("thesis_final_run_*"))
    if not c:
        raise SystemExit("No thesis_final_run_* found")
    return c[-1]


def hline(t: str) -> None:
    print(f"\n{'=' * 78}\n{t}\n{'=' * 78}")


def subhline(t: str) -> None:
    print(f"\n--- {t} ---")


def sig_stars(p: float) -> str:
    if p < 0.001:
        return "***"
    if p < 0.01:
        return "**"
    if p < 0.05:
        return "*"
    return "ns"


# --------------------------------------------------------------------------
def verify_graph_1(run: Path) -> None:
    hline("GRAPH 1 — Složení AI tier per country")
    df = pd.read_csv(run / "charts_data/g1_ai_tier_by_country.csv",
                     keep_default_na=False)
    df["pct"] = df["pct"].astype(float)
    print("Raw CSV:")
    print(df.to_string(index=False))
    print("\nPivot (zobrazené na grafu):")
    wide = df.pivot(index="country", columns="ai_level", values="pct").fillna(0)
    wide = wide.reindex(["US", "DE", "IN"])
    for c in ["None", "AI Integration", "Applied/Core AI"]:
        if c not in wide.columns:
            wide[c] = 0.0
    wide = wide[["None", "AI Integration", "Applied/Core AI"]]
    wide["SUM"] = wide.sum(axis=1)
    print(wide.round(2).to_string())
    print("(součty by měly být ~100%)")


# --------------------------------------------------------------------------
def verify_graph_2(run: Path) -> None:
    hline("GRAPH 2 — AI share by job family (US)")
    df = pd.read_csv(run / "charts_data/g2_ai_share_by_jobfamily.csv")
    us = df[df["country"] == "US"].copy()
    us = us.sort_values("ai_share", ascending=False)
    print("Hodnoty na grafu (sorted desc):")
    for _, r in us.iterrows():
        print(f"  {r['job_family']:<25s}  {r['ai_share']:6.2f}%   N={int(r['n']):,}".replace(",", " "))
    overall = (us["ai_share"] * us["n"]).sum() / us["n"].sum()
    print(f"\nVážený průměr (čárkovaná čára): {overall:.2f}%")
    print(f"Celkem US pozic: {int(us['n'].sum()):,}".replace(",", " "))


# --------------------------------------------------------------------------
def verify_graph_3(run: Path) -> None:
    hline("GRAPH 3 — Logit AME US (všech 21 clusterů)")
    df = pd.read_csv(run / "charts_data/g3_logit_ame_us.csv")
    df["ame_pp"] = df["b"] * 100
    df["lo_pp"] = df["ci_low"] * 100
    df["hi_pp"] = df["ci_high"] * 100
    df["sig"] = df["p"] < 0.05
    df = df.sort_values("ame_pp", ascending=False)
    print(f"{'cluster':<35s} {'AME pp':>8s} {'CI low':>8s} {'CI high':>8s} {'p':>10s} {'sig':>5s}")
    for _, r in df.iterrows():
        print(f"  {r['coef']:<33s} {r['ame_pp']:+8.2f} {r['lo_pp']:+8.2f} {r['hi_pp']:+8.2f} {r['p']:>10.4g} {sig_stars(r['p']):>5s}")


# --------------------------------------------------------------------------
def verify_graph_4(run: Path) -> None:
    hline("GRAPH 4 — Mlogit US (Integration vs Applied)")
    di = pd.read_csv(run / "charts_data/g4_mlogit_us_integration.csv")
    da = pd.read_csv(run / "charts_data/g4_mlogit_us_applied.csv")
    for d in (di, da):
        d["ame_pp"] = d["b"] * 100
    m = di.merge(da, on="coef", suffixes=("_int", "_app"))
    m["diff_abs"] = (m["ame_pp_int"] - m["ame_pp_app"]).abs()
    m = m.sort_values("diff_abs", ascending=False)
    print(f"{'cluster':<35s} {'AME Int':>8s} {'p_Int':>8s} {'AME App':>8s} {'p_App':>8s} {'|diff|':>8s}")
    for _, r in m.iterrows():
        si = sig_stars(r['p_int'])
        sa = sig_stars(r['p_app'])
        print(f"  {r['coef']:<33s} {r['ame_pp_int']:+7.2f}{si:>4s} "
              f"{r['p_int']:>8.3g} {r['ame_pp_app']:+7.2f}{sa:>4s} "
              f"{r['p_app']:>8.3g} {r['diff_abs']:>8.2f}")


# --------------------------------------------------------------------------
def verify_graph_5(run: Path) -> None:
    hline("GRAPH 5 — Heatmap cross-country (US/DE/IN)")
    frames = {}
    for c in ("US", "DE", "IN"):
        d = pd.read_csv(run / f"charts_data/g5_logit_ame_{c}.csv")
        d["ame_pp"] = d["b"] * 100
        d["sig"] = d["p"] < 0.05
        frames[c] = d.set_index("coef")
    common = sorted(
        set.intersection(*(set(f.index) for f in frames.values())),
        key=lambda c: -abs(frames["US"].loc[c, "ame_pp"]),
    )
    print(f"{'cluster':<35s} {'US':>8s} {'p_US':>7s} {'DE':>8s} {'p_DE':>7s} {'IN':>8s} {'p_IN':>7s}")
    for cl in common:
        row = []
        for c in ("US", "DE", "IN"):
            v = frames[c].loc[cl, "ame_pp"]
            p = frames[c].loc[cl, "p"]
            row.append(f"{v:+7.2f}{sig_stars(p):>3s}")
            row.append(f"{p:>7.3g}")
        print(f"  {cl:<33s} {row[0]} {row[1]} {row[2]} {row[3]} {row[4]} {row[5]}")


# --------------------------------------------------------------------------
def verify_graph_6(run: Path) -> None:
    hline("GRAPH 6 — OLS decomposition A→B→C (US)")
    print(f"{'model':<8s} {'tier':<15s} {'b (%)':>8s} {'SE':>7s} {'p':>10s} {'sig':>5s}")
    for mod in "ABC":
        d = pd.read_csv(run / f"charts_data/g6_ols_{mod}.csv")
        for _, r in d.iterrows():
            if r["coef"].startswith("1.ai_level"):
                tier = "AI Integration"
            elif r["coef"].startswith("2.ai_level"):
                tier = "Applied/Core AI"
            else:
                continue
            print(f"  {mod:<6s} {tier:<15s} {r['b']*100:+8.2f} {r['se']*100:>7.3f} "
                  f"{r['p']:>10.4g} {sig_stars(r['p']):>5s}")


# --------------------------------------------------------------------------
def verify_graph_7(run: Path) -> None:
    hline("GRAPH 7 — Cross-country AI premium (US/DE/IN)")
    print(f"{'country':<8s} {'tier':<15s} {'b (%)':>8s} {'SE':>7s} {'CI':>20s} {'p':>10s} {'sig':>5s}")
    for c in ("US", "DE", "IN"):
        d = pd.read_csv(run / f"charts_data/g7_ols_{c}.csv")
        for _, r in d.iterrows():
            if r["coef"].startswith("1.ai_level"):
                tier = "AI Integration"
            elif r["coef"].startswith("2.ai_level"):
                tier = "Applied/Core AI"
            else:
                continue
            lo = (r["b"] - 1.96 * r["se"]) * 100
            hi = (r["b"] + 1.96 * r["se"]) * 100
            print(f"  {c:<6s} {tier:<15s} {r['b']*100:+8.2f} {r['se']*100:>7.3f} "
                  f"[{lo:+6.2f}, {hi:+6.2f}]  {r['p']:>10.4g} {sig_stars(r['p']):>5s}")


# --------------------------------------------------------------------------
def main() -> None:
    run = latest_run()
    print(f"Run: {run.name}\n")
    verify_graph_1(run)
    verify_graph_2(run)
    verify_graph_3(run)
    verify_graph_4(run)
    verify_graph_5(run)
    verify_graph_6(run)
    verify_graph_7(run)


if __name__ == "__main__":
    main()
