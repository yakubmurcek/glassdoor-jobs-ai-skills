"""Build 7 thesis charts from CSV data exported by Stata section 14.

Usage:
    uv run python analysis/charts/build_charts.py [--run RUN_DIR]

By default finds the latest thesis_final_run_* directory under
analysis/stata/output/. Writes PNG files to analysis/charts/output/.
"""
from __future__ import annotations

import argparse
from dataclasses import dataclass
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from matplotlib.patches import Patch, Rectangle

# ---------------------------------------------------------------------------
# Paths & config
# ---------------------------------------------------------------------------
ROOT = Path(__file__).resolve().parents[2]
STATA_OUT = ROOT / "analysis" / "stata" / "output"
CHARTS_OUT = ROOT / "analysis" / "charts" / "output"
CHARTS_OUT.mkdir(parents=True, exist_ok=True)

plt.rcParams.update({
    "figure.dpi": 140,
    "font.size": 9,
    "font.family": "sans-serif",
    # Jemnější šrafování pro ns — tenčí linie, šedivá místo černé
    "hatch.linewidth": 0.6,
    "hatch.color": "#6a6a6a",
})

COLOR_POS = "#3c6ea8"
COLOR_NEG = "#b84a4a"
COLOR_INT = "#3c6ea8"
COLOR_APP = "#b84a4a"
COLOR_NS = "#cfd4dc"
COLOR_NEUTRAL = "#4e6b8a"

HATCH_NS = "//"
HATCH_NS_HEATMAP = "////"
ALPHA_NS = 0.5
ALPHA_SIG = 0.95
EDGE_COLOR = "black"
EDGE_WIDTH = 0.4


# ---------------------------------------------------------------------------
# Label maps
# ---------------------------------------------------------------------------
CLUSTER_LABELS = {
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

AI_LEVEL_LABELS = {0: "None", 1: "AI Integration", 2: "Applied/Core AI"}
COUNTRY_LABELS = {"US": "USA", "DE": "Německo", "IN": "Indie"}


def nice_cluster(coef: str) -> str:
    return CLUSTER_LABELS.get(coef, coef.replace("cluster_", "").replace("_", " ").title())


def sig_stars(p: float) -> str:
    if p < 0.001:
        return "***"
    if p < 0.01:
        return "**"
    if p < 0.05:
        return "*"
    return "ns"


def latest_run_dir(base: Path) -> Path:
    candidates = sorted(base.glob("thesis_final_run_*"))
    if not candidates:
        raise FileNotFoundError(f"No thesis_final_run_* directories in {base}")
    return candidates[-1]


def load_csv(run_dir: Path, name: str) -> pd.DataFrame:
    path = run_dir / "charts_data" / name
    if not path.exists():
        raise FileNotFoundError(f"Missing {path} — did Stata section 14 run?")
    return pd.read_csv(path)


# ---------------------------------------------------------------------------
# Graph 1 — AI tier composition per country (stacked bar)
# ---------------------------------------------------------------------------
def graph_1(run_dir: Path) -> None:
    path = run_dir / "charts_data" / "g1_ai_tier_by_country.csv"
    if not path.exists():
        raise FileNotFoundError(f"Missing {path} — did Stata section 14 run?")
    # keep_default_na=False: Stata zapsala "None" jako string, jinak pandas
    # to převede na NaN a ztratíme název sloupce.
    df = pd.read_csv(path, keep_default_na=False)
    df["pct"] = df["pct"].astype(float)
    wide = df.pivot(index="country", columns="ai_level", values="pct").fillna(0)
    wide = wide.reindex(["US", "DE", "IN"])
    countries = [COUNTRY_LABELS[c] for c in wide.index]

    fig, ax = plt.subplots(figsize=(6.2, 3.6))
    none = wide["None"].values
    integ = wide["AI Integration"].values
    appl = wide["Applied/Core AI"].values

    ax.bar(countries, none, label="None", color="#cfd8dc",
           edgecolor=EDGE_COLOR, linewidth=EDGE_WIDTH)
    ax.bar(countries, integ, bottom=none, label="AI Integration", color=COLOR_INT,
           edgecolor=EDGE_COLOR, linewidth=EDGE_WIDTH)
    ax.bar(countries, appl, bottom=none + integ, label="Applied/Core AI", color=COLOR_APP,
           edgecolor=EDGE_COLOR, linewidth=EDGE_WIDTH)

    for i in range(len(countries)):
        # "None" popisek uprostřed svého segmentu
        ax.text(i, none[i] / 2, f"{none[i]:.1f}%",
                ha="center", va="center", color="#555", fontsize=8)
        # Segmenty AI vždy dovnitř (i pro malé segmenty u Indie) — konzistence
        if integ[i] > 0.3:
            ax.text(i, none[i] + integ[i] / 2, f"{integ[i]:.1f}%",
                    ha="center", va="center", color="white", fontsize=7.5)
        if appl[i] > 0.3:
            ax.text(i, none[i] + integ[i] + appl[i] / 2, f"{appl[i]:.1f}%",
                    ha="center", va="center", color="white", fontsize=7.5)

    ax.set_ylabel("Podíl inzerátů (%)")
    ax.set_ylim(0, 101)
    ax.set_title("Složení AI požadavků podle země")
    ax.legend(loc="upper center", bbox_to_anchor=(0.5, -0.08),
              ncol=3, fontsize=8, frameon=False)
    plt.tight_layout()
    plt.savefig(CHARTS_OUT / "01_ai_tier_by_country.png", bbox_inches="tight")
    plt.close()


# ---------------------------------------------------------------------------
# Graph 2 — AI share by job family (US)
# ---------------------------------------------------------------------------
def graph_2(run_dir: Path) -> None:
    df = load_csv(run_dir, "g2_ai_share_by_jobfamily.csv")
    us = df[df["country"] == "US"].copy()
    us = us.sort_values("ai_share", ascending=False)
    overall = (us["ai_share"] * us["n"]).sum() / us["n"].sum()

    fig, ax = plt.subplots(figsize=(6.8, 3.6))
    y = np.arange(len(us))
    ax.barh(y, us["ai_share"], color=COLOR_NEUTRAL,
            edgecolor=EDGE_COLOR, linewidth=EDGE_WIDTH)
    ax.set_yticks(y)
    ax.set_yticklabels(us["job_family"])
    ax.invert_yaxis()
    ax.axvline(overall, color="gray", linestyle="--", linewidth=1)
    # rozšířit plot area nahoru, ať se popisek "průměr" vejde mimo sloupce
    ax.set_ylim(len(us) - 0.5, -1.2)  # po invert_yaxis je to "nahoru o 1.2"
    ax.text(overall + 0.8, -0.85, f"průměr {overall:.1f}%",
            color="gray", fontsize=8, va="center", ha="left")
    for i, (share, n) in enumerate(zip(us["ai_share"], us["n"])):
        ax.text(share + 0.5, i, f"{share:.1f}%  (N={int(n):,})".replace(",", " "),
                va="center", fontsize=8)
    ax.set_xlabel("Podíl pozic s AI požadavkem (%)")
    ax.set_title("AI požadavek podle profesní skupiny (US)")
    ax.set_xlim(0, max(us["ai_share"]) * 1.25)
    plt.tight_layout()
    plt.savefig(CHARTS_OUT / "02_ai_share_by_jobfamily_us.png")
    plt.close()


# ---------------------------------------------------------------------------
# Graph 3 — Logit US AME forest plot
# ---------------------------------------------------------------------------
def graph_3(run_dir: Path) -> None:
    df = load_csv(run_dir, "g3_logit_ame_us.csv")
    df = df.copy()
    df["ame_pp"] = df["b"] * 100
    df["lo_pp"] = df["ci_low"] * 100
    df["hi_pp"] = df["ci_high"] * 100
    df["label"] = df["coef"].map(nice_cluster)
    df["sig"] = df["p"] < 0.05
    df = df.sort_values("ame_pp")

    fig, ax = plt.subplots(figsize=(7.2, 6.2))
    y = np.arange(len(df))
    for i, (sig, v) in enumerate(zip(df["sig"].values, df["ame_pp"].values)):
        if not sig:
            color, hatch, alpha = COLOR_NS, HATCH_NS, ALPHA_NS
        elif v < 0:
            color, hatch, alpha = COLOR_NEG, "", ALPHA_SIG
        else:
            color, hatch, alpha = COLOR_POS, "", ALPHA_SIG
        ax.barh(i, v, color=color, hatch=hatch, alpha=alpha,
                edgecolor=EDGE_COLOR, linewidth=EDGE_WIDTH)
    ax.errorbar(df["ame_pp"], y,
                xerr=[df["ame_pp"] - df["lo_pp"], df["hi_pp"] - df["ame_pp"]],
                fmt="none", ecolor="black", elinewidth=0.7, capsize=2)
    ax.set_yticks(y)
    ax.set_yticklabels(df["label"])
    ax.axvline(0, color="black", linewidth=0.8)
    ax.set_xlabel("AME (procentní body)")
    ax.set_title("Binární logit: vliv skill clusterů na P(AI) — USA\n(chybová úsečka = 95% CI; šrafované = nesignifikantní)")
    legend = [
        Patch(facecolor=COLOR_POS, edgecolor=EDGE_COLOR, label="Pozitivní (p < 0,05)"),
        Patch(facecolor=COLOR_NEG, edgecolor=EDGE_COLOR, label="Negativní (p < 0,05)"),
        Patch(facecolor=COLOR_NS, edgecolor=EDGE_COLOR, hatch=HATCH_NS,
              alpha=ALPHA_NS, label="Nesignifikantní"),
    ]
    ax.legend(handles=legend, loc="lower right", fontsize=8, framealpha=0.95)
    plt.tight_layout()
    plt.savefig(CHARTS_OUT / "03_logit_ame_us.png")
    plt.close()


# ---------------------------------------------------------------------------
# Graph 4 — Mlogit US: Integration vs Applied (dumbbell)
# ---------------------------------------------------------------------------
def graph_4(run_dir: Path) -> None:
    dint = load_csv(run_dir, "g4_mlogit_us_integration.csv").copy()
    dapp = load_csv(run_dir, "g4_mlogit_us_applied.csv").copy()
    for d in (dint, dapp):
        d["ame_pp"] = d["b"] * 100
        d["sig"] = d["p"] < 0.05
        d["label"] = d["coef"].map(nice_cluster)
    merged = dint.merge(dapp, on=["coef", "label"], suffixes=("_int", "_app"))
    # Sort by absolute difference between the two AMEs (largest discriminators first)
    merged["diff"] = (merged["ame_pp_int"] - merged["ame_pp_app"]).abs()
    merged = merged.sort_values("diff", ascending=True)

    fig, ax = plt.subplots(figsize=(7.4, 6.4))
    y = np.arange(len(merged))
    for i, row in enumerate(merged.itertuples()):
        both_ns = (not row.sig_int) and (not row.sig_app)
        ax.plot([row.ame_pp_int, row.ame_pp_app], [i, i],
                color="gray",
                linewidth=0.8 if both_ns else 1.1,
                linestyle=":" if both_ns else "-",
                alpha=ALPHA_NS if both_ns else 0.9,
                zorder=1)
    # AI Integration
    for i, row in enumerate(merged.itertuples()):
        facecolor = COLOR_INT if row.sig_int else "white"
        ax.scatter(row.ame_pp_int, i, s=55, color=facecolor,
                   edgecolor=COLOR_INT, linewidth=1.5, zorder=3)
    # Applied
    for i, row in enumerate(merged.itertuples()):
        facecolor = COLOR_APP if row.sig_app else "white"
        ax.scatter(row.ame_pp_app, i, s=55, color=facecolor,
                   edgecolor=COLOR_APP, linewidth=1.5, zorder=3)

    ax.set_yticks(y)
    ax.set_yticklabels(merged["label"])
    ax.axvline(0, color="black", linewidth=0.8)
    ax.set_xlabel("AME (procentní body)")
    ax.set_title("Mlogit USA: AI Integration vs Applied/Core AI\n(prázdný kroužek = nesignifikantní; seřazeno podle |rozdílu|)")
    legend = [
        plt.Line2D([0], [0], marker="o", color="w", markerfacecolor=COLOR_INT,
                   markeredgecolor=COLOR_INT, markersize=9, label="AI Integration (signif)"),
        plt.Line2D([0], [0], marker="o", color="w", markerfacecolor=COLOR_APP,
                   markeredgecolor=COLOR_APP, markersize=9, label="Applied/Core AI (signif)"),
        plt.Line2D([0], [0], marker="o", color="w", markerfacecolor="white",
                   markeredgecolor="gray", markersize=9, label="Nesignifikantní"),
    ]
    ax.legend(handles=legend, loc="lower right", fontsize=8, framealpha=0.95)
    plt.tight_layout()
    plt.savefig(CHARTS_OUT / "04_mlogit_dumbbell_us.png")
    plt.close()


# ---------------------------------------------------------------------------
# Graph 5 — Heatmap cross-country (variant B — hatched = ns)
# ---------------------------------------------------------------------------
def graph_5(run_dir: Path) -> None:
    frames = {}
    for c in ("US", "DE", "IN"):
        d = load_csv(run_dir, f"g5_logit_ame_{c}.csv").copy()
        d["ame_pp"] = d["b"] * 100
        d["sig"] = d["p"] < 0.05
        d["label"] = d["coef"].map(nice_cluster)
        frames[c] = d[["coef", "label", "ame_pp", "sig"]].set_index("coef")

    common = sorted(set.intersection(*(set(df.index) for df in frames.values())),
                    key=lambda c: -abs(frames["US"].loc[c, "ame_pp"]))

    data = np.array([[frames[c].loc[cl, "ame_pp"] for c in ("US", "DE", "IN")] for cl in common])
    sig = np.array([[frames[c].loc[cl, "sig"] for c in ("US", "DE", "IN")] for cl in common])
    labels = [frames["US"].loc[cl, "label"] for cl in common]
    cols = [COUNTRY_LABELS[c] for c in ("US", "DE", "IN")]

    vmax = float(max(abs(data.min()), abs(data.max())))

    fig, ax = plt.subplots(figsize=(5.4, 7.4))
    im = ax.imshow(data, cmap="RdBu", vmin=-vmax, vmax=vmax, aspect="auto")

    # Hatching for ns cells — jemná šedá
    for i in range(len(labels)):
        for j in range(len(cols)):
            if not sig[i, j]:
                ax.add_patch(Rectangle((j - 0.5, i - 0.5), 1, 1,
                                       fill=False, hatch=HATCH_NS_HEATMAP,
                                       edgecolor="#6a6a6a", linewidth=0, alpha=0.45))

    ax.set_xticks(range(len(cols)))
    ax.set_xticklabels(cols)
    ax.set_yticks(range(len(labels)))
    ax.set_yticklabels(labels)

    for i in range(len(labels)):
        for j in range(len(cols)):
            v = data[i, j]
            ax.text(j, i, f"{v:+.1f}", ha="center", va="center",
                    color="white" if abs(v) > vmax * 0.55 else "black",
                    fontsize=7, fontweight="bold" if sig[i, j] else "normal")

    ax.set_title("AME skill clusterů napříč zeměmi\n(šrafované = nesignifikantní; p ≥ 0,05)", fontsize=10)
    plt.colorbar(im, ax=ax, shrink=0.75, label="AME (p.b.)")
    leg = [Patch(facecolor="white", edgecolor="black", hatch=HATCH_NS_HEATMAP, label="ns")]
    ax.legend(handles=leg, loc="upper center", bbox_to_anchor=(0.5, -0.07),
              fontsize=8, frameon=False)
    plt.tight_layout()
    plt.savefig(CHARTS_OUT / "05_heatmap_crosscountry.png")
    plt.close()


# ---------------------------------------------------------------------------
# Graph 6 — OLS decomposition A → B → C (US)
# ---------------------------------------------------------------------------
def graph_6(run_dir: Path) -> None:
    def load(model: str) -> dict:
        d = load_csv(run_dir, f"g6_ols_{model}.csv")
        # find 1.ai_level and 2.ai_level rows
        result = {}
        for _, row in d.iterrows():
            if row["coef"].startswith("1.ai_level"):
                result["integ"] = (row["b"] * 100, row["se"] * 100, row["p"])
            elif row["coef"].startswith("2.ai_level"):
                result["app"] = (row["b"] * 100, row["se"] * 100, row["p"])
        return result

    models = {"A": load("A"), "B": load("B"), "C": load("C")}
    labels = ["Model A\n(firemní profil)", "Model B\n(+ lidský kapitál)", "Model C\n(+ skill clusters + job family)"]
    integ = [models[m]["integ"][0] for m in "ABC"]
    integ_err = [1.96 * models[m]["integ"][1] for m in "ABC"]
    integ_p = [models[m]["integ"][2] for m in "ABC"]
    app = [models[m]["app"][0] for m in "ABC"]
    app_err = [1.96 * models[m]["app"][1] for m in "ABC"]
    app_p = [models[m]["app"][2] for m in "ABC"]

    fig, ax = plt.subplots(figsize=(6.6, 3.8))
    x = np.arange(3)
    w = 0.35
    for i in range(3):
        sig = integ_p[i] < 0.05
        ax.bar(x[i] - w/2, integ[i], w, yerr=integ_err[i],
               color=COLOR_INT,
               alpha=ALPHA_SIG if sig else ALPHA_NS,
               hatch="" if sig else HATCH_NS,
               edgecolor=EDGE_COLOR, linewidth=EDGE_WIDTH, capsize=4,
               label="AI Integration" if i == 0 else None)
    for i in range(3):
        sig = app_p[i] < 0.05
        ax.bar(x[i] + w/2, app[i], w, yerr=app_err[i],
               color=COLOR_APP,
               alpha=ALPHA_SIG if sig else ALPHA_NS,
               hatch="" if sig else HATCH_NS,
               edgecolor=EDGE_COLOR, linewidth=EDGE_WIDTH, capsize=4,
               label="Applied/Core AI" if i == 0 else None)
    ax.plot(x - w/2, integ, "o-", color="#1e4a80", linewidth=1, markersize=4, zorder=3)
    ax.plot(x + w/2, app, "o-", color="#802a2a", linewidth=1, markersize=4, zorder=3)

    for i, (v, p) in enumerate(zip(integ, integ_p)):
        ax.text(i - w/2, v + integ_err[i] + 0.3,
                f"{v:.1f}% {sig_stars(p)}", ha="center", fontsize=8)
    for i, (v, p) in enumerate(zip(app, app_p)):
        ax.text(i + w/2, v + app_err[i] + 0.3,
                f"{v:.1f}% {sig_stars(p)}", ha="center", fontsize=8)

    ax.set_xticks(x)
    ax.set_xticklabels(labels)
    ax.set_ylabel("AI mzdová prémie (%)")
    ax.set_ylim(0, max(app) * 1.35)
    ax.set_title("Dekompozice AI prémie: Model A → B → C (USA)\n(chybová úsečka = 95% CI; šrafované = nesignifikantní)")
    ax.legend(fontsize=8)
    plt.tight_layout()
    plt.savefig(CHARTS_OUT / "06_premium_decomposition.png")
    plt.close()


# ---------------------------------------------------------------------------
# Graph 7 — Cross-country AI premium (bar with CI, hatched = ns)
# ---------------------------------------------------------------------------
def graph_7(run_dir: Path) -> None:
    def load(c: str) -> dict:
        d = load_csv(run_dir, f"g7_ols_{c}.csv")
        result = {}
        for _, row in d.iterrows():
            if row["coef"].startswith("1.ai_level"):
                result["integ"] = (row["b"] * 100, row["se"] * 100, row["p"])
            elif row["coef"].startswith("2.ai_level"):
                result["app"] = (row["b"] * 100, row["se"] * 100, row["p"])
        return result

    data = {c: load(c) for c in ("US", "DE", "IN")}
    countries = [COUNTRY_LABELS[c] for c in ("US", "DE", "IN")]
    integ = [data[c]["integ"][0] for c in ("US", "DE", "IN")]
    integ_err = [1.96 * data[c]["integ"][1] for c in ("US", "DE", "IN")]
    integ_p = [data[c]["integ"][2] for c in ("US", "DE", "IN")]
    app = [data[c]["app"][0] for c in ("US", "DE", "IN")]
    app_err = [1.96 * data[c]["app"][1] for c in ("US", "DE", "IN")]
    app_p = [data[c]["app"][2] for c in ("US", "DE", "IN")]

    fig, ax = plt.subplots(figsize=(6.4, 3.8))
    x = np.arange(3)
    w = 0.35
    for i, (v, e, p) in enumerate(zip(integ, integ_err, integ_p)):
        sig = p < 0.05
        ax.bar(i - w/2, v, w, yerr=e, color=COLOR_INT,
               alpha=ALPHA_SIG if sig else ALPHA_NS,
               hatch="" if sig else HATCH_NS,
               edgecolor=EDGE_COLOR, linewidth=EDGE_WIDTH, capsize=4)
    for i, (v, e, p) in enumerate(zip(app, app_err, app_p)):
        sig = p < 0.05
        ax.bar(i + w/2, v, w, yerr=e, color=COLOR_APP,
               alpha=ALPHA_SIG if sig else ALPHA_NS,
               hatch="" if sig else HATCH_NS,
               edgecolor=EDGE_COLOR, linewidth=EDGE_WIDTH, capsize=4)
    for i, (v, p) in enumerate(zip(integ, integ_p)):
        ax.text(i - w/2, v + integ_err[i] + 0.3,
                f"{v:.1f}% {sig_stars(p)}", ha="center", fontsize=8)
    for i, (v, p) in enumerate(zip(app, app_p)):
        ax.text(i + w/2, v + app_err[i] + 0.3,
                f"{v:.1f}% {sig_stars(p)}", ha="center", fontsize=8)

    ax.set_xticks(x)
    ax.set_xticklabels(countries)
    ax.axhline(0, color="black", linewidth=0.6)
    ax.set_ylabel("AI mzdová prémie (%)")
    ax.set_title("AI mzdová prémie napříč zeměmi\n(šrafované = nesignifikantní; chybová úsečka = 95% CI)")
    legend = [
        Patch(facecolor=COLOR_INT, edgecolor=EDGE_COLOR, label="AI Integration"),
        Patch(facecolor=COLOR_APP, edgecolor=EDGE_COLOR, label="Applied/Core AI"),
        Patch(facecolor="white", edgecolor=EDGE_COLOR, hatch=HATCH_NS,
              alpha=ALPHA_NS, label="Nesignifikantní"),
    ]
    # větší prostor nahoře pro popisky nad CI a legendu pod grafem
    ymax = max([v + e for v, e in zip(integ, integ_err)] + [v + e for v, e in zip(app, app_err)])
    ax.set_ylim(top=ymax * 1.28)
    ax.legend(handles=legend, fontsize=8,
              loc="upper center", bbox_to_anchor=(0.5, -0.12),
              ncol=3, frameon=False)
    plt.tight_layout()
    plt.savefig(CHARTS_OUT / "07_premium_crosscountry.png", bbox_inches="tight")
    plt.close()


# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------
def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--run", type=Path, default=None,
                    help="Specific thesis_final_run_* directory")
    args = ap.parse_args()
    run_dir = args.run or latest_run_dir(STATA_OUT)
    print(f"Using run: {run_dir}")

    builders = [graph_1, graph_2, graph_3, graph_4, graph_5, graph_6, graph_7]
    for fn in builders:
        try:
            fn(run_dir)
            print(f"  OK {fn.__name__}")
        except FileNotFoundError as e:
            print(f"  SKIP {fn.__name__}: {e}")
        except Exception as e:
            print(f"  ERR  {fn.__name__}: {e}")
            raise

    print(f"\nGrafy ulozeny v: {CHARTS_OUT}")


if __name__ == "__main__":
    main()
